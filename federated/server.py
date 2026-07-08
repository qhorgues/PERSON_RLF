"""Flower ServerApp: global LitTBPS reference model, initial subspace parameters,
and centralized evaluation on the full test set (FedSH RQ6 convergence tracking).
"""

from typing import Dict, Tuple

import torch
from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from utils.logger import log as logger
from prettytable import PrettyTable

from federated.aggregation import build_strategy
from federated.strategy_subspace import SubspaceSelector, build_subspace_selector
from lightning_models import LitTBPS
from utils.metrics import rank


def evaluate_global(model: LitTBPS, test_loader, device: torch.device) -> Dict[str, float]:
    """Centralized R1/R5/R10/mAP/mINP evaluation on the full (t2i) test set."""
    model.eval()

    image_ids, image_feats, text_ids, text_feats = [], [], [], []
    with torch.no_grad():
        for batch, _batch_idx, dataloader_idx in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if dataloader_idx == 0:
                image_ids.append(batch["pids"].flatten())
                image_feats.append(model.get_image_features(batch["images"]))
            else:
                caption_input = {
                    "input_ids": batch["caption_input_ids"],
                    "attention_mask": batch["caption_attention_mask"],
                }
                text_ids.append(batch["pids"].flatten())
                text_feats.append(model.get_text_features(caption_input))

    image_ids = torch.cat(image_ids)
    image_feats = torch.cat(image_feats)
    text_ids = torch.cat(text_ids)
    text_feats = torch.cat(text_feats)

    similarity = torch.matmul(text_feats, image_feats.t())
    cmc, mAP, mINP, _ = rank(similarity, text_ids, image_ids, max_rank=10, get_mAP=True)

    return {
        "R1": cmc[0].item(),
        "R5": cmc[4].item(),
        "R10": cmc[9].item(),
        "mAP": mAP.item(),
        "mINP": mINP.item(),
    }


class FederatedServer:
    def __init__(self, config, reference_datamodule):
        self.config = config
        self.datamodule = reference_datamodule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = self._build_global_model()
        self.selector = self.build_selector()

    def _build_global_model(self) -> LitTBPS:
        tokenizer = self.datamodule.tokenizer
        train_loader = self.datamodule.train_dataloader()
        model = LitTBPS(
            self.config,
            vocab_size=tokenizer.true_vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            num_iters_per_epoch=len(train_loader),
            train_set_length=len(self.datamodule.train_set),
            num_classes=self.datamodule.num_classes,
        )
        if self.config.get("lora", None):
            model.setup_lora(self.config.lora)
        return model.to(self.device)

    def build_selector(self) -> SubspaceSelector:
        return build_subspace_selector(self.config, self.global_model)

    def initial_parameters(self):
        """W0 = SubspaceSelector.extract(global_model), as Flower `Parameters`."""
        return ndarrays_to_parameters(self.global_model.get_subspace_state_dict(self.selector))

    def get_evaluate_fn(self):
        """`evaluate_fn(server_round, parameters, config)`: centralized global evaluation."""
        test_loader = self.datamodule.test_dataloader()
        num_shared = self.selector.num_shared_parameters()

        def evaluate(
            server_round: int, parameters: NDArrays, _config: Dict[str, Scalar]
        ) -> Tuple[float, Dict[str, Scalar]]:
            self.global_model.load_subspace_state_dict(self.selector, parameters)
            results = evaluate_global(self.global_model, test_loader, self.device)

            table = PrettyTable(["round", "R1", "R5", "R10", "mAP", "mINP"])
            table.add_row(
                [server_round] + [f"{results[k]:.2f}" for k in ("R1", "R5", "R10", "mAP", "mINP")]
            )
            logger.info(f"[Global t2i eval]\n{table}")

            loss = 100.0 - results["R1"]

            metrics: Dict[str, Scalar] = {f"global_{k}": v for k, v in results.items()}
            metrics["num_shared_parameters"] = num_shared
            metrics["global_loss"] = loss

            # fan-out vers la façade (csv / wandb / plot / console selon la config)
            logger.log_metrics(metrics, step=server_round)

            return loss, metrics

        return evaluate

    def make_server_app(self) -> ServerApp:
        initial_parameters = self.initial_parameters()
        evaluate_fn = self.get_evaluate_fn()
        num_rounds = self.config.federated.num_rounds
        config = self.config

        def server_fn(_context: Context) -> ServerAppComponents:
            strategy = build_strategy(config, initial_parameters, evaluate_fn)
            return ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=num_rounds)
            )

        return ServerApp(server_fn=server_fn)
