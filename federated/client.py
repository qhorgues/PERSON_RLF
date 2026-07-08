"""Flower ClientApp: wraps LitTBPS + a client's local data partition.

  1. set_parameters(global_subspace)  -> inject the shared subspace received from the server
  2. reload local (non-shared) params from ClientStateStore (None on round 1)
  3. Trainer.fit for `local_epochs` epochs (2-stage policy if stn.enabled)
  4. persist local params back to ClientStateStore
  5. return (updated subspace, |D^k|, metrics)
"""

from typing import Dict, Tuple

import lightning as L
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from utils.logger import log as logger
from omegaconf import OmegaConf

from federated.fedprox import attach_fedprox_hook
from federated.state import ClientStateStore
from federated.strategy_subspace import SubspaceSelector
from lightning_data import TBPSDataModule
from lightning_models import LitTBPS

# Ray simulation workers unpickle `client_fn` by importing this module directly
# (without running trainer_federated.py's module scope), so the `tuple`/`eval`
# OmegaConf resolvers used by the config (e.g. img_size, augmentation sizes)
# must be registered here too.
OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args), replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TBPSFlowerClient(NumPyClient):
    def __init__(
        self,
        client_id: int,
        config,
        client_samples,
        num_examples: int,
        selector: SubspaceSelector,
        state_store: ClientStateStore,
        device: torch.device,
    ):
        self.client_id = client_id
        self.config = config
        self.client_samples = client_samples
        self.num_examples = num_examples
        self.selector = selector
        self.state_store = state_store
        self.device = device

        self.datamodule = self._build_local_datamodule()
        self.model = self._build_local_model()

    def _build_local_datamodule(self) -> TBPSDataModule:
        dm = TBPSDataModule(
            self.config,
            client_id=self.client_id,
            num_clients=self.config.federated.num_clients,
            partition_samples=self.client_samples,
        )
        dm.setup()
        return dm

    def _build_local_model(self) -> LitTBPS:
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

    def _build_trainer(self, _server_round: int) -> L.Trainer:
        trainer_cfg = self.config.trainer
        return L.Trainer(
            max_epochs=self.config.federated.local_epochs,
            accelerator=trainer_cfg.get("accelerator", "auto"),
            devices=1,
            precision=trainer_cfg.get("precision", "32-true"),
            accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
            gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
            gradient_clip_algorithm=trainer_cfg.get("gradient_clip_algorithm"),
            limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
            limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
            limit_test_batches=trainer_cfg.get("limit_test_batches", 1.0),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

    ############# FLOWER NUMPYCLIENT API #############
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.model.get_subspace_state_dict(self.selector)

    def set_parameters(self, parameters: NDArrays) -> None:
        self.model.load_subspace_state_dict(self.selector, parameters)
        if self.state_store.exists(self.client_id):
            local_state = self.state_store.load(self.client_id)
            self.selector.inject_local(self.model, local_state)

    def _apply_two_stage_policy(self, server_round: int) -> None:
        """STN 2-stage freeze/unfreeze policy (STNReID §III-B). No-op while stn.enabled=false.

        Stage 1 (server_round in stage1_rounds): STN trainable, ReID backbone frozen.
        Stage 2 (otherwise): STN frozen, ReID backbone fine-tuned.
        """
        if not self.config.federated.stn.enabled:
            return
        stn_module = getattr(self.model.model, "stn_module", None)
        if stn_module is None:
            return

        stage1 = self.config.federated.stn.two_stage.stage1_rounds
        in_stage1 = stage1[0] <= server_round < stage1[1]
        if in_stage1:
            stn_module.unfreeze_stn()
            self.model.freeze_reid()
            logger.info(
                f"[Client {self.client_id}] round {server_round}: STN stage 1 "
                "(STN trainable, ReID frozen)"
            )
        else:
            stn_module.freeze_stn()
            self.model.unfreeze_reid()
            logger.info(
                f"[Client {self.client_id}] round {server_round}: STN stage 2 "
                "(STN frozen, ReID fine-tuned)"
            )

    def _set_fedprox(self, parameters: NDArrays, mu: float) -> None:
        """Anchor the FedProx proximal term to the received global subspace (Phase 2)."""
        attach_fedprox_hook(self.model, parameters, mu, self.selector.selected_keys())

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        logger.info(f"[Client {self.client_id}] round {server_round}: fit on {self.num_examples} samples")

        self.set_parameters(parameters)
        self._apply_two_stage_policy(server_round)

        mu = float(config.get("proximal_mu", 0.0))
        if mu > 0:
            self._set_fedprox(parameters, mu)

        trainer = self._build_trainer(server_round)
        trainer.fit(self.model, train_dataloaders=self.datamodule.train_dataloader())

        # Release the FedProx anchor (frees the cached global-subspace tensors).
        self.model.global_subspace = None
        self.model.proximal_mu = 0.0

        self.state_store.save(self.client_id, self.selector.extract_local(self.model))

        metrics = {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if v.numel() == 1
        }
        # client_id lets the server-side strategy fan-out these metrics to a
        # per-client output (utils.logger `output=`); see aggregation.py.
        metrics["client_id"] = self.client_id
        return self.get_parameters({}), self.num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        self.set_parameters(parameters)

        trainer = self._build_trainer(server_round)
        results = trainer.validate(self.model, dataloaders=self.datamodule.val_dataloader())
        metrics = {k: float(v) for k, v in (results[0] if results else {}).items()}

        loss = 100.0 - metrics.get("val_score", 0.0)
        metrics["client_id"] = self.client_id  # -> per-client output, server-side
        return loss, self.num_examples, metrics


def make_client_app(config, partitioner, selector: SubspaceSelector, state_store: ClientStateStore) -> ClientApp:
    """Build the ClientApp; `client_fn` is closed over (config, partitioner, selector, state_store)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        client_samples = partitioner.client_samples(partition_id)
        num_examples = partitioner.client_num_examples(partition_id)

        client = TBPSFlowerClient(
            client_id=partition_id,
            config=config,
            client_samples=client_samples,
            num_examples=num_examples,
            selector=selector,
            state_store=state_store,
            device=device,
        )
        return client.to_client()

    return ClientApp(client_fn=client_fn)
