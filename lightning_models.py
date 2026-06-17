from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from lightning.pytorch.utilities import grad_norm
from prettytable import PrettyTable

from model.build import build_backbone_with_proper_layer_resize
from model.lora import get_lora_model
from model.tbps import TBPS
from solver import build_lr_scheduler, build_optimizer
from utils.metrics import rank, rank2


class DataType(Enum):
    """Enum for different types of data processing"""

    IMAGE = "image"
    TEXT = "text"


@dataclass
class ModelSample:
    """Data container for model samples"""

    pids: torch.Tensor
    images: Optional[torch.Tensor] = None
    caption_input_ids: Optional[torch.Tensor] = None
    caption_attention_mask: Optional[torch.Tensor] = None

    def to_device(self, device: torch.device) -> "ModelSample":
        """Move sample data to specified device"""
        self.pids = self.pids.to(device)
        if self.images is not None:
            self.images = self.images.to(device)
        if self.caption_input_ids is not None:
            self.caption_input_ids = self.caption_input_ids.to(device)
        if self.caption_attention_mask is not None:
            self.caption_attention_mask = self.caption_attention_mask.to(device)
        return self


@dataclass
class MetricsContainer:
    """Container for storing metrics data"""

    text_ids: List[torch.Tensor] = field(default_factory=list)
    image_ids: List[torch.Tensor] = field(default_factory=list)
    text_feats: List[torch.Tensor] = field(default_factory=list)
    image_feats: List[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        """Clear all stored metrics data"""
        self.text_ids.clear()
        self.image_ids.clear()
        self.text_feats.clear()
        self.image_feats.clear()

    def concatenate(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate all stored tensors"""
        return (
            torch.cat(self.text_ids),
            torch.cat(self.image_ids),
            torch.cat(self.text_feats),
            torch.cat(self.image_feats),
        )


class ModelException(Exception):
    """Custom exception for model-related errors"""

    pass


class LitTBPS(L.LightningModule):
    def __init__(
        self,
        config,
        vocab_size,
        pad_token_id,
        num_iters_per_epoch,
        train_set_length,
        num_classes=11003,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        # Initialize model components
        try:
            self._initialize_model(
                vocab_size, pad_token_id, num_classes, num_iters_per_epoch
            )
        except Exception as e:
            raise ModelException(f"Failed to initialize model: {str(e)}")

        # Initialize state
        self._initialize_state()
        self.num_epoch_for_boosting = self.config.backbone.num_epoch_for_boosting
        if self.num_epoch_for_boosting > 0:
            logger.info(
                f"Boosting weights will be calculated every {self.num_epoch_for_boosting} epochs"
            )
        self.weights = None
        self.train_set_length = train_set_length

        # FedProx (Phase 2) — neutral by default; set per-round by the Flower client
        # via `federated.fedprox.attach_fedprox_hook`. proximal_mu=0 -> no-op.
        self.proximal_mu = 0.0
        self.global_subspace = None
        self.subspace_keys = None

    ############# SETTING UP LORA ######################
    def setup_lora(self, lora_config: Dict) -> None:
        """Setup LORA for the model"""
        self.backbone = get_lora_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    ############# INITIALIZATION FUNCTIONS #############
    def _initialize_model(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_classes: int,
        num_iters_per_epoch: int,
    ) -> None:
        """Initialize model components and configuration"""
        self.num_iters_per_epoch = (
            num_iters_per_epoch // self.config.trainer.accumulate_grad_batches
        )

        # Build model components
        self.backbone = build_backbone_with_proper_layer_resize(self.config.backbone)
        self.model = TBPS(
            config=self.config,
            backbone=self.backbone,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_classes=num_classes,
        )

    def _initialize_state(self) -> None:
        """Initialize model state containers"""
        self.metrics_container = MetricsContainer()
        self.test_img_data: List[ModelSample] = []
        self.test_txt_data: List[ModelSample] = []
        self.test_final_outputs: List[Dict] = []

    ############# INFERENCE FUNCTIONS #############
    def get_image_features(
        self, image: torch.Tensor, return_last_hidden: bool = False
    ) -> torch.Tensor:
        """
        Get image features using the model

        Args:
            image: Input image tensor
            return_last_hidden: Whether to return last hidden state

        Returns:
            Image features tensor
        """
        try:
            return self.model.encode_image(image, return_last_hidden)
        except Exception as e:
            raise ModelException(f"Failed to extract image features: {str(e)}")

    def get_text_features(
        self, caption_input: Dict[str, torch.Tensor], return_last_hidden: bool = False
    ) -> torch.Tensor:
        """
        Get text features using the model

        Args:
            caption_input: Dictionary containing input_ids and attention_mask
            return_last_hidden: Whether to return last hidden state

        Returns:
            Text features tensor
        """
        try:
            return self.model.encode_text(caption_input, return_last_hidden)
        except Exception as e:
            raise ModelException(f"Failed to extract text features: {str(e)}")

    ############# END INFERENCE FUNCTIONS #############

    ############# FEDERATED LEARNING FUNCTIONS #############
    def get_subspace_state_dict(self, selector) -> List[np.ndarray]:
        """Extract the shared subspace (`SubspaceSelector.extract`) for Flower."""
        return selector.extract(self)

    def load_subspace_state_dict(self, selector, ndarrays: List[np.ndarray]) -> None:
        """Inject a shared subspace (`SubspaceSelector.inject`) received from Flower."""
        selector.inject(self, ndarrays)

    def freeze_reid(self) -> None:
        """Freeze the ReID backbone + heads (everything except the STN module).

        Used by the STN 2-stage policy (STNReID §III-B) so Stage 1 trains a strong STN
        on top of a frozen (weak) ReID. Must be called before `Trainer.fit` so the
        optimizer/scheduler are built over the right trainable set.
        """
        for name, param in self.model.named_parameters():
            if "stn" not in name:
                param.requires_grad = False

    def unfreeze_reid(self) -> None:
        """Unfreeze the ReID backbone + heads (Stage 2), honoring `backbone.freeze.*`."""
        freeze_vision = self.config.backbone.freeze.vision
        freeze_text = self.config.backbone.freeze.text
        for name, param in self.model.named_parameters():
            if "stn" in name:
                continue
            if "vision_model" in name:
                param.requires_grad = not freeze_vision
            elif "text_model" in name:
                param.requires_grad = not freeze_text
            else:
                param.requires_grad = True

    ############# END FEDERATED LEARNING FUNCTIONS #############

    ############# TRAINING FUNCTIONS #############
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step implementation

        Args:
            batch: Input batch dictionary
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        try:
            # Compute alpha for soft labels
            epoch = self.trainer.current_epoch
            alpha = self._compute_alpha(epoch)

            ret = self.model(batch, alpha, self.weights)
            loss = sum(v for k, v in ret.items() if k.endswith("loss"))

            # FedProx proximal term (Phase 2) over the shared subspace, if enabled.
            if self.proximal_mu > 0 and self.global_subspace is not None:
                from federated.fedprox import proximal_term

                prox = proximal_term(
                    self, self.global_subspace, self.subspace_keys, self.proximal_mu
                )
                loss = loss + prox
                self.log("prox_loss", prox, on_step=True, on_epoch=True, prog_bar=False)

            # Log metrics
            self._log_training_metrics(ret, alpha, loss, epoch, batch_idx)

            return loss

        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise ModelException(f"Training step failed: {str(e)}")

    def _compute_alpha(self, epoch: int) -> float:
        """Compute alpha value for soft labels"""
        alpha = self.config.loss.softlabel_ratio
        if epoch == 0:
            step = self.trainer.global_step
            alpha *= min(1.0, step / self.num_iters_per_epoch)
        return alpha

    def _log_training_metrics(
        self,
        ret: Dict[str, torch.Tensor],
        alpha: float,
        loss: torch.Tensor,
        epoch: int,
        batch_idx: int,
    ) -> None:
        """Log training metrics"""
        # Log individual losses
        self.log_dict(ret, on_step=True, on_epoch=True, prog_bar=False)
        self.log("alpha", alpha, on_step=True, on_epoch=True, prog_bar=False)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if epoch == 0 and batch_idx == 0:
            logger.info(f"Initial loss: {loss.item():.4f}")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config.optimizer, self.model)

        self.config.scheduler.n_iter_per_epoch = self.num_iters_per_epoch
        scheduler = build_lr_scheduler(self.config.scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        all_norms = norms[f"grad_{float(2)}_norm_total"]
        self.log("grad_norm", all_norms, on_step=True, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """End of training epoch"""
        if (
            self.num_epoch_for_boosting > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.num_epoch_for_boosting == 0
        ):
            self.calculate_boosting_weights()

    def calculate_boosting_weights(self):
        model = self.model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, ids = [], [], [], [], []
        for batch in self.trainer.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # images = batch['images'].to(device)
            # caption_ids = batch['caption_ids'].to(device)
            pid = batch["pids"]
            batch_ids = batch["id"]
            with torch.no_grad():
                img_feat, text_feat = model.forward2(batch)
            qids.append(pid.view(-1))
            qfeats.append(text_feat)
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
            ids.append(batch_ids.view(-1))
        qfeats = torch.cat(qfeats, 0)
        gfeats = torch.cat(gfeats, 0)

        # Concatenate all features
        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features
        similarity = qfeats @ gfeats.t()
        similarity = similarity.cpu()

        qids = torch.cat(qids, 0).cpu()
        gids = torch.cat(gids, 0).cpu()
        ids = torch.cat(ids, 0).cpu()

        _, _, c = rank2(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10)
        change = ids[c == 1].tolist()

        # Update weights
        weights = torch.ones(self.train_set_length)
        weights = weights.to(device)
        weights.requires_grad = False
        weights[change] = 1.6
        logger.info(f"Boosting weights for ids {change}, set to 1.6")

        # Cleanup
        del qids, gids, qfeats, gfeats, similarity, ids, change

    ############# END TRAINING FUNCTIONS #############

    ############ METRICS FUNCTIONS ############
    def _compute_metrics(
        self,
        return_ranking: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Compute evaluation metrics"""
        text_ids, image_ids, text_feats, image_feats = (
            self.metrics_container.concatenate()
        )

        # Compute similarities
        t2i_similarity = torch.matmul(text_feats, image_feats.t())
        i2t_similarity = t2i_similarity.t()

        # Calculate metrics for both directions
        t2i_metrics, t2i_ranking = self._compute_ranking_metrics(
            t2i_similarity, text_ids, image_ids, return_ranking
        )
        i2t_metrics, i2t_ranking = self._compute_ranking_metrics(
            i2t_similarity, image_ids, text_ids, return_ranking
        )

        return {
            "t2i": t2i_metrics,
            "i2t": i2t_metrics,
        }, {
            "t2i": t2i_ranking,
            "i2t": i2t_ranking,
        }

    @staticmethod
    def _compute_ranking_metrics(
        similarity: torch.Tensor,
        query_ids: torch.Tensor,
        gallery_ids: torch.Tensor,
        return_ranking: bool = True,
    ) -> Dict[str, float]:
        """Compute ranking metrics"""
        cmc, mAP, mINP, ranking = rank(
            similarity=similarity,
            q_pids=query_ids,
            g_pids=gallery_ids,
            max_rank=10,
            get_mAP=True,
        )

        return {
            "R1": cmc[0].item(),
            "R5": cmc[4].item(),
            "R10": cmc[9].item(),
            "mAP": mAP.item(),
            "mINP": mINP.item(),
        }, ranking if return_ranking else None

    def _log_metrics(self, results: Dict[str, Dict[str, float]], phase: str) -> None:
        """Log metrics results"""
        # Create results table
        table = PrettyTable(["Task", "R1", "R5", "R10", "mAP", "mINP"])

        # Add results and log metrics
        for task, metrics in results.items():
            # Add to table
            row = [task] + [f"{v:.2f}" for v in metrics.values()]
            table.add_row(row)

            # Log individual metrics
            for name, value in metrics.items():
                self.log(f"{phase}_{task}_{name}", value, on_epoch=True)

        # Log overall score
        self.log(f"{phase}_score", results["t2i"]["R1"], on_epoch=True, prog_bar=True)

        # Print table
        logger.info(f"\n{phase.capitalize()} Results:\n{table}")

    ############# END METRICS FUNCTIONS #############

    ############# VALIDATION FUNCTIONS #############
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step implementation"""
        try:
            self._process_features(batch, DataType.IMAGE)
            self._process_features(batch, DataType.TEXT)
        except Exception as e:
            logger.error(f"Error in validation step: {str(e)}")
            raise ModelException(f"Validation step failed: {str(e)}")

    def _process_features(
        self,
        batch: Dict[str, Any],
        data_type: DataType,
    ) -> None:
        """Process features for either image or text data"""
        if data_type == DataType.IMAGE and batch.get("img"):
            pid = batch["img"]["pids"]
            img_feat = self.model.encode_image(batch["img"]["images"])
            self.metrics_container.image_ids.append(pid.flatten())
            self.metrics_container.image_feats.append(img_feat)

        elif data_type == DataType.TEXT and batch.get("txt"):
            pid = batch["txt"]["pids"]
            caption_input = {
                "input_ids": batch["txt"]["caption_input_ids"],
                "attention_mask": batch["txt"]["caption_attention_mask"],
            }
            text_feat = self.model.encode_text(caption_input)
            self.metrics_container.text_ids.append(pid.flatten())
            self.metrics_container.text_feats.append(text_feat)

    def on_validation_epoch_start(self) -> None:
        """Initialize validation data containers"""
        self.metrics_container.clear()

    def on_validation_epoch_end(self) -> None:
        """Process validation results at epoch end and cleaning up"""
        try:
            results, _ = self._compute_metrics(return_ranking=False)
            self._log_metrics(results, "val")
            self.metrics_container.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in validation epoch end: {str(e)}")
            raise ModelException(f"Validation epoch end failed: {str(e)}")

    ############# TEST TIME RELATED FUNCTIONS #############
    def test_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int
    ) -> None:
        """Process test batch which is sequentialy built for image and text data
        Dataloader index is used to differentiate between image (0) and text data (1)"""
        try:
            if dataloader_idx == 0:
                self._process_test_image_batch(batch)
            else:
                self._process_test_text_batch(batch)
        except Exception as e:
            logger.error(f"Error in test step: {str(e)}")
            raise ModelException(f"Test step failed: {str(e)}")

    def _process_test_image_batch(self, batch: Dict[str, Any]) -> None:
        """Process test image batch"""
        # Store CPU data
        self.test_img_data.extend(
            [
                ModelSample(
                    pids=batch["pids"][i].cpu(),
                    images=batch["images"][i].cpu(),
                )
                for i in range(len(batch["images"]))
            ]
        )

        # Process features
        img_feat = self.model.encode_image(batch["images"])
        self.metrics_container.image_ids.append(batch["pids"].flatten())
        self.metrics_container.image_feats.append(img_feat)

    def _process_test_text_batch(self, batch: Dict[str, Any]) -> None:
        """Process test text batch"""
        # Store CPU data
        self.test_txt_data.extend(
            [
                ModelSample(
                    pids=batch["pids"][i].cpu(),
                    caption_input_ids=batch["caption_input_ids"][i].cpu(),
                    caption_attention_mask=batch["caption_attention_mask"][i].cpu(),
                )
                for i in range(len(batch["caption_input_ids"]))
            ]
        )

        # Process features
        caption_inputs = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }
        text_feat = self.model.encode_text(caption_inputs)
        self.metrics_container.text_ids.append(batch["pids"].flatten())
        self.metrics_container.text_feats.append(text_feat)

    def _process_wrong_predictions(self, ranking: torch.Tensor) -> List[Dict]:
        """Process wrong predictions efficiently"""
        wrong_predictions = []

        # Find wrong predictions
        for query_idx, pred_ranking in enumerate(ranking):
            true_pid = self.test_txt_data[query_idx].pids.item()
            pred_pids = [
                self.test_img_data[idx].pids.item() for idx in pred_ranking[:10]
            ]

            # Check if the first prediction is correct
            if pred_pids[0] != true_pid:
                prediction = {
                    "query": self.test_txt_data[query_idx].caption_input_ids,
                    "predictions": [
                        {
                            "image": self.test_img_data[idx].images,
                            "pid": pid,
                        }
                        for idx, pid in zip(pred_ranking[:10], pred_pids)
                    ],
                }

                # Find correct image and pid
                correct_img = next(
                    (
                        sample
                        for sample in self.test_img_data
                        if sample.pids.item() == true_pid
                    ),
                    None,
                )
                if correct_img:
                    prediction["correct_img"] = {
                        "image": correct_img.images,
                        "pid": true_pid,
                    }

                wrong_predictions.append(prediction)

        return wrong_predictions

    def on_test_epoch_start(self) -> None:
        """Initialize test data containers"""
        self.test_img_data.clear()
        self.test_txt_data.clear()
        self.metrics_container.clear()

    def on_test_epoch_end(self) -> None:
        """Process test results"""
        try:
            # Compute metrics
            results, ranking = self._compute_metrics(return_ranking=True)
            self._log_metrics(results, "test")

            # Process wrong predictions for text-to-image ranking only
            self.t2i_ranking = ranking["t2i"]
            self.test_final_outputs = self._process_wrong_predictions(self.t2i_ranking)

            # Cleanup
            self.test_img_data.clear()
            self.test_txt_data.clear()
            self.metrics_container.clear()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in test epoch end: {str(e)}")
            raise ModelException(f"Test epoch end failed: {str(e)}")

    ############# END TEST TIME RELATED FUNCTIONS #############
