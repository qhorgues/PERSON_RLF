"""Flower ClientApp: wraps LitTBPS + a client's local data partition.

  1. set_parameters(global_subspace)  -> inject the shared subspace received from the server
  2. reload local (non-shared) params from ClientStateStore (None on round 1)
  3. Trainer.fit for `local_epochs` epochs (2-stage policy if stn.enabled)
  4. persist local params back to ClientStateStore
  5. return (updated subspace, |D^k|, metrics)
"""

from typing import Any, Dict, Tuple

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


# Persistent per-process runtime cache. With mono-GPU (num_gpus=1.0) a single Ray
# actor process is reused across every round/client, so building the heavy LitTBPS
# once and reusing it avoids re-allocating GPU memory (and reloading the backbone
# checkpoint) on every round — the dominant source of mid-run CUDA allocations. Each
# Ray actor imports this module independently, so the cache is naturally per-process.
_RUNTIME: Dict[str, Any] = {}


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

        self._init_local_snapshot = None
        self.datamodule = self._build_local_datamodule()
        self.model = self._get_or_build_model()

    def _memory_cfg(self):
        """`federated.memory` sub-config (empty dict if absent)."""
        return self.config.federated.get("memory", {})

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
        from utils.gpu_mem import apply_memory_fraction

        apply_memory_fraction(self._memory_cfg().get("cuda_memory_fraction", None), self.device)
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

    def _get_or_build_model(self) -> LitTBPS:
        """Return the per-process LitTBPS, building it once when `reuse_model` is set.

        The model architecture is identical across clients — `num_classes` is global
        (computed from the full train set before partitioning, `lightning_data.py`) —
        so one instance serves every client: `set_parameters` injects the shared
        subspace and the client's personalized (local) params at the start of each
        round. Only the dataset-size-dependent scalars are refreshed per client.
        """
        if not self._memory_cfg().get("reuse_model", True):
            return self._build_local_model()

        if "model" not in _RUNTIME:
            model = self._build_local_model()
            _RUNTIME["model"] = model
            # W0 local (non-shared) snapshot, restored for a client's very first round
            # so the reused model does not inherit the previously-served client's
            # personalized params (see `set_parameters`).
            _RUNTIME["init_local"] = self.selector.extract_local(model)
            _RUNTIME["warmed_up"] = False
            logger.info(
                f"[Client {self.client_id}] built persistent per-process model "
                "(reused across all rounds/clients)"
            )

        model = _RUNTIME["model"]
        self._init_local_snapshot = _RUNTIME["init_local"]
        self._sync_per_client_scalars(model)
        return model

    def _sync_per_client_scalars(self, model: LitTBPS) -> None:
        """Refresh the cached model's dataset-size-dependent scalars for this client;
        they drive the LR scheduler (`configure_optimizers`) and boosting."""
        train_loader = self.datamodule.train_dataloader()
        accum = self.config.trainer.get("accumulate_grad_batches", 1)
        model.num_iters_per_epoch = max(1, len(train_loader) // accum)
        model.train_set_length = len(self.datamodule.train_set)

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
        elif self._init_local_snapshot is not None:
            # Persistent model reused across clients: reset the non-shared params to
            # the W0 snapshot so this client's first round starts from the shared init
            # rather than the previously-served client's personalized params.
            self.selector.inject_local(self.model, self._init_local_snapshot)

    def _maybe_warmup(self) -> None:
        """Pre-reserve the peak GPU footprint once per process (config-gated, off by default).

        Runs one throwaway training step (materializes activations + grads + optimizer
        state) plus one eval pass (reserves the feature-bank / similarity peak), then
        restores the original weights so training is not perturbed. Because
        `empty_cache` is disabled, the reserved pool is retained for the real rounds —
        so no further cudaMalloc is needed once the rounds begin, and any OOM surfaces
        at t=0 instead of mid-run.
        """
        if not self._memory_cfg().get("warmup", False):
            return
        if _RUNTIME.get("warmed_up", False):
            return
        if not torch.cuda.is_available():
            _RUNTIME["warmed_up"] = True
            return

        from utils.gpu_mem import format_memory, reset_peak_stats

        logger.info(f"[Client {self.client_id}] warm-up: pre-reserving peak GPU memory")
        # Snapshot weights + buffers on CPU so the throwaway step does not change them.
        snapshot = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        try:
            warm_trainer = L.Trainer(
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=0,
                num_sanity_val_steps=0,
                accelerator=self.config.trainer.get("accelerator", "auto"),
                devices=1,
                precision=self.config.trainer.get("precision", "32-true"),
                accumulate_grad_batches=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            warm_trainer.fit(self.model, train_dataloaders=self.datamodule.train_dataloader())
            eval_trainer = self._build_trainer(0)
            eval_trainer.validate(self.model, dataloaders=self.datamodule.val_dataloader())
        except Exception as exc:  # warm-up is best-effort; never abort a run for it
            logger.warning(f"[Client {self.client_id}] warm-up skipped: {exc}")
        finally:
            self.model.load_state_dict(snapshot)
            self.model.global_subspace = None
            self.model.proximal_mu = 0.0
            _RUNTIME["warmed_up"] = True
        logger.info(f"[Client {self.client_id}] warm-up done — {format_memory(self.device)}")
        reset_peak_stats(self.device)

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

    def _run_local_fit(self, fit_callable):
        """Run the local fit, optionally retrying-and-waiting on CUDA OOM (config-gated).

        On a GPU shared with other jobs, a round can hit OOM simply because a co-tenant
        is momentarily peaking. When `federated.memory.oom_retry.enabled`, we wait for
        memory to free up and retry instead of failing the round. Disabled => fail fast
        (the server-side `_AbortOnFrozenGlobal` guard then surfaces the OOM).
        """
        oom_cfg = self._memory_cfg().get("oom_retry", {}) or {}
        if not oom_cfg.get("enabled", False):
            return fit_callable()
        from utils.gpu_mem import run_with_oom_retry

        return run_with_oom_retry(
            fit_callable,
            wait_seconds=float(oom_cfg.get("wait_seconds", 30.0)),
            max_retries=oom_cfg.get("max_retries", None),
            device=self.device,
            label=f"[Client {self.client_id}]",
        )

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        logger.info(f"[Client {self.client_id}] round {server_round}: fit on {self.num_examples} samples")

        self._maybe_warmup()

        mu = float(config.get("proximal_mu", 0.0))

        def _prepare_and_fit():
            # Re-establish the round-start state on every (re)try so a retry after a
            # partial/failed step trains from the same weights, not half-updated ones.
            self.set_parameters(parameters)
            self._apply_two_stage_policy(server_round)
            if mu > 0:
                self._set_fedprox(parameters, mu)
            trainer = self._build_trainer(server_round)
            trainer.fit(self.model, train_dataloaders=self.datamodule.train_dataloader())
            return trainer

        trainer = self._run_local_fit(_prepare_and_fit)

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

        if self._memory_cfg().get("log_gpu_memory", True):
            from utils.gpu_mem import format_memory, memory_stats_gib

            logger.info(
                f"[Client {self.client_id}] round {server_round} — "
                f"{format_memory(self.device)}"
            )
            mem = memory_stats_gib(self.device)
            if mem:
                metrics["gpu_reserved_gib"] = mem["reserved"]
                metrics["gpu_max_reserved_gib"] = mem["max_reserved"]

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
