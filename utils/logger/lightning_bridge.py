"""LightningLoggerBridge: adapts the `Logger` façade to the
`lightning.pytorch.loggers.Logger` contract.

Attached to `L.Trainer(logger=bridge)`, it forwards all of `LitTBPS`'s
`self.log(...)` / `self.log_dict(...)` metrics (losses, total_loss, grad_norm,
R@K, val_score) to the façade -> fan-out (console/csv/wandb/plot), without
modifying `lightning_models.py`.

`finalize()` does NOT close the façade (a run may chain fit -> test); closing is
explicit via `log.close()` at the end of the entrypoint.
"""

from __future__ import annotations

from typing import Any, Optional

from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .interface import LoggerInterface


class LightningLoggerBridge(LightningLogger):
    def __init__(self, facade: LoggerInterface, save_dir: str, name: Optional[str] = None):
        super().__init__()
        self._facade = facade
        self._save_dir = save_dir
        self._name = name or ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return ""

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        self._facade.log_metrics(metrics, step)

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        try:
            params = dict(params)
        except (TypeError, ValueError):
            params = {"hparams": params}
        self._facade.log_hyperparams(params)

    # --- gateways used by trainer.py (delegation to the sinks) ----------
    @rank_zero_only
    def log_table(self, key: str, columns: Any = None, data: Any = None, **kwargs: Any) -> None:
        self._facade.log_table(key, columns, data)

    @rank_zero_only
    def watch(self, model: Any, log: str = "all", *args: Any, **kwargs: Any) -> None:
        self._facade.watch(model, log)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # does not close the façade: closing is explicit via log.close()
        pass
