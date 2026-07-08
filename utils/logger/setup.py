"""Hydra entry point of the logging system: builds the unified façade
(text + metrics) via `build_logger`, exposes it to Lightning through a
`LightningLoggerBridge`, and provides the checkpoint callback.

NOTE: this module is *optional project glue*, not part of the portable logging
core. It depends on Hydra + Lightning (and bundles a `ModelCheckpoint`, which is
orchestration rather than logging). It is imported lazily by the package
`__getattr__`, so `import utils.logger` never pulls Hydra/Lightning; a project
without them can still use the façade (`log`, sinks, `build_logger`).
"""

import os
from typing import Any, Dict, Optional, Tuple

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from .factory import build_logger


def get_config_overrides() -> Dict[str, Any]:
    """Extract and process config overrides from Hydra."""
    hydra_cfg = HydraConfig.get()
    overrides = {}

    if hasattr(hydra_cfg, "overrides") and getattr(hydra_cfg.overrides, "task", None):
        for override in hydra_cfg.overrides.task:
            if override[0] == "+":
                continue  # Skip the adding of the task
            key, value = override.split("=")
            try:
                # Try to evaluate the value if it's a number
                value = eval(value)
            except:
                pass
            overrides[key] = value

    return overrides


def generate_experiment_name(config_name: str, overrides: Dict[str, Any]) -> str:
    """Generate experiment name based on config overrides."""
    if not overrides:
        return f"{config_name}_base"

    # Sort overrides for consistent naming
    override_parts = []
    for key, value in sorted(overrides.items()):
        # Skip special Hydra keys and None values
        if key.startswith("hydra.") or value is None:
            continue
        # Handle nested configs
        if isinstance(value, (dict, DictConfig)):
            value = "custom"
        override_parts.append(f"{key}_{value}")

    return f"{config_name}_{'_'.join(override_parts)}"


def _resolve_experiment_name(
    config: DictConfig, experiment_name: Optional[str] = None
) -> str:
    if experiment_name is not None:
        return experiment_name
    if config.logger.experiment_name is not None:
        return config.logger.experiment_name
    config_name = HydraConfig.get().job.config_name
    return generate_experiment_name(config_name, get_config_overrides())


def setup_checkpoint_callback(config: DictConfig, log_dir: str) -> ModelCheckpoint:
    """Set up checkpoint callback."""
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    return ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=config.logger.checkpoint.filename,
        save_top_k=config.logger.checkpoint.save_top_k,
        monitor=config.logger.checkpoint.monitor,
        mode=config.logger.checkpoint.mode,
        save_last=config.logger.checkpoint.save_last,
    )


def setup_logging(
    config: DictConfig,
    experiment_name: Optional[str] = None,
) -> Tuple["LightningLoggerBridge", ModelCheckpoint]:  # noqa: F821
    """Set up logging and checkpoints.

    Builds the unified façade (text + metrics: console/file/csv/wandb/
    tensorboard/plot according to `config.logger.sinks`) + interception, then
    returns:
      - a `LightningLoggerBridge` (to pass to `L.Trainer(logger=...)`) that
        forwards `self.log(...)` metrics to the façade;
      - the checkpoint callback.
    """
    from . import log
    from .lightning_bridge import LightningLoggerBridge

    exp_name = _resolve_experiment_name(config, experiment_name)
    save_dir = HydraConfig.get().runtime.output_dir

    build_logger(config, experiment_name=exp_name, save_dir=save_dir)

    bridge = LightningLoggerBridge(log, save_dir=save_dir, name=exp_name)
    checkpoint_callback = setup_checkpoint_callback(config, save_dir)

    return bridge, checkpoint_callback
