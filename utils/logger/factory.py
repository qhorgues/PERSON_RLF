"""`build_logger(config)`: builds the global `log` façade from the Hydra config
(`config.logger.sinks`), installs interception of third-party logs, and returns
the façade.

The `log` façade is a *stable singleton*: `build_logger` mutates its sinks
in-place (via `set_sinks`) so that every `from utils.logger import log` stays
valid even if imported before the call.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from .interface import INFO, LoggerInterface, level_to_int
from .intercept import install_interception
from .sinks.console import ConsoleSink
from .sinks.csv import CsvSink
from .sinks.file import FileSink


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Lenient access over DictConfig / dict / None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _enabled(sinks_cfg: Any, name: str) -> bool:
    return bool(_get(_get(sinks_cfg, name), "enabled", False))


def _options(sink_cfg: Any) -> dict:
    """Free-form `options:` sub-block of a sink, as a plain dict (forwarded
    uniformly to every sink so new tuning knobs need no factory change)."""
    return _to_plain_dict(_get(sink_cfg, "options")) or {}


def _hydra_run_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.output_dir
    except Exception:
        return os.getcwd()


def _to_plain_dict(cfg: Any) -> Optional[dict]:
    if cfg is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    return dict(cfg) if hasattr(cfg, "keys") else None


def build_sinks(
    config: Any,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> List[LoggerInterface]:
    logger_cfg = _get(config, "logger")
    sinks_cfg = _get(logger_cfg, "sinks")
    run_dir = save_dir or _hydra_run_dir()
    sinks: List[LoggerInterface] = []

    # Back-compat: no `sinks` block -> console INFO only.
    if sinks_cfg is None:
        sinks.append(ConsoleSink(level=INFO))
        return sinks

    # --- messages -------------------------------------------------------
    if _enabled(sinks_cfg, "console"):
        c = _get(sinks_cfg, "console")
        sinks.append(
            ConsoleSink(level=level_to_int(_get(c, "level", "INFO")), options=_options(c))
        )

    if _enabled(sinks_cfg, "file"):
        f = _get(sinks_cfg, "file")
        sinks.append(
            FileSink(
                os.path.join(run_dir, _get(f, "filename", "train.log")),
                level=level_to_int(_get(f, "level", "DEBUG")),
                max_mb=_get(f, "max_mb", 10),
                backups=_get(f, "backups", 5),
                options=_options(f),
            )
        )

    # --- metrics --------------------------------------------------------
    if _enabled(sinks_cfg, "csv"):
        cs = _get(sinks_cfg, "csv")
        sinks.append(
            CsvSink(
                os.path.join(run_dir, _get(cs, "filename", "metrics.csv")),
                options=_options(cs),
            )
        )

    if _enabled(sinks_cfg, "wandb"):
        w = _get(sinks_cfg, "wandb")
        from .sinks.wandb import WandbSink

        sinks.append(
            WandbSink(
                project=_get(w, "project", None),
                entity=_get(w, "entity", None),
                group=_get(w, "group", None),
                mode=_get(w, "mode", "online"),
                watch_mode=_get(w, "watch", "all"),
                name=experiment_name,
                save_dir=run_dir,
                config=_to_plain_dict(config),
                options=_options(w),
            )
        )

    if _enabled(sinks_cfg, "tensorboard"):
        tb = _get(sinks_cfg, "tensorboard")
        from .sinks.tensorboard import TensorBoardSink

        sinks.append(
            TensorBoardSink(save_dir=run_dir, name=experiment_name, options=_options(tb))
        )

    if _enabled(sinks_cfg, "plot"):
        p = _get(sinks_cfg, "plot")
        from .sinks.plot import RealtimePlotSink

        sinks.append(
            RealtimePlotSink(
                out_path=os.path.join(run_dir, _get(p, "filename", "fl_results.png")),
                every_n=_get(p, "every_n", 1),
                title=_get(p, "title", None),
                options=_options(p),
            )
        )

    if not sinks:
        sinks.append(ConsoleSink(level=INFO))
    return sinks


def build_logger(
    config: Any,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
):
    """(Re)configure the global `log` façade and install interception."""
    from . import log  # stable singleton defined in __init__

    log.set_sinks(build_sinks(config, experiment_name=experiment_name, save_dir=save_dir))
    log.setup()

    third_party = _to_plain_dict(_get(_get(config, "logger"), "third_party")) or {}
    default_level = third_party.pop("default", "INFO")
    install_interception(log, third_party, default_level=default_level)
    return log
