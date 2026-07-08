"""Unified logging system — Composite `log` façade + configurable sinks.

Application usage:
    from utils.logger import log
    log.info("message")
    log.log_metrics({"loss": 0.1}, step=3)

    # Optional named output: file sinks write to `<name>.log`, the console
    # prefixes the block with a named separator.
    log.info(big_table_block, output="partition")

`log` is a stable singleton: `build_logger(config)` (called by `setup_logging`)
reconfigures its sinks in-place, so imports made beforehand stay valid.

Portability: only the generic core is imported eagerly (no hard dependency on
Hydra/Lightning and no reference to the host project name). The optional project
glue — `setup_logging`, `setup_checkpoint_callback`, `generate_experiment_name`,
`get_config_overrides` (Hydra) and `LightningLoggerBridge` (Lightning) — is
resolved lazily via `__getattr__`, so `import utils.logger` never pulls those
heavy deps, yet `from utils.logger import setup_logging` keeps working. Drop this
folder into any package and it works unchanged.
"""

from .interface import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    LoggerInterface,
)
from .core import Logger
from .sinks import ConsoleSink, FileSink

# Global façade. Default sinks (console INFO) until build_logger has run:
# guarantees logging works even very early during startup.
log = Logger([ConsoleSink(level=INFO)])

from .factory import build_logger, build_sinks  # noqa: E402  (depends on `log`)

# Optional project glue (Hydra / Lightning) — loaded on first access only, so the
# portable core stays free of those dependencies. Names stay importable as before.
_LAZY = {
    "LightningLoggerBridge": ("lightning_bridge", "LightningLoggerBridge"),
    "setup_logging": ("setup", "setup_logging"),
    "setup_checkpoint_callback": ("setup", "setup_checkpoint_callback"),
    "generate_experiment_name": ("setup", "generate_experiment_name"),
    "get_config_overrides": ("setup", "get_config_overrides"),
}


def __getattr__(name):  # PEP 562: lazy attribute resolution at module level
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{target[0]}", __name__)
    return getattr(module, target[1])


__all__ = [
    "log",
    "Logger",
    "LoggerInterface",
    "ConsoleSink",
    "FileSink",
    "build_logger",
    "build_sinks",
    "LightningLoggerBridge",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "setup_logging",
    "setup_checkpoint_callback",
    "generate_experiment_name",
    "get_config_overrides",
]
