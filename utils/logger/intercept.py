"""Redirect *third-party* logs to the `Logger` façade.

Captures, via a handler placed on the stdlib root logger:
  - Flower / Ray (driver), HF transformers, and the modules that still use
    `logging.getLogger` (data/*, model/build);
  - the `warnings` module (`logging.captureWarnings`).

The system's private loggers (the `INTERNAL_LOGGER_PREFIX` namespace, derived
from this package's path) are ignored to avoid any loop with the `FileSink`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .interface import INFO, INTERNAL_LOGGER_PREFIX, LoggerInterface, level_to_int

_INTERNAL_PREFIX = INTERNAL_LOGGER_PREFIX

# Libraries that are especially noisy at DEBUG/INFO: WARNING floor by default
# (overridable via `third_party` in the config if needed).
_DEFAULT_QUIET = {
    "matplotlib": "WARNING",
    "PIL": "WARNING",
    "urllib3": "WARNING",
    "filelock": "WARNING",
    "fsspec": "WARNING",
    "h5py": "WARNING",
}


class InterceptHandler(logging.Handler):
    """Stdlib handler that re-emits each record into the façade."""

    def __init__(self, facade: LoggerInterface):
        super().__init__(level=0)
        self._facade = facade

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        caller = f"{record.name}:{record.lineno}"
        self._facade.log(record.levelno, message, caller=caller)


def install_interception(
    facade: LoggerInterface,
    third_party: Optional[Dict[str, str]] = None,
    default_level=INFO,
) -> None:
    root = logging.getLogger()
    root.handlers = [InterceptHandler(facade)]
    # Floor level for *intercepted* logs (stdlib). Does NOT affect application
    # DEBUG, which goes straight from the façade to the sinks without stdlib.
    root.setLevel(level_to_int(default_level))

    # Existing named loggers (flwr, ray, transformers, urllib3...): clear their
    # handlers so they propagate to the root (hence to our handler).
    for name in list(logging.root.manager.loggerDict.keys()):
        if name.startswith(_INTERNAL_PREFIX):
            continue  # our private loggers (FileSink): leave them alone
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # per-logger levels: anti-noise defaults + config overrides
    merged = {**_DEFAULT_QUIET, **(third_party or {})}
    for name, lvl in merged.items():
        logging.getLogger(name).setLevel(level_to_int(lvl))

    logging.captureWarnings(True)
