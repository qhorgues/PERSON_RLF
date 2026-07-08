"""FileSink: messages -> rotating file (in the Hydra run-dir).

Uses a *private* stdlib logger (`propagate=False`, under the
`INTERNAL_LOGGER_PREFIX` namespace ignored by the InterceptHandler) to get
rotation without looping back into the interception placed on the root.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

from ..interface import DEBUG, INTERNAL_LOGGER_PREFIX, LoggerInterface

_INSTANCE_COUNTER = 0


class FileSink(LoggerInterface):
    """Text sink for a rotating file. Only handles `log()` (messages)."""

    def __init__(
        self,
        path: str,
        level: int = DEBUG,
        max_mb: int = 10,
        backups: int = 5,
        options: Optional[dict] = None,
    ):
        global _INSTANCE_COUNTER
        super().__init__(level=level, options=options)
        self._path = path
        self._max_mb = int(max_mb)
        self._backups = int(backups)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        _INSTANCE_COUNTER += 1
        self._id = _INSTANCE_COUNTER
        self._logger = logging.getLogger(f"{INTERNAL_LOGGER_PREFIX}filesink.{self._id}")
        self._logger.setLevel(level)
        self._logger.propagate = False  # does not bubble up to the root (no loop)
        self._logger.handlers = [self._make_handler(path)]
        # Lazily-created private loggers for named outputs (`output=` -> <name>.log).
        self._named_loggers: dict[str, logging.Logger] = {}

    def _make_handler(self, path: str) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            path, maxBytes=self._max_mb * 1024 * 1024, backupCount=self._backups
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(caller)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        return handler

    def _named_logger(self, output: str) -> logging.Logger:
        """Return (creating on first use) a private logger writing to `<output>.log`
        next to the main file, with the same rotation and formatting."""
        logger = self._named_loggers.get(output)
        if logger is None:
            path = os.path.join(os.path.dirname(self._path) or ".", f"{output}.log")
            logger = logging.getLogger(f"{INTERNAL_LOGGER_PREFIX}filesink.{self._id}.{output}")
            logger.setLevel(self.level)
            logger.propagate = False
            logger.handlers = [self._make_handler(path)]
            self._named_loggers[output] = logger
        return logger

    def log(
        self,
        level: int,
        message: str,
        caller: Optional[str] = None,
        output: Optional[str] = None,
        **ctx: Any,
    ) -> None:
        extra = {"caller": caller or "-"}
        # Additive routing: always keep the full record in the main file...
        self._logger.log(level, message, extra=extra)
        # ...and, if a named output is given, also write to the dedicated file.
        if output:
            self._named_logger(output).log(level, message, extra=extra)

    def close(self) -> None:
        for logger in (self._logger, *self._named_loggers.values()):
            for handler in logger.handlers:
                try:
                    handler.close()
                except Exception:
                    pass
