"""LoggerInterface: common contract for the `Logger` façade and every sink.

Covers the project's two kinds of logging:
  - text messages (stdlib-style levels)            -> console / file
  - metrics & artifacts (scalars, tables, ...)     -> csv / wandb / plot

The base class provides **no-op-by-default** bodies: each sink overrides only
what concerns it (heterogeneous fan-out orchestrated by `Logger`).
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC
from typing import Any, Optional

# Levels = stdlib integers, to interoperate with the InterceptHandler.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Private stdlib-logger namespace, *derived from this package's dotted path* so
# it carries no hard-coded project name and cannot collide with the host project.
# `__name__` == "<pkg>.interface" -> "<pkg>._private." (e.g. "utils.logger._private.").
# Shared by FileSink (private logger names) and the interception layer (skip check),
# so the two can never drift and the folder stays droppable into any project.
INTERNAL_LOGGER_PREFIX = __name__.rsplit(".", 1)[0] + "._private."


def level_to_int(level: Any) -> int:
    """Accept an integer (10..50) or a name ('INFO', 'debug'...) -> integer."""
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    return value if isinstance(value, int) else INFO


class LoggerInterface(ABC):
    # minimum priority accepted for messages (metrics are not filtered)
    level: int = INFO

    def __init__(self, level: int = INFO, options: Optional[dict] = None):
        """Shared plumbing for every sink.

        `options`: free-form, sink-specific parameters coming straight from the
        config (e.g. matplotlib `dpi`/`figsize`/`rcParams` for the plot sink).
        The factory forwards them uniformly, so a sink can accept new tuning
        knobs without the factory (or this interface) knowing about them.
        """
        self.level = level
        self.options = dict(options or {})

    def _opt(self, key: str, default: Any = None) -> Any:
        """Read a free-form option (see `options`)."""
        return getattr(self, "options", {}).get(key, default)

    # --- messages -------------------------------------------------------
    def log(self, level: int, message: str, **ctx: Any) -> None:
        """Emit a text message. No-op by default.

        Recognized `ctx` keys (threaded transparently by the `Logger` façade,
        like `caller`):
          - `output`: optional name for a dedicated output. File-writing sinks
            interpret it as a filename (`<output>.log`); other sinks (console)
            render it as a named separator before the message.
        """

    # --- metrics / artifacts -------------------------------------------
    # `output` (optional): name of a dedicated output. File-writing sinks
    # (csv) interpret it as a filename (`<output>.csv`); key-based sinks
    # (wandb/tensorboard) namespace the keys (`<output>/<name>`).
    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        """Emit scalars (`{name: value}`). No-op by default."""

    def log_table(
        self, key: str, columns: Any, data: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        """Emit a table. No-op by default."""

    def log_image(
        self, key: str, image: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        """Emit an image. No-op by default."""

    def log_hyperparams(self, params: dict) -> None:
        """Emit hyperparameters. No-op by default."""

    def watch(self, model: Any, log: str = "all") -> None:
        """Track a model's gradients/weights (W&B-specific hook). No-op by default."""

    # --- lifecycle ------------------------------------------------------
    def setup(self) -> None:
        """Open resources (file, W&B run, figure...). No-op by default."""

    def close(self) -> None:
        """Flush / close resources. No-op by default."""

    # --- syntactic sugar (messages) ------------------------------------
    def debug(self, message: str, **ctx: Any) -> None:
        self.log(DEBUG, message, **ctx)

    def info(self, message: str, **ctx: Any) -> None:
        self.log(INFO, message, **ctx)

    def warning(self, message: str, **ctx: Any) -> None:
        self.log(WARNING, message, **ctx)

    def error(self, message: str, **ctx: Any) -> None:
        self.log(ERROR, message, **ctx)

    def critical(self, message: str, **ctx: Any) -> None:
        self.log(CRITICAL, message, **ctx)

    def exception(self, message: str, **ctx: Any) -> None:
        """ERROR + current traceback (equivalent to loguru/stdlib `.exception`)."""
        self.log(ERROR, f"{message}\n{traceback.format_exc()}", **ctx)
