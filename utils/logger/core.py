"""`Logger`: Composite façade of the logging system.

It "only" loops over the configured sinks:
  - messages  -> priority-filtered fan-out (`level >= sink.level`)
  - metrics   -> unconditional fan-out

`Logger` itself implements `LoggerInterface`: a `Logger` can therefore appear in
the sinks of another `Logger` (nesting / composite).
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

from .interface import LoggerInterface

_PKG_DIR = os.path.dirname(__file__)


def _caller() -> str:
    """Return `module:line` of the first frame outside the utils.logger package."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PKG_DIR):
            module = frame.f_globals.get("__name__", "?")
            return f"{module}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def _emergency(message: str) -> None:
    """Write to the real stderr (never re-intercepted) — a failing sink must not
    bring training down."""
    try:
        sys.__stderr__.write(f"[logger] {message}\n")
    except Exception:
        pass


class Logger(LoggerInterface):
    def __init__(self, sinks: Optional[List[LoggerInterface]] = None):
        self._sinks: List[LoggerInterface] = list(sinks) if sinks else []

    # --- sink management (build_logger mutates the list in-place) -------
    def set_sinks(self, sinks: List[LoggerInterface]) -> None:
        self._sinks = list(sinks)

    def add_sink(self, sink: LoggerInterface) -> None:
        self._sinks.append(sink)

    @property
    def sinks(self) -> List[LoggerInterface]:
        return self._sinks

    # --- messages: level-filtered fan-out ------------------------------
    def log(self, level: int, message: str, **ctx: Any) -> None:
        caller = ctx.pop("caller", None) or _caller()
        for sink in self._sinks:
            if level >= sink.level:
                try:
                    sink.log(level, message, caller=caller, **ctx)
                except Exception:
                    _emergency(f"{type(sink).__name__}.log failed")

    # --- metrics: unconditional fan-out --------------------------------
    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        for sink in self._sinks:
            try:
                sink.log_metrics(metrics, step, output=output)
            except Exception:
                _emergency(f"{type(sink).__name__}.log_metrics failed")

    def log_table(
        self, key: str, columns: Any, data: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        for sink in self._sinks:
            try:
                sink.log_table(key, columns, data, step, output=output)
            except Exception:
                _emergency(f"{type(sink).__name__}.log_table failed")

    def log_image(
        self, key: str, image: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        for sink in self._sinks:
            try:
                sink.log_image(key, image, step, output=output)
            except Exception:
                _emergency(f"{type(sink).__name__}.log_image failed")

    def log_hyperparams(self, params: dict) -> None:
        for sink in self._sinks:
            try:
                sink.log_hyperparams(params)
            except Exception:
                _emergency(f"{type(sink).__name__}.log_hyperparams failed")

    def watch(self, model: Any, log: str = "all") -> None:
        for sink in self._sinks:
            try:
                sink.watch(model, log)
            except Exception:
                _emergency(f"{type(sink).__name__}.watch failed")

    # --- lifecycle ------------------------------------------------------
    def setup(self) -> None:
        for sink in self._sinks:
            try:
                sink.setup()
            except Exception:
                _emergency(f"{type(sink).__name__}.setup failed")

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                _emergency(f"{type(sink).__name__}.close failed")
