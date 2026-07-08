"""ConsoleSink: messages -> stderr, timestamped format + per-level ANSI color."""

from __future__ import annotations

import datetime
import logging
import sys
from typing import Any, Optional

from ..interface import INFO, LoggerInterface

_COLORS = {
    logging.DEBUG: "\x1b[36m",     # cyan
    logging.INFO: "\x1b[32m",      # green
    logging.WARNING: "\x1b[33m",   # yellow
    logging.ERROR: "\x1b[31m",     # red
    logging.CRITICAL: "\x1b[41m",  # red background
}
_RESET = "\x1b[0m"


class ConsoleSink(LoggerInterface):
    """Text sink for the console. Only handles `log()` (messages)."""

    def __init__(
        self,
        level: int = INFO,
        stream: Any = None,
        color: Optional[bool] = None,
        options: Optional[dict] = None,
    ):
        super().__init__(level=level, options=options)
        self._stream = stream if stream is not None else sys.stderr
        # color only if TTY (otherwise it pollutes redirected logs)
        self._color = self._stream.isatty() if color is None else color

    def log(
        self,
        level: int,
        message: str,
        caller: Optional[str] = None,
        output: Optional[str] = None,
        **ctx: Any,
    ) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        name = logging.getLevelName(level)
        loc = caller or "-"
        if self._color and level in _COLORS:
            name = f"{_COLORS[level]}{name:<8}{_RESET}"
        else:
            name = f"{name:<8}"
        # Named output -> bare separator line delimiting the block (no ts/level prefix).
        prefix = self._separator(output, level) if output else ""
        line = f"{prefix}{ts} | {name} | {loc} - {message}\n"
        try:
            self._stream.write(line)
            self._stream.flush()
        except Exception:
            pass

    def _separator(self, output: str, level: int) -> str:
        """A bare, level-colored rule line naming the output, e.g. `──── partition ────`."""
        rule = f"──── {output} " + "─" * max(4, 40 - len(output))
        if self._color and level in _COLORS:
            rule = f"{_COLORS[level]}{rule}{_RESET}"
        return f"{rule}\n"
