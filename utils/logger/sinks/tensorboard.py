"""TensorBoardSink: scalar metrics -> TensorBoard (SummaryWriter)."""

from __future__ import annotations

import os
from typing import Any, Optional

from ..interface import CRITICAL, LoggerInterface

_NEVER = CRITICAL + 10


class TensorBoardSink(LoggerInterface):
    def __init__(self, save_dir: str, name: Optional[str] = None, options: Optional[dict] = None):
        super().__init__(level=_NEVER, options=options)
        self._log_dir = os.path.join(save_dir, name) if name else save_dir
        self._writer = None
        self._step = 0

    def setup(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=self._log_dir)

    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        if self._writer is None:
            return
        step = step if step is not None else self._step
        prefix = f"{output}/" if output else ""
        for key, value in metrics.items():
            try:
                self._writer.add_scalar(f"{prefix}{key}", float(value), step)
            except (TypeError, ValueError):
                continue
        self._step = step + 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
