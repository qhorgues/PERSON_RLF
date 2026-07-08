"""CsvSink: metrics -> CSV file (schema compatible with plot_fl_results.py).

Dynamic header: if new keys appear (train/val transition, STN metrics...), the
file is rewritten with the union of columns; otherwise the row is simply
appended. Missing cells stay empty.

Named outputs (`output=`): each name gets its own CSV alongside the base file
(`output="client_0"` -> `client_0.csv`), with an independent header/rows state.
The default (`output=None`) writes to the base `metrics.csv`.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from ..interface import CRITICAL, LoggerInterface

_NEVER = CRITICAL + 10  # never receives a text message


def _to_scalar(value: Any) -> Any:
    """Cast tensor/np -> float; None/'' unchanged; otherwise best-effort float."""
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class _CsvStream:
    """Independent CSV state (path + dynamic header + rows) for one output."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fieldnames: List[str] = ["step"]
        self.rows: List[dict] = []

    def add(self, row: dict) -> None:
        # Merge into the last row instead of appending a new one when it shares
        # the same step (e.g. train metrics from `aggregate_fit` then eval
        # metrics from `aggregate_evaluate` for the same `server_round`).
        if self.rows and self.rows[-1].get("step") == row.get("step"):
            self.rows[-1].update(row)
            new_keys = [k for k in row if k not in self.fieldnames]
            if new_keys:
                self.fieldnames.extend(new_keys)
            self.rewrite()
            return

        self.rows.append(row)
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            self.rewrite()
        else:
            self._append(row)

    def rewrite(self) -> None:
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def _append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, restval="", extrasaction="ignore")
            writer.writerow(row)


class CsvSink(LoggerInterface):
    """Only handles `log_metrics()`. One CSV per named output."""

    def __init__(self, path: str, options: Optional[dict] = None):
        super().__init__(level=_NEVER, options=options)
        self._path = path
        self._dir = os.path.dirname(path) or "."
        # Per-output streams; key None -> base metrics.csv.
        self._streams: Dict[Optional[str], _CsvStream] = {}

    def _stream(self, output: Optional[str]) -> _CsvStream:
        stream = self._streams.get(output)
        if stream is None:
            path = self._path if output is None else os.path.join(self._dir, f"{output}.csv")
            stream = _CsvStream(path)
            self._streams[output] = stream
        return stream

    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        stream = self._stream(output)
        row = {"step": step if step is not None else len(stream.rows)}
        for key, value in metrics.items():
            row[key] = _to_scalar(value)
        stream.add(row)

    def close(self) -> None:
        # guarantees a consistent file even if no rewrite happened
        for stream in self._streams.values():
            if stream.rows:
                stream.rewrite()
