"""Sinks: `LoggerInterface` implementations, each targeting one output."""

from .console import ConsoleSink
from .csv import CsvSink
from .file import FileSink

# WandbSink / TensorBoardSink / RealtimePlotSink are imported lazily by the
# factory (heavy / optional dependencies).

__all__ = ["ConsoleSink", "FileSink", "CsvSink"]
