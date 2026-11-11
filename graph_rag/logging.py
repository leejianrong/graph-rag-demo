"""Logging seam (ARCHITECTURE §8).

Everyone logs through :func:`get_logger` and configures once via
:func:`configure_logging`, so basic Python logging can be swapped for structured
/ JSON logging later without touching call sites.
"""

from __future__ import annotations

import json
import logging

from opentelemetry import trace

__all__ = ["configure_logging", "get_logger"]

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


class _TraceFilter(logging.Filter):
    """Put the current trace id on every record, so a log line can be matched to its trace."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = trace.get_current_span().get_span_context()
        record.trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else ""
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Only the message and its traceback: no other fields of a call."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", ""),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Configure root logging with a basic handler + format.

    Idempotent-ish via ``force=True``: calling it again re-applies the level and
    format rather than stacking handlers. Swap this body for a structured logging
    setup later; call sites using :func:`get_logger` do not change.

    Args:
        level: A standard logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        log_format: ``"text"`` (default) or ``"json"`` (GRA-56).
    """
    logging.basicConfig(
        level=level.upper(),
        format=_DEFAULT_FORMAT,
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_TraceFilter())
        if log_format == "json":
            handler.setFormatter(_JsonFormatter())


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger` obtained through the seam.

    Args:
        name: Logger name, conventionally the module ``__name__``.

    Returns:
        A standard library logger; the returned type is stable even if the
        backend implementation changes later.
    """
    return logging.getLogger(name)
