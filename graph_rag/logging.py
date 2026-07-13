"""Logging seam (ARCHITECTURE §8).

Everyone logs through :func:`get_logger` and configures once via
:func:`configure_logging`, so basic Python logging can be swapped for structured
/ JSON logging later without touching call sites.
"""

from __future__ import annotations

import logging

__all__ = ["configure_logging", "get_logger"]

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a basic handler + format.

    Idempotent-ish via ``force=True``: calling it again re-applies the level and
    format rather than stacking handlers. Swap this body for a structured logging
    setup later; call sites using :func:`get_logger` do not change.

    Args:
        level: A standard logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    logging.basicConfig(
        level=level.upper(),
        format=_DEFAULT_FORMAT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger` obtained through the seam.

    Args:
        name: Logger name, conventionally the module ``__name__``.

    Returns:
        A standard library logger; the returned type is stable even if the
        backend implementation changes later.
    """
    return logging.getLogger(name)
