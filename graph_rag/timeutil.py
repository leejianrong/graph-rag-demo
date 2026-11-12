"""One timestamp format for records that carry a time, so strings sort as times do."""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now_iso"]


def utc_now_iso() -> str:
    """The current UTC time as fixed-width ISO-8601 with microseconds (sorts as text)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
