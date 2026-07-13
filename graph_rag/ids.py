"""Deterministic document identity (ADR-0001).

The document ID is derived purely from ``{bucket}/{object_key}`` so that
reprocessing the same object always resolves to the same record and therefore
*overwrites* rather than duplicating (idempotent ingestion).
"""

from __future__ import annotations

import hashlib

__all__ = ["document_id"]


def document_id(bucket: str, object_key: str) -> str:
    """Return a deterministic, idempotent document ID for an object.

    The ID is the SHA-256 hex digest of ``f"{bucket}/{object_key}"``. The same
    ``(bucket, object_key)`` always yields the same ID, so re-ingesting an object
    overwrites its prior record instead of creating a duplicate (ADR-0001).

    Args:
        bucket: The object-store bucket the document lives in.
        object_key: The object key within that bucket.

    Returns:
        A 64-character lowercase hex SHA-256 digest.
    """
    canonical = f"{bucket}/{object_key}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
