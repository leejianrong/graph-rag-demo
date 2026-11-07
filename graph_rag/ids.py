"""Deterministic document and entity identity (ADR-0001, ADR-0004, ADR-0015).

The document ID is derived purely from ``{bucket}/{object_key}`` so that
reprocessing the same object always resolves to the same record and therefore
*overwrites* rather than duplicating (idempotent ingestion). A non-default
:class:`~graph_rag.scope.Scope` is hashed into both kinds of id, so two tenants' same-named
things get different ids; the default scope hashes the strings used before scopes existed,
so stored data stays valid.
"""

from __future__ import annotations

import hashlib

from graph_rag.scope import DEFAULT_SCOPE, Scope

__all__ = ["document_id", "entity_id"]


def document_id(bucket: str, object_key: str, scope: Scope = DEFAULT_SCOPE) -> str:
    """Return a deterministic, idempotent document ID for an object.

    The ID is the SHA-256 hex digest of ``f"{bucket}/{object_key}"``. The same
    ``(bucket, object_key)`` always yields the same ID, so re-ingesting an object
    overwrites its prior record instead of creating a duplicate (ADR-0001).

    Args:
        bucket: The object-store bucket the document lives in.
        object_key: The object key within that bucket.
        scope: Whose document it is; the default scope keeps the original digest.

    Returns:
        A 64-character lowercase hex SHA-256 digest.
    """
    canonical = f"{bucket}/{object_key}"
    if not scope.is_default:
        canonical = f"{scope.key}|{canonical}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entity_id(
    entity_type: str, normalized_name: str, scope: Scope = DEFAULT_SCOPE, salt: str = ""
) -> str:
    """Return the canonical id minted for a create-new entity (ADR-0004).

    ``"e-"`` plus 16 hex digits of a SHA-256 over type and normalized name, with the scope
    in front for a non-default scope. ``salt`` separates two distinct entities that share
    a type and name inside one scope.
    """
    base = f"el:{entity_type}:{normalized_name}"
    if not scope.is_default:
        base = f"el:{scope.key}:{entity_type}:{normalized_name}"
    if salt:
        base = f"{base}|{salt}"
    return "e-" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
