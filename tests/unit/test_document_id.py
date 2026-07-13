"""Unit tests for the deterministic document-ID contract (ADR-0001, TESTING §4).

The ID must be deterministic (same ``{bucket, object_key}`` -> same ID), distinct
for distinct inputs, and stable across re-ingest (which is what makes reprocessing
overwrite rather than duplicate).
"""

from __future__ import annotations

import hashlib

from graph_rag.ids import document_id


def test_document_id_is_deterministic() -> None:
    """Same (bucket, object_key) always yields the same ID."""
    first = document_id("documents", "a.md")
    second = document_id("documents", "a.md")
    assert first == second


def test_document_id_differs_for_different_object_key() -> None:
    """Different object keys in the same bucket yield different IDs."""
    assert document_id("documents", "a.md") != document_id("documents", "b.md")


def test_document_id_differs_for_different_bucket() -> None:
    """The bucket participates in the ID: same key, different bucket -> different ID."""
    assert document_id("documents", "a.md") != document_id("other", "a.md")


def test_document_id_is_sha256_of_bucket_slash_key() -> None:
    """The ID is exactly the SHA-256 hex digest of ``f'{bucket}/{object_key}'``."""
    expected = hashlib.sha256(b"documents/a.md").hexdigest()
    assert document_id("documents", "a.md") == expected


def test_document_id_is_64_char_hex() -> None:
    """A SHA-256 hex digest is 64 lowercase hex characters."""
    doc_id = document_id("documents", "nested/path/report.md")
    assert len(doc_id) == 64
    assert all(c in "0123456789abcdef" for c in doc_id)


def test_document_id_is_stable_across_reingest() -> None:
    """Re-ingesting the same object resolves to the same ID (idempotent overwrite).

    Framing of ADR-0001: because the ID is a pure function of the location, a
    second ingestion of the same object targets the same record and overwrites it,
    never creating a duplicate.
    """
    original = document_id("documents", "report.md")
    reingested = document_id("documents", "report.md")
    assert original == reingested
