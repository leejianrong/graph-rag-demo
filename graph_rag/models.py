"""Shared Pydantic v2 data models for the Graph RAG pipeline.

These models are part of the Slice V1 foundation contract. Adapters (Agent B) and
the orchestrator/tests (Agent C) code against them.

Two models matter in V1:

* :class:`IngestTrigger` — the Kafka trigger payload, carrying only
  ``{bucket, object_key}`` (ADR-0001).
* :class:`DocumentRecord` — the ``ES-Documents`` record. V1 writes ``text`` at
  ingestion; the enrichment fields (``mentions``, ``coref_clusters``,
  ``el_result``, ``vectors``) are declared as optional now so later slices extend
  the record in place at the V4 entity-linking checkpoint without breaking V1.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["IngestTrigger", "DocumentRecord"]


class IngestTrigger(BaseModel):
    """The Kafka trigger payload — carries ONLY ``bucket`` and ``object_key``.

    Published by ``POST /ingest`` after the bytes land in the object store; the
    thin Kafka consumer resolves it to a ``process_document({bucket, object_key})``
    call (ADR-0001).
    """

    bucket: str
    object_key: str

    def to_json(self) -> str:
        """Serialize this trigger to a JSON string (Kafka message value)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> IngestTrigger:
        """Deserialize a Kafka message value (JSON ``str``/``bytes``) to a trigger."""
        return cls.model_validate_json(payload)


class DocumentRecord(BaseModel):
    """The ``ES-Documents`` record for one document.

    In V1 only ``document_id``, ``bucket``, ``object_key`` and ``text`` are set —
    the bare record created at ingestion, before processing. The remaining fields
    are populated at the entity-linking checkpoint (V4) and stay ``None`` until
    then, so V1 code and later slices share one schema.
    """

    document_id: str
    bucket: str
    object_key: str
    text: str  # raw document text, written at ingestion (V1)

    # --- Enrichment fields: populated at the EL checkpoint (V4) ---------------
    # Declared Optional/None-default so later slices extend the record in place
    # without breaking the V1 write model. Kept as loose types here; the concrete
    # sub-schemas are pinned by the slices that own them (V2/V3/V4).
    mentions: list[dict[str, Any]] | None = None  # populated at EL checkpoint (V4)
    coref_clusters: list[dict[str, Any]] | None = None  # populated at EL checkpoint (V4)
    el_result: dict[str, Any] | None = None  # populated at EL checkpoint (V4)
    vectors: dict[str, Any] | None = None  # populated at EL checkpoint (V4)

    # Ignore anything from stored JSON that a later slice adds and this code
    # doesn't yet know about, rather than raising.
    model_config = {"extra": "ignore"}

    def to_json(self) -> str:
        """Serialize this record to a JSON string (e.g. for the document store)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> DocumentRecord:
        """Deserialize a JSON ``str``/``bytes`` document into a record."""
        return cls.model_validate_json(payload)
