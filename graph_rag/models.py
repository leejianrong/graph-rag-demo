"""Shared Pydantic v2 data models for the Graph RAG pipeline.

These models are part of the Slice V1 foundation contract. Adapters (Agent B) and
the orchestrator/tests (Agent C) code against them.

V1 models:

* :class:`IngestTrigger` — the Kafka trigger payload, carrying only
  ``{bucket, object_key}`` (ADR-0001).
* :class:`DocumentRecord` — the ``ES-Documents`` record. V1 writes ``text`` at
  ingestion; the enrichment fields (``mentions``, ``coref_clusters``,
  ``el_result``, ``vectors``) are declared as optional now so later slices extend
  the record in place at the V4 entity-linking checkpoint without breaking V1.

V2 (NER) adds the in-memory enrichment carry (ADR-0002, ARCHITECTURE §4):

* :class:`Mention` / :class:`Sentence` — one NER mention (typed + char span) and
  one segmented sentence.
* :class:`PipelineResult` — the object the orchestrator RETURNS. It carries the
  raw :class:`DocumentRecord` plus the enrichment computed so far, held
  **in-memory** and NOT persisted to ES until the V4 EL checkpoint. Later slices
  extend it in place (V3 ``coref_clusters``, V4 ``el_result``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "IngestTrigger",
    "DocumentRecord",
    "CuratedType",
    "Mention",
    "Sentence",
    "PipelineResult",
]

# The curated NER type set (ADR-0002): spaCy's OntoNotes labels narrowed to the
# types this graph needs. ``GPE`` and ``LOC`` both map to ``LOCATION``; labels
# outside this set are dropped. ``PRODUCT`` is optional-but-included.
CuratedType = Literal[
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "EVENT",
    "NORP",
    "PRODUCT",
]


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


# --- V2 (NER) in-memory enrichment ------------------------------------------


class Mention(BaseModel):
    """One typed NER mention with character offsets into the raw document text.

    The offsets are half-open ``[char_start, char_end)`` slices of the *raw*
    ``DocumentRecord.text``, so ``text == raw[char_start:char_end]`` holds. They
    align coref mentions (V3), attach provenance to triples (V5) and drive UI
    highlighting (ADR-0002).
    """

    text: str
    type: CuratedType
    char_start: int
    char_end: int


class Sentence(BaseModel):
    """One segmented sentence with character offsets into the raw document text.

    Produced in the same spaCy pass as the mentions (ADR-0002). ``index`` is the
    zero-based position of the sentence in the document. ``text`` equals
    ``raw[char_start:char_end]``.
    """

    text: str
    char_start: int
    char_end: int
    index: int


class PipelineResult(BaseModel):
    """The object the orchestrator RETURNS — the in-memory enrichment carry.

    Bundles the raw :class:`DocumentRecord` (already persisted to ES at
    ingestion) with the enrichment computed so far in the pipeline. Per the write
    model (ARCHITECTURE §4, ADR-0001), this enrichment is held **in-memory** and
    is NOT persisted to ``ES-Documents`` until the V4 entity-linking checkpoint —
    in V2 the ES record still stores raw text only.

    V2 populates ``mentions`` and ``sentences``. Later slices extend this object
    in place: V3 adds a ``coref_clusters`` field, V4 adds an ``el_result`` field
    and then writes the whole thing back into ``record`` at the EL checkpoint.
    """

    record: DocumentRecord
    mentions: list[Mention] = Field(default_factory=list)
    sentences: list[Sentence] = Field(default_factory=list)
