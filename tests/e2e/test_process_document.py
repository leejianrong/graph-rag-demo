"""Fast end-to-end suite for the V1 ingestion path (TESTING §2, primary gate).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against the in-memory fakes — no Docker, deterministic, $0. This is the
pattern every later slice follows: assert on the *external behavior at the seam*
(the written ``ES-Documents`` record), not on internal call sequences.

Covers the three V1 guarantees:

* the raw record is written with the deterministic ``document_id`` and raw text;
* idempotent overwrite — re-ingest updates in place, never duplicates (R1.5);
* log-and-drop — a failing document returns ``None``, does not raise, and does not
  wedge the loop; the next good trigger still processes (ADR-0001).
"""

from __future__ import annotations

import pytest

from graph_rag.fakes import InMemoryDocumentStore, InMemoryObjectStore
from graph_rag.ids import document_id
from graph_rag.models import IngestTrigger
from graph_rag.orchestrator import Orchestrator

BUCKET = "documents"
KEY = "a.md"


@pytest.fixture
def orchestrator(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> Orchestrator:
    """An orchestrator wired to the shared in-memory fakes."""
    return Orchestrator(object_store=object_store, document_store=document_store)


def test_writes_raw_record_with_deterministic_id(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """A processed document is stored with the deterministic id and raw text."""
    object_store.put(BUCKET, KEY, b"hello graph rag")
    trigger = IngestTrigger(bucket=BUCKET, object_key=KEY)

    record = orchestrator.process_document(trigger)

    expected_id = document_id(BUCKET, KEY)
    assert record is not None
    assert record.document_id == expected_id
    assert record.bucket == BUCKET
    assert record.object_key == KEY
    assert record.text == "hello graph rag"

    # Asserted at the seam: the record is actually in the document store.
    stored = document_store.get(expected_id)
    assert stored is not None
    assert stored.text == "hello graph rag"

    # V1 writes RAW text only — no enrichment.
    assert stored.mentions is None
    assert stored.coref_clusters is None
    assert stored.el_result is None
    assert stored.vectors is None


def test_decodes_utf8_text(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
) -> None:
    """Bytes are decoded as UTF-8 into the record text."""
    object_store.put(BUCKET, KEY, "café — déjà vu".encode())
    record = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))
    assert record is not None
    assert record.text == "café — déjà vu"


def test_reingest_overwrites_no_duplicate(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """Re-ingesting the same object overwrites its record (R1.5, ADR-0001)."""
    trigger = IngestTrigger(bucket=BUCKET, object_key=KEY)

    object_store.put(BUCKET, KEY, b"first version")
    first = orchestrator.process_document(trigger)

    # Same location -> new content -> re-ingest.
    object_store.put(BUCKET, KEY, b"second version")
    second = orchestrator.process_document(trigger)

    assert first is not None
    assert second is not None
    # Same deterministic id both times.
    assert first.document_id == second.document_id

    # Exactly ONE record, and its text is the latest.
    assert len(document_store._records) == 1  # noqa: SLF001 — asserting no duplication.
    stored = document_store.get(document_id(BUCKET, KEY))
    assert stored is not None
    assert stored.text == "second version"


def test_missing_object_is_logged_and_dropped(
    orchestrator: Orchestrator,
    document_store: InMemoryDocumentStore,
) -> None:
    """A missing object (get raises) is dropped: returns None, does not raise."""
    # Nothing put into the object store -> FileNotFoundError inside process_document.
    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key="missing.md"))

    assert result is None
    # Nothing persisted for the failed document.
    assert document_store.get(document_id(BUCKET, "missing.md")) is None


def test_downstream_write_failure_is_logged_and_dropped(
    object_store: InMemoryObjectStore,
) -> None:
    """A failing document-store write is dropped (returns None), not raised."""

    class ExplodingDocumentStore(InMemoryDocumentStore):
        """Document store whose upsert always fails, to exercise log-and-drop."""

        def upsert(self, record) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated ES write failure")

    orchestrator = Orchestrator(object_store=object_store, document_store=ExplodingDocumentStore())
    object_store.put(BUCKET, KEY, b"content")

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is None


def test_failure_does_not_wedge_the_loop(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """After a dropped document, the next good trigger still processes.

    This is the point of log-and-drop (ADR-0001): one bad document must not wedge
    the consumer loop.
    """
    # First trigger: object is missing -> dropped.
    bad = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key="missing.md"))
    assert bad is None

    # Loop continues: the next, good trigger processes normally.
    object_store.put(BUCKET, "good.md", b"good content")
    good = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key="good.md"))

    assert good is not None
    assert good.text == "good content"
    assert document_store.get(document_id(BUCKET, "good.md")) is not None
