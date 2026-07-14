"""Fast end-to-end suite for the ingestion + NER path (TESTING §2, primary gate).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against the in-memory fakes — no Docker, no spaCy model, deterministic,
$0. The NER stage is injected as a :class:`~graph_rag.fakes.FakeNerStage` so the
fast suite proves the *wiring* + the :class:`~graph_rag.models.PipelineResult`
carry, not spaCy quality. This is the pattern every later slice follows: assert on
the *external behavior at the seam*, not on internal call sequences.

Covers the V1 guarantees (still intact under V2) plus the V2 carry:

* the raw record is written with the deterministic ``document_id`` and raw text —
  and NER output is NOT persisted to the record (raw text only until V4);
* the returned ``PipelineResult`` carries the curated-type mentions (with char
  offsets) + sentences computed in-memory;
* idempotent overwrite — re-ingest updates in place, never duplicates (R1.5);
* log-and-drop — a failing document returns ``None``, does not raise, and does not
  wedge the loop; the next good trigger still processes (ADR-0001).
"""

from __future__ import annotations

import pytest

from graph_rag.fakes import FakeNerStage, InMemoryDocumentStore, InMemoryObjectStore
from graph_rag.ids import document_id
from graph_rag.models import IngestTrigger, Mention, Sentence
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.coref import FakeCorefStage

BUCKET = "documents"
KEY = "a.md"


@pytest.fixture
def orchestrator(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    ner_stage: FakeNerStage,
    coref_stage: FakeCorefStage,
) -> Orchestrator:
    """An orchestrator wired to the shared in-memory fakes + canned NER/coref stages.

    Injecting :class:`~graph_rag.stages.coref.FakeCorefStage` keeps the fast suite
    LLM-free: without it the orchestrator would build the real ``LLMCorefStage``
    default and try a provider call.
    """
    return Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner_stage,
        coref_stage=coref_stage,
    )


def test_writes_raw_record_with_deterministic_id(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """A processed document is stored with the deterministic id and raw text."""
    object_store.put(BUCKET, KEY, b"hello graph rag")
    trigger = IngestTrigger(bucket=BUCKET, object_key=KEY)

    result = orchestrator.process_document(trigger)

    expected_id = document_id(BUCKET, KEY)
    assert result is not None
    record = result.record
    assert record.document_id == expected_id
    assert record.bucket == BUCKET
    assert record.object_key == KEY
    assert record.text == "hello graph rag"

    # Asserted at the seam: the record is actually in the document store.
    stored = document_store.get(expected_id)
    assert stored is not None
    assert stored.text == "hello graph rag"

    # V2 still writes RAW text only — NER output is carried in-memory, not
    # persisted to the ES record until the V4 EL checkpoint. The enrichment
    # fields therefore stay at their empty defaults on the stored raw record.
    assert stored.mentions == []
    assert stored.coref_clusters == []
    assert stored.el_result == []
    assert stored.sentence_vectors is None


def test_result_carries_canned_mentions_and_sentences(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """The returned PipelineResult carries the NER stage's mentions + sentences.

    Proves the in-memory carry + wiring: curated-type mentions with char offsets
    and segmented sentences flow through to the orchestrator's result, while the
    stored ES record stays raw (not persisted until V4).
    """
    text = "Ada Lovelace worked in London."
    object_store.put(BUCKET, KEY, text.encode())

    canned_mentions = [
        Mention(text="Ada Lovelace", type="PERSON", char_start=0, char_end=12),
        Mention(text="London", type="LOCATION", char_start=22, char_end=28),
    ]
    canned_sentences = [
        Sentence(text=text, char_start=0, char_end=len(text), index=0),
    ]
    ner_stage = FakeNerStage(mentions=canned_mentions, sentences=canned_sentences)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner_stage,
        coref_stage=FakeCorefStage(),
    )

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is not None
    # The raw record is still written with raw text.
    assert result.record.text == text
    # Mentions carried in-memory, with curated types + char offsets.
    assert result.mentions == canned_mentions
    assert [m.type for m in result.mentions] == ["PERSON", "LOCATION"]
    assert (result.mentions[0].char_start, result.mentions[0].char_end) == (0, 12)
    # Sentences carried in-memory.
    assert result.sentences == canned_sentences
    # Not persisted to the ES record yet (raw-only write model).
    stored = document_store.get(document_id(BUCKET, KEY))
    assert stored is not None
    assert stored.mentions == []


def test_decodes_utf8_text(
    orchestrator: Orchestrator,
    object_store: InMemoryObjectStore,
) -> None:
    """Bytes are decoded as UTF-8 into the record text."""
    object_store.put(BUCKET, KEY, "café — déjà vu".encode())
    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))
    assert result is not None
    assert result.record.text == "café — déjà vu"


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
    assert first.record.document_id == second.record.document_id

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
    ner_stage: FakeNerStage,
) -> None:
    """A failing document-store write is dropped (returns None), not raised."""

    class ExplodingDocumentStore(InMemoryDocumentStore):
        """Document store whose upsert always fails, to exercise log-and-drop."""

        def upsert(self, record) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated ES write failure")

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=ExplodingDocumentStore(),
        ner_stage=ner_stage,
    )
    object_store.put(BUCKET, KEY, b"content")

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is None


def test_ner_failure_is_logged_and_dropped(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """A failing NER stage is dropped (returns None), not raised (ADR-0001)."""

    class ExplodingNerStage:
        """NER stage whose analyze always fails, to exercise log-and-drop."""

        def analyze(self, text: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated spaCy failure")

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ExplodingNerStage(),
    )
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
    assert good.record.text == "good content"
    assert document_store.get(document_id(BUCKET, "good.md")) is not None
