"""Fast end-to-end suite for the V3 coref path (TESTING §2, primary gate).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against in-memory fakes — no Docker, no spaCy model, no LLM provider,
deterministic, $0. The coref stage is exercised two ways, both offline:

* an :class:`~graph_rag.stages.coref.LLMCorefStage` backed by
  :class:`~graph_rag.fakes.FakeLLMClient` (canned STRUCTURED response) — proves the
  real structured-output *call path* + the cache/no-recompute counter; and
* a :class:`~graph_rag.stages.coref.FakeCorefStage` — proves the wiring + carry.

Covers the V3 demo (non-destructive cluster map; identical re-run does no extra
LLM work) and re-asserts the V1/V2 guarantees still hold under V3 (deterministic
id, raw-only ES write, idempotent overwrite, log-and-drop — including a coref
failure).
"""

from __future__ import annotations

import pytest

from graph_rag.fakes import FakeLLMClient, FakeNerStage, InMemoryDocumentStore, InMemoryObjectStore
from graph_rag.ids import document_id
from graph_rag.models import CorefCluster, IngestTrigger, Mention
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.coref import FakeCorefStage, LLMCorefStage

BUCKET = "documents"
KEY = "a.md"

# A doc with a repeat ("Ada Lovelace" / "Ada") and a pronoun ("She").
TEXT = "Ada Lovelace lived in London. She loved math. Ada wrote the first program."

CANNED_MENTIONS = [
    Mention(text="Ada Lovelace", type="PERSON", char_start=0, char_end=12),
    Mention(text="London", type="LOCATION", char_start=21, char_end=27),
    Mention(text="Ada", type="PERSON", char_start=46, char_end=49),
]
# The canonical cluster the fake LLM "returns": pronoun + repeat collapse onto the
# most complete proper-name mention. Non-destructive — surface forms, not offsets.
CANNED_CLUSTERS = [
    CorefCluster(canonical="Ada Lovelace", members=["Ada Lovelace", "She", "Ada"]),
    CorefCluster(canonical="London", members=["London"]),
]


def _put_doc(object_store: InMemoryObjectStore) -> None:
    object_store.put(BUCKET, KEY, TEXT.encode())


def test_result_carries_non_destructive_cluster_map_via_llm_stage(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """The PipelineResult carries a non-destructive coref cluster map (LLM path).

    Uses ``LLMCorefStage`` over a ``FakeLLMClient`` with canned structured output,
    so the real structured-output call path runs offline. Asserts the cluster map
    groups mention surface forms (incl. pronoun/repeat) onto an in-doc canonical,
    and that the raw text is untouched (a map, not a rewrite).
    """
    _put_doc(object_store)
    llm = FakeLLMClient(clusters=CANNED_CLUSTERS)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=LLMCorefStage(llm),
    )

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is not None
    assert llm.calls == 1  # coref made exactly one structured LLM call
    # Non-destructive: raw text preserved verbatim (map, not rewrite).
    assert result.record.text == TEXT
    # The cluster map: pronoun + repeat collapsed onto the in-doc canonical.
    assert result.coref_clusters == CANNED_CLUSTERS
    ada = result.coref_clusters[0]
    assert ada.canonical == "Ada Lovelace"
    assert set(ada.members) == {"Ada Lovelace", "She", "Ada"}
    # Every canonical/member is a verbatim surface form of the raw text.
    for cluster in result.coref_clusters:
        assert cluster.canonical in TEXT
        for member in cluster.members:
            assert member in TEXT


def test_identical_rerun_does_no_extra_llm_work(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """The V3 demo's $0 re-run: a second identical ingest is deterministic + free.

    With the fake LLM standing in for the response cache (ADR-0010), a re-ingest
    of the identical document yields an identical cluster map. (The real
    ``LiteLLMClient`` proves the provider is not re-called on the cache hit — see
    ``tests/unit/test_llm_cache.py``.)
    """
    _put_doc(object_store)
    llm = FakeLLMClient(clusters=CANNED_CLUSTERS)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=LLMCorefStage(llm),
    )
    trigger = IngestTrigger(bucket=BUCKET, object_key=KEY)

    first = orchestrator.process_document(trigger)
    second = orchestrator.process_document(trigger)

    assert first is not None and second is not None
    assert first.coref_clusters == second.coref_clusters == CANNED_CLUSTERS
    # Same deterministic id -> idempotent overwrite, exactly one stored record.
    assert first.record.document_id == second.record.document_id
    assert len(document_store._records) == 1  # noqa: SLF001 — asserting no duplication.


def test_coref_not_persisted_to_es_record(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """V3 still writes RAW text only — coref is in-memory, not on the ES record."""
    _put_doc(object_store)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=FakeCorefStage(clusters=CANNED_CLUSTERS),
    )

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is not None
    assert result.coref_clusters == CANNED_CLUSTERS
    stored = document_store.get(document_id(BUCKET, KEY))
    assert stored is not None
    assert stored.text == TEXT
    # Enrichment (incl. coref) is NOT persisted until the V4 EL checkpoint.
    assert stored.coref_clusters is None
    assert stored.mentions is None


def test_coref_failure_is_logged_and_dropped(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """A failing coref stage is dropped (returns None), not raised (ADR-0001)."""

    class ExplodingCorefStage:
        """Coref stage whose resolve always fails, to exercise log-and-drop."""

        def resolve(self, text, mentions):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated LLM failure")

    _put_doc(object_store)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=ExplodingCorefStage(),
    )

    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))

    assert result is None


def test_coref_failure_does_not_wedge_the_loop(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """After a coref-dropped document, the next good trigger still processes."""

    class SometimesExplodingCorefStage:
        """Fails on the first call, succeeds afterwards."""

        def __init__(self) -> None:
            self._calls = 0

        def resolve(self, text, mentions):  # type: ignore[no-untyped-def]
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("simulated LLM failure")
            return list(CANNED_CLUSTERS)

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=SometimesExplodingCorefStage(),
    )

    object_store.put(BUCKET, "bad.md", TEXT.encode())
    bad = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key="bad.md"))
    assert bad is None

    object_store.put(BUCKET, "good.md", TEXT.encode())
    good = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key="good.md"))
    assert good is not None
    assert good.coref_clusters == CANNED_CLUSTERS


def test_default_coref_stage_is_llm_backed(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """Constructing without a coref stage defaults to a real LLMCorefStage.

    Mirrors the NER default (real stage). Construction must be cheap + offline: no
    LiteLLM import, no provider call, no API key. We only assert the default type;
    we never call it (which would need a provider).
    """
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(),
    )
    assert isinstance(orchestrator._coref_stage, LLMCorefStage)  # noqa: SLF001


@pytest.mark.parametrize("clusters", [[], list(CANNED_CLUSTERS)])
def test_empty_and_populated_cluster_maps(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    clusters: list[CorefCluster],
) -> None:
    """The carry handles both an empty and a populated cluster map."""
    _put_doc(object_store)
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=FakeNerStage(mentions=CANNED_MENTIONS),
        coref_stage=FakeCorefStage(clusters=clusters),
    )
    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))
    assert result is not None
    assert result.coref_clusters == clusters
