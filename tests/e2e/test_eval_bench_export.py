"""Fast end-to-end suite for the eval-bench export hook (opt-in, additive).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against in-memory fakes plus a recording fake
:class:`~graph_rag.adapters.eval_bench_export.EvalBenchExportStage` stand-in — no
Docker, no model, no LLM, deterministic, ``$0`` (ADR-0010 style, mirroring
``tests/e2e/test_kg_build.py``).

Covers the two guarantees the brief calls out:

* **wired**: the hook fires exactly once per successfully-processed document,
  handed the same ``(record, canonical_entities, triples)`` the graph checkpoint
  itself just upserted/wrote, plus the trigger's ``source_url``;
* **omitted (the default)**: leaving ``eval_bench_export`` unset changes nothing
  about the returned :class:`~graph_rag.models.PipelineResult` or what got
  written to the document/entity/graph stores — V1-V5 behaviour is unaffected.
"""

from __future__ import annotations

from typing import Any

from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    FakeNerStage,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.models import (
    CanonicalEntity,
    CorefCluster,
    DocumentRecord,
    IngestTrigger,
    Mention,
    Sentence,
    Triple,
)
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.coref import FakeCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.kg_build import KgBuildStage, TripleList

BUCKET = "documents"
KEY = "acme.md"
URL = "https://example.com/articles/acme-in-london"

SENT0 = "Ada Lovelace works for Acme Corp."
SENT1 = "Acme Corp is based in London."
TEXT = f"{SENT0} {SENT1}"

ADA = "Ada Lovelace"
ACME = "Acme Corp"
LONDON = "London"


class _RecordingEvalBenchExport:
    """A no-op fake standing in for a real ``EvalBenchExportStage`` (test-only).

    Records every ``write`` call's arguments on :attr:`calls` so the test can
    assert on the exact shape the orchestrator hands over, without touching
    Elasticsearch/Neo4j.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write(
        self,
        record: DocumentRecord,
        canonical_entities: list[CanonicalEntity],
        triples: list[Triple],
        *,
        source_url: str | None,
    ) -> None:
        self.calls.append(
            {
                "record": record,
                "canonical_entities": canonical_entities,
                "triples": triples,
                "source_url": source_url,
            }
        )


def _stages() -> tuple[FakeNerStage, FakeCorefStage]:
    """Canned NER + coref for the fixture doc (three typed, one-mention clusters)."""
    mentions = [
        Mention(text=ADA, type="PERSON", char_start=0, char_end=len(ADA)),
        Mention(text=ACME, type="ORG", char_start=23, char_end=23 + len(ACME)),
        Mention(
            text=LONDON,
            type="LOCATION",
            char_start=len(SENT0) + 15,
            char_end=len(SENT0) + 15 + len(LONDON),
        ),
    ]
    sentences = [
        Sentence(text=SENT0, char_start=0, char_end=len(SENT0), index=0),
        Sentence(text=SENT1, char_start=len(SENT0) + 1, char_end=len(TEXT), index=1),
    ]
    ner = FakeNerStage(mentions=mentions, sentences=sentences)
    coref = FakeCorefStage(
        clusters=[
            CorefCluster(canonical=ADA, members=[ADA]),
            CorefCluster(canonical=ACME, members=[ACME]),
            CorefCluster(canonical=LONDON, members=[LONDON]),
        ]
    )
    return ner, coref


def _build_orchestrator(
    *,
    eval_bench_export: _RecordingEvalBenchExport | None,
) -> tuple[Orchestrator, InMemoryObjectStore, InMemoryDocumentStore, InMemoryGraphStore]:
    object_store = InMemoryObjectStore()
    document_store = InMemoryDocumentStore()
    entity_store = InMemoryEntityStore()
    graph_store = InMemoryGraphStore()
    embedder = FakeEmbedder()
    ner, coref = _stages()

    object_store.put(BUCKET, KEY, TEXT.encode())

    # Entity linking mints canonical ids deterministically from type+name; we
    # don't need the exact ids for this test (only that they end up consistent
    # between the EL result and the canned triples), so drive EL once up front
    # isn't necessary — KgBuildStage.build resolves subject/object ids from its
    # own entity-map arg, so an empty/ignored triple response is fine here: this
    # test only needs canonical_entities to be non-empty for the hook to fire.
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=EntityLinkingStage(entity_store, embedder),
        graph_store=graph_store,
        kg_build_stage=KgBuildStage(FakeLLMClient(structured_response=TripleList(triples=[]))),
        eval_bench_export=eval_bench_export,
    )
    return orchestrator, object_store, document_store, graph_store


def test_hook_fires_with_expected_shape_when_stage_provided() -> None:
    """Wired: the hook fires once, handed the SAME record/entities/triples/url."""
    export = _RecordingEvalBenchExport()
    orchestrator, _obj, _doc, graph = _build_orchestrator(eval_bench_export=export)

    result = orchestrator.process_document(
        IngestTrigger(bucket=BUCKET, object_key=KEY, source_url=URL)
    )

    assert result is not None
    assert len(export.calls) == 1
    call = export.calls[0]

    # Same record the pipeline persisted (by identity of content, not object).
    assert call["record"].document_id == result.record.document_id
    assert call["record"].text == TEXT

    # Same canonical entities the graph checkpoint upserted as nodes.
    assert {e.canonical_id for e in call["canonical_entities"]} == {
        n.canonical_id
        for n in graph._nodes.values()  # noqa: SLF001 - test assertion
    }
    assert len(call["canonical_entities"]) == 3  # Ada, Acme, London

    # Same triples the pipeline returned (here: none, since the canned LLM
    # response is empty — still proves the exact list identity/shape is passed).
    assert call["triples"] == result.triples

    # The trigger's source_url reached the hook untouched.
    assert call["source_url"] == URL


def test_omitting_eval_bench_export_leaves_pipeline_behaviour_unchanged() -> None:
    """Default (omitted): behaviour is byte-for-byte identical to the wired run."""
    recording_export = _RecordingEvalBenchExport()
    with_export, _obj1, doc1, graph1 = _build_orchestrator(eval_bench_export=recording_export)
    without_export, _obj2, doc2, graph2 = _build_orchestrator(eval_bench_export=None)

    trigger = IngestTrigger(bucket=BUCKET, object_key=KEY, source_url=URL)
    result_with = with_export.process_document(trigger)
    result_without = without_export.process_document(trigger)

    assert result_with is not None
    assert result_without is not None

    # Identical PipelineResult content either way.
    assert result_with.record.document_id == result_without.record.document_id
    assert result_with.record.text == result_without.record.text
    assert result_with.el_result == result_without.el_result
    assert result_with.triples == result_without.triples

    # Identical persisted state either way.
    assert doc1.get(result_with.record.document_id) == doc2.get(result_without.record.document_id)
    assert graph1.node_count() == graph2.node_count()
    assert graph1.edge_count() == graph2.edge_count() == 0


def test_hook_not_reached_without_graph_store_or_kg_build_stage() -> None:
    """No graph_store/kg_build_stage wired -> the export hook is never reached.

    The hook lives inside the same opt-in graph-checkpoint block (it needs that
    block's canonical_entities/triples), so an eval_bench_export supplied WITHOUT
    a graph store + KG-build stage simply never fires — proving it adds no new
    failure mode to the V1-V4 (pre-graph) configuration.
    """
    export = _RecordingEvalBenchExport()
    object_store = InMemoryObjectStore()
    document_store = InMemoryDocumentStore()
    entity_store = InMemoryEntityStore()
    embedder = FakeEmbedder()
    ner, coref = _stages()
    object_store.put(BUCKET, KEY, TEXT.encode())

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=EntityLinkingStage(entity_store, embedder),
        eval_bench_export=export,
        # graph_store + kg_build_stage omitted on purpose.
    )
    result = orchestrator.process_document(
        IngestTrigger(bucket=BUCKET, object_key=KEY, source_url=URL)
    )

    assert result is not None
    assert export.calls == []
