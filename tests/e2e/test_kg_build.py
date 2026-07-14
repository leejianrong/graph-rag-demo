"""Fast end-to-end suite for the V5 KG-build + graph checkpoint (TESTING §2, gate).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against in-memory fakes — no Docker, no model, no LLM, deterministic,
``$0``. The real :class:`~graph_rag.stages.kg_build.KgBuildStage` runs over a
:class:`~graph_rag.fakes.FakeLLMClient` (canned triples) and the real
:class:`~graph_rag.stages.entity_linking.EntityLinkingStage` over
:class:`~graph_rag.fakes.InMemoryEntityStore` + :class:`~graph_rag.fakes.FakeEmbedder`;
the graph is written to :class:`~graph_rag.fakes.InMemoryGraphStore` (only the ports
are faked, ADR-0010).

Covers the V5 demo (triples over canonical IDs; multi-label/type nodes; full
edge provenance with offsets resolved from OUR segmentation; a rare relation →
``RELATED_TO`` + ``raw_predicate``; a dated fact → edge qualifier, no DATE node)
plus the graph-idempotency GAP (re-ingest REPLACES a doc's edges) and the surviving
V1–V4 guarantees (deterministic id; log-and-drop, incl. a KG-build failure).
"""

from __future__ import annotations

import hashlib

from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    FakeNerStage,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.ids import document_id
from graph_rag.models import CorefCluster, IngestTrigger, Mention, Sentence
from graph_rag.normalize import normalize_name
from graph_rag.orchestrator import Orchestrator
from graph_rag.predicates import Predicate
from graph_rag.stages.coref import FakeCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.kg_build import KgBuildStage, LLMTriple, TripleList

BUCKET = "documents"

# A two-sentence document: a WORKS_FOR fact (sentence 0) and a dated LOCATED_IN
# fact (sentence 1). The offsets below are the exact spans of each sentence.
KEY = "acme.md"
SENT0 = "Ada Lovelace works for Acme Corp."
SENT1 = "Acme Corp is based in London on 2001-05-04."
TEXT = f"{SENT0} {SENT1}"

ADA = "Ada Lovelace"
ACME = "Acme Corp"
LONDON = "London"


def _canonical_id(entity_type: str, surface: str) -> str:
    """Recompute EL's deterministic create-new ``canonical_id`` (ADR-0004 scheme).

    Mirrors :meth:`EntityLinkingStage._mint_id` for a first-seen entity so the
    canned LLM triples can reference the exact canonical IDs the pipeline mints —
    the ids are a pure function of ``type`` + normalized surface.
    """
    base = f"el:{entity_type}:{normalize_name(surface)}"
    return "e-" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


ADA_ID = _canonical_id("PERSON", ADA)
ACME_ID = _canonical_id("ORG", ACME)
LONDON_ID = _canonical_id("LOCATION", LONDON)


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


def _canned_triples(*, rare: bool = False) -> TripleList:
    """The canned LLM triples the KG-build stage receives for the fixture doc.

    Two facts: a clean-mapping ``WORKS_FOR`` (sentence 0) and a dated
    ``LOCATED_IN`` (sentence 1). ``rare=True`` swaps the first predicate for a
    phrase outside the closed set to exercise the ``RELATED_TO`` fallback.
    """
    first_predicate = "secretly admires" if rare else "employed by"
    return TripleList(
        triples=[
            LLMTriple(
                subject_id=ADA_ID, predicate=first_predicate, object_id=ACME_ID, sentence_index=0
            ),
            LLMTriple(
                subject_id=ACME_ID,
                predicate="based in",
                object_id=LONDON_ID,
                sentence_index=1,
                date="2001-05-04",
                confidence=0.88,
            ),
        ]
    )


def _run(
    *,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    graph_store: InMemoryGraphStore,
    embedder: FakeEmbedder,
    triple_response: TripleList,
    key: str = KEY,
    text: str = TEXT,
):
    """Ingest one doc through a fresh orchestrator sharing the stores + graph."""
    object_store.put(BUCKET, key, text.encode())
    ner, coref = _stages()
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=EntityLinkingStage(entity_store, embedder),
        graph_store=graph_store,
        kg_build_stage=KgBuildStage(FakeLLMClient(structured_response=triple_response)),
    )
    return orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=key))


def _fresh() -> tuple[
    InMemoryObjectStore,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    FakeEmbedder,
]:
    """A fresh set of shared fakes for one scenario."""
    return (
        InMemoryObjectStore(),
        InMemoryDocumentStore(),
        InMemoryEntityStore(),
        InMemoryGraphStore(),
        FakeEmbedder(),
    )


def test_triples_reference_canonical_ids_not_strings() -> None:
    """Every written edge's subject/object is a canonical ID present in the EL store."""
    obj, doc, ent, graph, emb = _fresh()
    result = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert result is not None
    assert len(result.triples) == 2
    known_ids = {e.canonical_id for e in ent.all()}
    for triple in result.triples:
        assert triple.subject_id in known_ids
        assert triple.object_id in known_ids
    # The graph holds the same edges (written at the checkpoint).
    assert graph.edge_count() == 2


def test_nodes_are_multi_label_and_carry_type() -> None:
    """Each canonical entity becomes a node carrying its type (the second label)."""
    obj, doc, ent, graph, emb = _fresh()
    _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert graph.node_count() == 3
    assert graph.get_node(ADA_ID).type == "PERSON"  # type: ignore[union-attr]
    assert graph.get_node(ACME_ID).type == "ORG"  # type: ignore[union-attr]
    assert graph.get_node(LONDON_ID).type == "LOCATION"  # type: ignore[union-attr]


def test_edges_carry_full_provenance_resolved_from_segmentation() -> None:
    """Edges carry doc/sentence/span/confidence, offsets resolved from OUR segmentation."""
    obj, doc, ent, graph, emb = _fresh()
    result = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert result is not None
    doc_id = document_id(BUCKET, KEY)
    by_pred = {t.predicate: t for t in result.triples}

    works_for = by_pred[Predicate.WORKS_FOR]
    prov = works_for.provenance
    assert prov.source_doc_id == doc_id
    assert prov.sentence_index == 0
    assert prov.source_sentence == SENT0
    assert prov.char_start == 0
    assert prov.char_end == len(SENT0)
    assert TEXT[prov.char_start : prov.char_end] == SENT0

    located_in = by_pred[Predicate.LOCATED_IN]
    lprov = located_in.provenance
    assert lprov.sentence_index == 1
    assert lprov.source_sentence == SENT1
    assert TEXT[lprov.char_start : lprov.char_end] == SENT1
    assert lprov.confidence == 0.88


def test_rare_predicate_falls_back_to_related_to_with_raw() -> None:
    """A rare relation → ``RELATED_TO`` edge preserving the model's phrase."""
    obj, doc, ent, graph, emb = _fresh()
    result = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(rare=True),
    )
    assert result is not None
    rare = next(t for t in result.triples if t.subject_id == ADA_ID)
    assert rare.predicate == Predicate.RELATED_TO
    assert rare.provenance.raw_predicate == "secretly admires"


def test_dated_fact_is_edge_qualifier_no_date_node() -> None:
    """A dated fact sets the edge ``date`` qualifier; NO DATE node is created."""
    obj, doc, ent, graph, emb = _fresh()
    result = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert result is not None
    located_in = next(t for t in result.triples if t.predicate == Predicate.LOCATED_IN)
    assert located_in.date == "2001-05-04"
    # Only the three entity nodes exist — the date is not a node.
    assert graph.node_count() == 3
    assert {n.type for n in graph.khop([ADA_ID, ACME_ID, LONDON_ID], hops=1).nodes} == {
        "PERSON",
        "ORG",
        "LOCATION",
    }


def test_reingest_replaces_edges_no_duplication() -> None:
    """GRAPH IDEMPOTENCY (the gap): re-ingesting a doc REPLACES its edges (R1.5)."""
    obj, doc, ent, graph, emb = _fresh()
    first = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert first is not None
    assert graph.edge_count() == 2

    second = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert second is not None
    # Delete-then-write at the checkpoint keeps the edge count stable (no dup).
    assert graph.edge_count() == 2
    assert graph.node_count() == 3
    # ES + entity stores stayed idempotent too (V1/V4 guarantees hold).
    assert len(doc._records) == 1  # noqa: SLF001 — asserting no duplication.
    assert ent.count() == 3


def test_deterministic_id_under_v5() -> None:
    """V1 guarantee still holds: the record uses the deterministic document id."""
    obj, doc, ent, graph, emb = _fresh()
    result = _run(
        object_store=obj,
        document_store=doc,
        entity_store=ent,
        graph_store=graph,
        embedder=emb,
        triple_response=_canned_triples(),
    )
    assert result is not None
    assert result.record.document_id == document_id(BUCKET, KEY)


def test_kg_build_failure_is_logged_and_dropped() -> None:
    """A failing KG-build stage is dropped (returns None), not raised (ADR-0001)."""

    class ExplodingKgStage:
        """KG-build stage whose build always fails, to exercise log-and-drop."""

        def build(self, document_id, text, sentences, el_result, canonical_entities):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated KG-build failure")

    obj, doc, ent, graph, emb = _fresh()
    obj.put(BUCKET, KEY, TEXT.encode())
    ner, coref = _stages()
    orchestrator = Orchestrator(
        object_store=obj,
        document_store=doc,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=EntityLinkingStage(ent, emb),
        graph_store=graph,
        kg_build_stage=ExplodingKgStage(),  # type: ignore[arg-type]
    )
    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))
    assert result is None
    assert graph.edge_count() == 0  # nothing written for the dropped doc


def test_kg_build_skipped_without_graph_store() -> None:
    """No graph store wired → no triples built, V1–V4 behaviour unchanged."""
    obj, doc, ent, _graph, emb = _fresh()
    obj.put(BUCKET, KEY, TEXT.encode())
    ner, coref = _stages()
    orchestrator = Orchestrator(
        object_store=obj,
        document_store=doc,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=EntityLinkingStage(ent, emb),
        # graph_store + kg_build_stage omitted → checkpoint skipped.
    )
    result = orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=KEY))
    assert result is not None
    assert result.triples == []
    # EL still ran (V4 guarantee): canonical entities exist.
    assert ent.count() == 3
