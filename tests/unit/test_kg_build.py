"""Unit tests for the V5 KG-build stage (fast, offline, ``$0``).

Exercises :class:`~graph_rag.stages.kg_build.KgBuildStage` at the LLM-port seam
against a :class:`~graph_rag.fakes.FakeLLMClient` with canned triples (ADR-0010) —
no Docker, no model, no provider. Pins the load-bearing per-triple rules
(ADR-0006, TESTING §4):

* predicate mapping to the closed set + the ``RELATED_TO`` fallback preserving the
  raw phrase;
* **provenance offset resolution** — a cited ``sentence_index`` resolves to the
  correct sentence's ``char_start``/``char_end`` from OUR segmentation, never the
  LLM;
* DATE modeled as the edge qualifier (never a node);
* triples referencing unknown canonical IDs are dropped;
* an out-of-range ``sentence_index`` is dropped.
"""

from __future__ import annotations

from graph_rag.fakes import FakeLLMClient
from graph_rag.models import CanonicalEntity, Sentence
from graph_rag.predicates import Predicate
from graph_rag.stages.entity_linking import ELResult
from graph_rag.stages.kg_build import KgBuildStage, LLMTriple, TripleList

DOC_ID = "doc-1"

# A two-sentence document; the offsets below are the exact spans of each sentence.
SENT0 = "Ada Lovelace works for Acme Corp."
SENT1 = "Acme Corp is based in London on 2001-05-04."
TEXT = f"{SENT0} {SENT1}"
SENTENCES = [
    Sentence(text=SENT0, char_start=0, char_end=len(SENT0), index=0),
    Sentence(text=SENT1, char_start=len(SENT0) + 1, char_end=len(TEXT), index=1),
]

# This document's canonical entities (the only allowed subject/object IDs).
ADA = CanonicalEntity(canonical_id="e-ada", name="Ada Lovelace", type="PERSON")
ACME = CanonicalEntity(canonical_id="e-acme", name="Acme Corp", type="ORG")
LONDON = CanonicalEntity(canonical_id="e-london", name="London", type="LOCATION")
CANONICALS = [ADA, ACME, LONDON]


def _stage(*triples: LLMTriple) -> KgBuildStage:
    """A KG-build stage whose fake LLM returns exactly ``triples``."""
    return KgBuildStage(FakeLLMClient(structured_response=TripleList(triples=list(triples))))


def _build(stage: KgBuildStage):
    """Run the stage over the fixed doc + canonical entities."""
    return stage.build(DOC_ID, TEXT, SENTENCES, ELResult(links=[], sentence_vectors=[]), CANONICALS)


def test_known_predicate_maps_to_closed_set_no_raw_predicate() -> None:
    """A phrase matching the closed set maps cleanly (no ``raw_predicate`` kept)."""
    (triple,) = _build(
        _stage(
            LLMTriple(
                subject_id="e-ada", predicate="employed by", object_id="e-acme", sentence_index=0
            )
        )
    )
    assert triple.predicate == Predicate.WORKS_FOR
    assert triple.provenance.raw_predicate is None


def test_unknown_predicate_falls_back_to_related_to_preserving_raw() -> None:
    """A rare/unknown phrase → ``RELATED_TO`` with the original phrase preserved."""
    (triple,) = _build(
        _stage(
            LLMTriple(
                subject_id="e-ada",
                predicate="secretly admires",
                object_id="e-acme",
                sentence_index=0,
            )
        )
    )
    assert triple.predicate == Predicate.RELATED_TO
    assert triple.provenance.raw_predicate == "secretly admires"


def test_provenance_offsets_resolved_from_segmentation() -> None:
    """The cited ``sentence_index`` fills the sentence text + span from OUR segmentation.

    The LLM cites only an index; the stage resolves ``source_sentence`` +
    ``char_start``/``char_end`` from the supplied sentences — and ``TEXT`` sliced
    by those offsets equals the sentence text.
    """
    (triple,) = _build(
        _stage(
            LLMTriple(
                subject_id="e-acme", predicate="located in", object_id="e-london", sentence_index=1
            )
        )
    )
    prov = triple.provenance
    assert prov.sentence_index == 1
    assert prov.source_sentence == SENT1
    assert prov.char_start == len(SENT0) + 1
    assert prov.char_end == len(TEXT)
    # The resolved offsets index back into the raw text (provenance is trustworthy).
    assert TEXT[prov.char_start : prov.char_end] == SENT1


def test_date_is_edge_qualifier_not_a_node() -> None:
    """A dated fact sets ``Triple.date``; no DATE entity/node is introduced."""
    (triple,) = _build(
        _stage(
            LLMTriple(
                subject_id="e-acme",
                predicate="located in",
                object_id="e-london",
                sentence_index=1,
                date="2001-05-04",
            )
        )
    )
    assert triple.date == "2001-05-04"
    # The subject/object are the entity nodes — never a date.
    assert {triple.subject_id, triple.object_id} == {"e-acme", "e-london"}


def test_confidence_is_carried_onto_provenance() -> None:
    """The model's optional confidence lands on the edge provenance."""
    (triple,) = _build(
        _stage(
            LLMTriple(
                subject_id="e-ada",
                predicate="works for",
                object_id="e-acme",
                sentence_index=0,
                confidence=0.91,
            )
        )
    )
    assert triple.provenance.confidence == 0.91
    assert triple.provenance.source_doc_id == DOC_ID


def test_triple_with_unknown_canonical_id_is_dropped() -> None:
    """A subject/object not among the doc's canonical IDs is dropped, not written."""
    triples = _build(
        _stage(
            LLMTriple(
                subject_id="e-ada", predicate="works for", object_id="e-ghost", sentence_index=0
            ),
            LLMTriple(
                subject_id="e-ada", predicate="works for", object_id="e-acme", sentence_index=0
            ),
        )
    )
    assert len(triples) == 1
    assert triples[0].object_id == "e-acme"


def test_out_of_range_sentence_index_is_dropped() -> None:
    """A citation to a non-existent sentence is logged-and-skipped, not resolved."""
    triples = _build(
        _stage(
            LLMTriple(
                subject_id="e-ada", predicate="works for", object_id="e-acme", sentence_index=9
            ),
            LLMTriple(
                subject_id="e-acme", predicate="located in", object_id="e-london", sentence_index=1
            ),
        )
    )
    assert len(triples) == 1
    assert triples[0].provenance.sentence_index == 1


def test_no_canonical_entities_skips_the_llm() -> None:
    """With no canonical entities there are no valid edges, and the LLM is not called."""
    llm = FakeLLMClient(structured_response=TripleList(triples=[]))
    stage = KgBuildStage(llm)
    triples = stage.build(DOC_ID, TEXT, SENTENCES, ELResult(links=[], sentence_vectors=[]), [])
    assert triples == []
    assert llm.calls == 0  # short-circuited before the provider call
