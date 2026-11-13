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
from graph_rag.stages.kg_build import KgBuildStage, LLMTriple, TripleList, build_kg_prompt

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


def test_prompt_lists_the_closed_predicate_vocabulary() -> None:
    """The prompt names the closed predicate set so the model prefers it (B7, E-fix).

    Handing the model the vocabulary sharply reduces vague phrases that fall back to
    the ``RELATED_TO`` open predicate, which was leaving the real-stack graph all
    generic edges.
    """
    prompt = build_kg_prompt(TEXT, CANONICALS)
    for predicate in Predicate:
        if predicate is Predicate.RELATED_TO:
            continue  # the reserved fallback, offered as a last resort not a menu item
        assert predicate.value in prompt
    assert "RELATED_TO only when" in prompt


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


def test_ceo_role_and_event_place_scripted_answers() -> None:
    """A CEO title lands on a HAS_ROLE edge and an event's place stays off the person (GRA-162).

    The scripted answer is what the model returns under the prompt's role and event rules; the
    stage keeps the title as an edge property (not an entity), an acquisition as ACQUIRED, and
    drops a stray ``role`` from a non-role edge.
    """
    sents = [
        "Acme acquired Northwind in 2020.",
        "Its CEO Klaus Weber works there.",
        "Maria Santos presented the prototype at the Hamburg Summit in Hamburg.",
    ]
    text = " ".join(sents)
    sentences, start = [], 0
    for i, s in enumerate(sents):
        sentences.append(Sentence(text=s, char_start=start, char_end=start + len(s), index=i))
        start += len(s) + 1
    ents = [
        CanonicalEntity(canonical_id="acme", name="Acme", type="ORG"),
        CanonicalEntity(canonical_id="nw", name="Northwind", type="ORG"),
        CanonicalEntity(canonical_id="klaus", name="Klaus Weber", type="PERSON"),
        CanonicalEntity(canonical_id="maria", name="Maria Santos", type="PERSON"),
        CanonicalEntity(canonical_id="summit", name="Hamburg Summit", type="EVENT"),
        CanonicalEntity(canonical_id="hh", name="Hamburg", type="LOCATION"),
    ]
    prompt = build_kg_prompt(text, ents)
    assert "HAS_ROLE" in prompt and "`role`" in prompt
    assert "not where the person is" in prompt and "ACQUIRED" in prompt

    stage = _stage(
        LLMTriple(
            subject_id="acme", predicate="acquired", object_id="nw", sentence_index=0, date="2020"
        ),
        LLMTriple(
            subject_id="klaus", predicate="HAS_ROLE", object_id="acme", sentence_index=1, role="CEO"
        ),
        LLMTriple(
            subject_id="klaus",
            predicate="WORKS_FOR",
            object_id="acme",
            sentence_index=1,
            role="CEO",
        ),
        LLMTriple(
            subject_id="maria", predicate="PARTICIPATED_IN", object_id="summit", sentence_index=2
        ),
        LLMTriple(subject_id="summit", predicate="LOCATED_IN", object_id="hh", sentence_index=2),
    )
    triples = stage.build(DOC_ID, text, sentences, ELResult(links=[], sentence_vectors=[]), ents)
    edges = {(t.subject_id, t.predicate, t.object_id): t for t in triples}

    assert edges[("acme", "ACQUIRED", "nw")].date == "2020"
    assert edges[("klaus", "HAS_ROLE", "acme")].role == "CEO"
    assert edges[("klaus", "WORKS_FOR", "acme")].role is None
    assert ("summit", "LOCATED_IN", "hh") in edges
    assert ("maria", "LOCATED_IN", "hh") not in edges
