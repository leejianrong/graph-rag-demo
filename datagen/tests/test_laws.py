"""LAW entities can be placed: a law attaches to what it governs through GOVERNED_BY (GRA-94)."""

from __future__ import annotations

import pytest
from kggen.questions import QUESTION_TYPES, QuestionSpec, generate_questions
from kggen.render import render_documents
from kggen.spec import PREDICATE_SIGNATURES, WorldSpec
from kggen.world import generate


def law_spec(**changes) -> WorldSpec:
    return WorldSpec(
        n_entities=400,
        avg_degree=5,
        seed=4,
        type_mix={"PERSON": 0.3, "ORG": 0.25, "LOCATION": 0.15, "PRODUCT": 0.1, "LAW": 0.2},
        **changes,
    )


def connected_laws(world) -> tuple[int, int]:
    laws = {e.id for e in world.entities if e.type == "LAW"}
    touched = {x for f in world.edges for x in (f.subject, f.object)}
    return len(laws & touched), len(laws)


def test_a_spec_with_laws_is_accepted_and_most_laws_take_part_in_a_fact() -> None:
    got, total = connected_laws(generate(law_spec()))
    assert total and got / total > 0.6


def test_a_heavier_governed_by_weight_connects_more_laws() -> None:
    """A rare relation leaves some of its entities out by chance; a heavier weight fixes that."""
    light = connected_laws(generate(law_spec()))
    mix = {p: 1.0 for p in PREDICATE_SIGNATURES} | {"GOVERNED_BY": 6.0}
    heavy = connected_laws(generate(law_spec(predicate_mix=mix)))
    assert heavy[0] / heavy[1] > light[0] / light[1] and heavy[0] / heavy[1] > 0.9


def test_laws_are_only_ever_the_object_of_governed_by() -> None:
    world = generate(law_spec())
    types = {e.id: e.type for e in world.entities}
    for fact in world.edges:
        if "LAW" in (types[fact.subject], types[fact.object]):
            assert fact.predicate == "GOVERNED_BY" and types[fact.object] == "LAW"
            assert (types[fact.subject], "LAW") in PREDICATE_SIGNATURES["GOVERNED_BY"]
    assert any(f.predicate == "GOVERNED_BY" for f in world.edges)


def test_a_world_without_laws_is_unchanged_by_the_new_relation() -> None:
    plain = WorldSpec(n_entities=200, avg_degree=4, seed=1)
    assert "GOVERNED_BY" not in {f.predicate for f in generate(plain).edges}


def test_laws_alone_cannot_form_a_world() -> None:
    with pytest.raises(ValueError):
        WorldSpec(n_entities=50, avg_degree=3, type_mix={"LAW": 1.0})


def test_a_law_fact_is_written_as_a_sentence_naming_both_entities() -> None:
    world = generate(law_spec())
    names = {e.id: e.name for e in world.entities}
    fact = next(f for f in world.edges if f.predicate == "GOVERNED_BY")
    sentence = next(
        s for d in render_documents(world) for s in d.sentences if fact.id in s.fact_ids
    )
    assert names[fact.subject] in sentence.text or sentence.mentions
    assert "governed by" in sentence.text or "regulated by" in sentence.text
    assert {m.entity_id for m in sentence.mentions} == {fact.subject, fact.object}


def test_questions_can_be_derived_through_a_law() -> None:
    world = generate(law_spec(churn_rate=0.2, retraction_rate=0.05))
    spec = QuestionSpec(
        n_questions=60, hop_mix={1: 1.0, 2: 1.0}, type_mix={t: 1.0 for t in QUESTION_TYPES}
    )
    asked = generate_questions(world, spec)
    via_law = [
        q
        for q in asked
        if any(
            f.id in set(q.supporting_facts) and f.predicate == "GOVERNED_BY" for f in world.edges
        )
    ]
    assert via_law
    assert any("governed by" in q.question for q in via_law)
