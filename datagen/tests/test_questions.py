"""Questions come from the world model, carry gold paths, and have a unique answer."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from kggen.cli import main
from kggen.questions import VERB, QuestionSpec, generate_questions
from kggen.spec import WorldSpec
from kggen.world import WorldGenerationError, generate


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": 500, "avg_degree": 5, "seed": 2, "alias_rate": 0.5}
    return WorldSpec.model_validate({**base, **overrides})


@pytest.fixture(scope="module")
def world():
    return generate(make(churn_rate=0.1, retraction_rate=0.05, homonym_rate=0.05))


def test_every_question_has_a_gold_path_of_real_final_state_facts(world) -> None:
    current = {e.id: e for e in world.current_edges()}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        assert len(q.supporting_facts) == q.hops
        assert all(f in current for f in q.supporting_facts)


def test_the_gold_path_is_a_connected_chain_from_start_to_answer(world) -> None:
    facts = {e.id: e for e in world.edges}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        at = q.start
        for fid in q.supporting_facts:
            e = facts[fid]
            assert at in (e.subject, e.object)
            at = e.object if e.subject == at else e.subject
        assert at == q.answer


def test_answers_are_unique_in_the_final_state(world) -> None:
    facts = {e.id: e for e in world.edges}
    current = world.current_edges()
    for q in generate_questions(world, QuestionSpec(n_questions=40)):
        frontier = {q.start}
        for fid in q.supporting_facts:
            e = facts[fid]
            forward = e.subject in frontier
            frontier = {
                (c.object if forward else c.subject)
                for c in current
                if c.predicate == e.predicate and (c.subject if forward else c.object) in frontier
            }
        assert frontier == {q.answer}


def test_hop_distribution_follows_the_spec(world) -> None:
    mix = {1: 1.0, 2: 3.0, 4: 1.0}
    qs = generate_questions(world, QuestionSpec(n_questions=50, hop_mix=mix))
    assert Counter(q.hops for q in qs) == {1: 10, 2: 30, 4: 10}


def test_questions_name_a_uniquely_named_start_and_accept_aliases(world) -> None:
    names = Counter(e.name for e in world.entities)
    by_id = {e.id: e for e in world.entities}
    for q in generate_questions(world, QuestionSpec(n_questions=60)):
        start = by_id[q.start]
        assert names[start.name] == 1 and start.name in q.question
        assert q.accepted_answers[0] == q.answer_name
        assert set(q.accepted_answers[1:]) == set(by_id[q.answer].aliases)
        assert not by_id[q.answer].distractor and not start.distractor


def test_questions_read_from_the_relation_wording(world) -> None:
    facts = {e.id: e for e in world.edges}
    for q in generate_questions(world, QuestionSpec(n_questions=30)):
        assert q.question.startswith("Name the ") and q.question.endswith(".")
        assert all(VERB[facts[f].predicate] in q.question for f in q.supporting_facts)


def test_generation_is_deterministic_and_seeded(world) -> None:
    a = generate_questions(world, QuestionSpec(n_questions=30, seed=1))
    assert a == generate_questions(world, QuestionSpec(n_questions=30, seed=1))
    assert a != generate_questions(world, QuestionSpec(n_questions=30, seed=2))
    assert len({q.id for q in a}) == len(a) == 30


def test_an_impossible_request_is_an_error() -> None:
    tiny = generate(make(n_entities=6, avg_degree=1, alias_rate=0.0))
    with pytest.raises(WorldGenerationError):
        generate_questions(tiny, QuestionSpec(n_questions=500, hop_mix={4: 1.0}))


@pytest.mark.parametrize("bad", [{}, {0: 1.0}, {5: 1.0}, {1: -1.0}, {1: 0.0}])
def test_invalid_hop_mix_is_rejected(bad: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        QuestionSpec(hop_mix=bad)


def test_cli_writes_questions(tmp_path) -> None:
    out = tmp_path / "q.json"
    code = main(
        [
            "questions",
            "generate",
            "--preset",
            "small",
            "--n",
            "12",
            "--hops",
            "1:1,2:1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert len(data) == 12 and {q["hops"] for q in data} == {1, 2}
    assert all(q["supporting_facts"] for q in data)
