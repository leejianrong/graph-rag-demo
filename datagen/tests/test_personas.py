"""Question personas: a newcomer, a learner who converses, and an expert (GRA-187)."""

from __future__ import annotations

import json

import pytest
from kggen.cli import main
from kggen.questions import (
    PERSONA_NAMES,
    PERSONAS,
    QuestionSpec,
    generate_questions,
    with_persona,
)
from kggen.spec import load_preset
from kggen.world import generate


@pytest.fixture(scope="module")
def world():
    return generate(load_preset("small", "hard").with_overrides(name_style="bank"))


def ask(world, persona: str, n: int = 40, **changes):
    spec = with_persona(QuestionSpec(n_questions=n, phrasing="varied", **changes), persona)
    return generate_questions(world, spec)


def test_no_persona_changes_nothing(world) -> None:
    plain = generate_questions(world, QuestionSpec(n_questions=20))
    assert all(q.persona is None and not q.turns for q in plain)
    assert not {"persona", "turns"} & set(plain[0].to_dict())
    assert "persona" not in json.loads(QuestionSpec().model_dump_json())
    assert with_persona(QuestionSpec(), "none") == QuestionSpec()


def test_a_persona_sets_the_mix_and_an_unknown_one_is_refused() -> None:
    expert = with_persona(QuestionSpec(), "expert")
    assert expert.persona == "expert" and expert.type_mix == PERSONAS["expert"]["type_mix"]
    assert list(PERSONA_NAMES) == ["none", "newcomer", "learner", "expert", "mixed"]
    with pytest.raises(ValueError, match="unknown persona"):
        with_persona(QuestionSpec(), "wizard")


def test_questions_are_stamped_with_who_asked(world) -> None:
    for persona in ("newcomer", "learner", "expert"):
        found = ask(world, persona)
        assert found and {q.persona for q in found} == {persona}
        assert all(q.to_dict()["persona"] == persona for q in found)


# --- the newcomer ---------------------------------------------------------------------------------


def test_a_newcomer_asks_in_the_first_person_and_asks_who_of_a_person(world) -> None:
    found = ask(world, "newcomer", 60)
    entity = [q for q in found if q.type == "entity"]
    assert entity and all(q.question.endswith("?") for q in entity)
    people = [q for q in entity if world_type(world, q.answer) == "PERSON"]
    assert people and all("who" in q.question.lower() for q in people)
    others = [q for q in entity if world_type(world, q.answer) != "PERSON"]
    assert others and all("what" in q.question.lower() for q in others)
    open_ended = [q for q in found if q.type == "open_ended"]
    assert open_ended and all(not q.question.startswith("Summarize") for q in open_ended)
    assert {q.type for q in found} <= set(PERSONAS["newcomer"]["type_mix"])


def world_type(world, entity_id: str) -> str:
    return next(e.type for e in world.entities if e.id == entity_id)


# --- the learner ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conversations(world):
    found = [q for q in ask(world, "learner", 80) if q.type == "conversation"]
    assert found
    return found


def test_a_conversation_follows_each_answer_with_the_next_question(world, conversations) -> None:
    facts = {f.id: f for f in world.edges}
    for q in conversations:
        assert len(q.turns) == q.hops >= 2
        at = q.start
        for turn in q.turns:
            (fid,) = turn["supporting_facts"]
            f = facts[fid]
            assert at in (f.subject, f.object), "each turn starts from the last answer"
            at = f.object if f.subject == at else f.subject
            assert turn["answer"] == at
        assert q.answer == at and q.answer_name == q.turns[-1]["answer_name"]
        assert q.supporting_facts == tuple(t["supporting_facts"][0] for t in q.turns)


def test_the_last_turn_refers_back_with_a_pronoun_and_the_transcript_has_the_rest(
    world, conversations
) -> None:
    names = {e.id: e.name for e in world.entities}
    for q in conversations:
        last = q.question
        assert q.turns[-1]["question"] == last
        assert any(w in last.rstrip(".").split() for w in ("it", "they", "them"))
        assert names[q.start] not in last, "the last turn does not name where it started"
        text = q.transcript()
        assert text.count("Q:") == len(q.turns) and text.endswith(last)
        assert all(t["answer_name"] in text for t in q.turns[:-1])
    single = ask_plain_entity()
    assert single.transcript() == single.question


def ask_plain_entity():
    from kggen.questions import Question

    return Question("Q0", "Who?", 1, "E0", "E1", "x", ("x",), ("F0",))


def test_every_turn_has_one_answer_among_the_facts(world, conversations) -> None:
    live = world.current_edges()
    facts = {f.id: f for f in world.edges}
    for q in conversations:
        for turn in q.turns:
            f = facts[turn["supporting_facts"][0]]
            before = f.subject if turn["answer"] == f.object else f.object
            same = [
                e
                for e in live
                if e.predicate == f.predicate
                and ((e.subject == before) if turn["answer"] == f.object else (e.object == before))
            ]
            assert len(same) == 1


# --- the expert ---------------------------------------------------------------------------------


def test_an_expert_asks_what_two_entities_share(world) -> None:
    found = [q for q in ask(world, "expert", 60) if q.type == "common"]
    assert found
    facts = {f.id: f for f in world.edges}
    for q in found:
        assert 1 <= len(q.answers) <= 5 and q.grading == "set_f1" and q.hops == 2
        assert len(q.gold_paths) == len(q.answers)
        for answer, (first, second) in zip(q.answers, q.gold_paths, strict=True):
            a_side, b_side = facts[first], facts[second]
            assert answer in (a_side.subject, a_side.object)
            assert answer in (b_side.subject, b_side.object)
            ends = {a_side.subject, a_side.object, b_side.subject, b_side.object} - {answer}
            assert len(ends) == 2, "two different entities share it"
        assert q.question.count("and") >= 1


def test_the_expert_mix_has_the_hard_kinds(world) -> None:
    kinds = {q.type for q in ask(world, "expert", 80)}
    assert {"common", "path"} <= kinds
    assert kinds - {"entity"} <= set(PERSONAS["expert"]["type_mix"])  # entity makes up a shortfall


# --- mixed and the command line -------------------------------------------------------------------


def test_a_mixed_set_splits_between_all_three(world) -> None:
    found = generate_questions(world, QuestionSpec(n_questions=30, persona="mixed"))
    by = {p: sum(q.persona == p for q in found) for p in ("newcomer", "learner", "expert")}
    assert sum(by.values()) == len(found) == 30 and all(v == 10 for v in by.values())
    assert len({q.id for q in found}) == len(found)


def test_questions_are_repeatable(world) -> None:
    again = [q.question for q in ask(world, "learner", 30)]
    assert again == [q.question for q in ask(world, "learner", 30)]


def test_cli_persona_sets_the_mix_unless_types_are_given(tmp_path) -> None:
    out = tmp_path / "q.json"
    assert (
        main(
            [
                "questions",
                "generate",
                "--preset",
                "small",
                "--persona",
                "expert",
                "--n",
                "20",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    kinds = {q["type"] for q in json.loads(out.read_text())}
    assert "common" in kinds
    assert (
        main(
            [
                "questions",
                "generate",
                "--preset",
                "small",
                "--persona",
                "expert",
                "--types",
                "entity:1",
                "--n",
                "5",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads(out.read_text())
    assert {q["type"] for q in data} == {"entity"} and {q["persona"] for q in data} == {"expert"}
