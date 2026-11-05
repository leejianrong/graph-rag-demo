"""Re-answering chain questions with names as the identity (GRA-124)."""

from __future__ import annotations

import json

from kgbench import ambiguity
from kggen.questions import QuestionSpec, generate_questions
from kggen.spec import WorldSpec
from kggen.world import generate


def entity(i: str, name: str, kind: str = "ORG", aliases: tuple[str, ...] = ()) -> dict:
    return {"id": i, "name": name, "type": kind, "aliases": list(aliases)}


def edge(i: str, subject: str, predicate: str, obj: str) -> dict:
    return {"id": i, "subject": subject, "predicate": predicate, "object": obj, "valid_to": None}


def question(answers: list[str]) -> dict:
    return {
        "id": "Q",
        "type": "entity",
        "start": "S",
        "answers": answers,
        "gold_paths": [["F1"]],
        "shared_name": True,
    }


def test_two_entities_with_the_same_name_give_a_second_answer():
    world = {
        "entities": [entity("S", "Start"), entity("A", "Aa"), entity("B", "Bb", aliases=("Aa",))],
        "edges": [edge("F1", "S", "OWNS", "A"), edge("F2", "S", "OWNS", "B")],
    }
    # the gold is A; B carries A's name as an alias, so they are one answer to a name reader
    assert ambiguity.measure([question(["A"])], world)["entity"]["ambiguous"] == 0
    # but a differently named second owned entity is a real extra answer
    world["entities"].append(entity("C", "Cc"))
    world["edges"].append(edge("F3", "S", "OWNS", "C"))
    assert ambiguity.measure([question(["A"])], world)["entity"]["ambiguous"] == 1


def test_a_shared_name_on_the_way_opens_a_second_branch_of_the_same_type():
    world = {
        "entities": [
            entity("S", "Start"),
            entity("M1", "Mid"),
            entity("M2", "Mid"),
            entity("X", "Xx"),
            entity("Y", "Yy"),
        ],
        "edges": [
            edge("F1", "S", "OWNS", "M1"),
            edge("F2", "M1", "OWNS", "X"),
            edge("F3", "M2", "OWNS", "Y"),
        ],
    }
    q = {**question(["X"]), "gold_paths": [["F1", "F2"]]}
    assert ambiguity.measure([q], world)["entity"]["extra"] == 1  # Yy, through the other "Mid"


def test_a_world_with_no_shared_names_has_no_extra_answers():
    world = generate(
        WorldSpec.model_validate({"n_entities": 200, "avg_degree": 4, "seed": 1, "alias_rate": 0.0})
    )
    questions = [
        q.to_dict()
        for q in generate_questions(world, QuestionSpec(n_questions=60, type_mix={"entity": 1}))
    ]
    result = ambiguity.measure(questions, json.loads(world.to_json()))
    assert result["entity"]["ambiguous"] == 0 and result["entity"]["flagged"] == 0


def test_table_renders():
    world = {
        "entities": [entity("S", "Start"), entity("A", "Aa")],
        "edges": [edge("F1", "S", "OWNS", "A")],
    }
    text = ambiguity.format_table(ambiguity.measure([question(["A"])], world))
    assert "entity" in text and "0%" in text
