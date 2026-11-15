"""Questions can record, or avoid, a second answer that only names reveal (GRA-183)."""

from __future__ import annotations

import pytest
from kggen.ambiguity import NameReader
from kggen.questions import QuestionSpec, generate_questions
from kggen.spec import WorldSpec, load_preset
from kggen.world import Edge, Entity, World, generate

CHAINS = {"entity", "set", "count"}


def tiny_world(*, homonym: bool) -> World:
    """Ann works for Acme, which makes Widget. A second Acme (if ``homonym``) makes Gadget."""
    other = "Acme" if homonym else "Apex"
    return World(
        spec=WorldSpec(n_entities=5, avg_degree=2),
        entities=(
            Entity("P1", "Ann Ashgrove", "PERSON", 0),
            Entity("O1", "Acme", "ORG", 0),
            Entity("O2", other, "ORG", 0),
            Entity("X1", "Widget", "PRODUCT", 0),
            Entity("X2", "Gadget", "PRODUCT", 0),
        ),
        edges=(
            Edge("F1", "P1", "WORKS_FOR", "O1"),
            Edge("F2", "O1", "PRODUCES", "X1"),
            Edge("F3", "O2", "PRODUCES", "X2"),
        ),
    )


def test_two_companies_with_one_name_make_the_chain_ambiguous() -> None:
    """The hand-checkable case: names alone reach Widget and Gadget, the gold is Widget."""
    reader = NameReader(tiny_world(homonym=True))
    assert reader.ambiguous("P1", ("F1", "F2"), ("X1",)) is True
    assert len(reader.reached("P1", ("F1", "F2"))) == 2


def test_without_the_shared_name_the_same_chain_has_one_answer() -> None:
    """Revert proof: remove the homonym and the flag goes away."""
    reader = NameReader(tiny_world(homonym=False))
    assert reader.ambiguous("P1", ("F1", "F2"), ("X1",)) is False


def test_a_shared_answer_name_is_not_a_second_answer() -> None:
    """If the gold answer itself shares a name, both carriers are the same answer to a reader."""
    world = tiny_world(homonym=False)
    world = World(
        world.spec,
        (*world.entities[:4], Entity("X2", "Widget", "PRODUCT", 0)),
        (*world.edges[:2], Edge("F3", "O1", "PRODUCES", "X2")),
    )
    assert NameReader(world).ambiguous("P1", ("F1", "F2"), ("X1", "X2")) is False


# --- on generated worlds --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hard_world() -> World:
    spec = load_preset("small", "hard").with_overrides(homonym_rate=0.4, alias_rate=0.7, seed=3)
    return generate(spec)


def ask(world: World, ambiguity: str, n: int = 60):
    types = {"entity": 2.0, "set": 1.0, "count": 1.0, "compare": 1.0, "superlative": 1.0}
    return generate_questions(
        world,
        QuestionSpec(n_questions=n, type_mix=types, ambiguity=ambiguity),  # type: ignore[arg-type]
    )


def test_ignore_is_the_default_and_leaves_the_flag_out_of_the_file(hard_world: World) -> None:
    default = generate_questions(hard_world, QuestionSpec(n_questions=30))
    assert all(q.ambiguous_under_names is None for q in default)
    assert all("ambiguous_under_names" not in q.to_dict() for q in default)
    assert default == generate_questions(
        hard_world, QuestionSpec(n_questions=30, ambiguity="ignore")
    )


def test_mark_records_the_flag_on_chain_questions_and_changes_nothing_else(
    hard_world: World,
) -> None:
    plain, marked = ask(hard_world, "ignore"), ask(hard_world, "mark")

    def without(q):
        return {k: v for k, v in q.to_dict().items() if k != "ambiguous_under_names"}

    assert [without(q) for q in plain] == [without(q) for q in marked]
    assert all(isinstance(q.ambiguous_under_names, bool) for q in marked if q.type in CHAINS)
    assert all(q.ambiguous_under_names is None for q in marked if q.type not in CHAINS)
    assert any(q.ambiguous_under_names for q in marked)  # the knob made some


def test_an_ambiguous_question_always_touches_a_shared_name(hard_world: World) -> None:
    """shared_name is the warning; ambiguity is the confirmed case, so it is a subset."""
    marked = ask(hard_world, "mark")
    flagged = [q for q in marked if q.ambiguous_under_names]
    assert flagged and all(q.shared_name for q in flagged)
    warned = [q for q in marked if q.shared_name and q.type in CHAINS]
    assert len(flagged) < len(warned)  # the warning over-warns, which is why this exists


def test_drop_leaves_no_chain_question_a_name_reader_can_misanswer(hard_world: World) -> None:
    kept = ask(hard_world, "drop")
    reader = NameReader(hard_world)
    assert len(kept) == 60
    for q in kept:
        if q.type in CHAINS and q.gold_paths:
            assert not reader.ambiguous(q.start, q.gold_paths[0], q.answers), q.question


def test_the_default_ambiguity_is_not_written_to_a_spec_file() -> None:
    assert "ambiguity" not in QuestionSpec().model_dump_json()
    assert '"ambiguity":"drop"' in QuestionSpec(ambiguity="drop").model_dump_json().replace(" ", "")
    assert QuestionSpec.model_validate_json("{}").ambiguity == "ignore"
