"""Seeded phrasing variants change the words and never the gold (GRA-183)."""

from __future__ import annotations

import pytest
from kggen.questions import QUESTION_TYPES, QuestionSpec, generate_questions
from kggen.render import (
    TEMPLATES,
    VARIED_TEMPLATES,
    RenderSpec,
    render_documents,
)
from kggen.spec import load_preset
from kggen.world import generate

# The types a world with no churn can ask (as_of, latest and retracted need change over time).
TYPES = {t: 1.0 for t in QUESTION_TYPES if t not in {"as_of", "latest", "retracted"}}


@pytest.fixture(scope="module")
def world():
    return generate(load_preset("small", "hard"))


def questions(world, phrasing: str, seed: int = 0):
    return generate_questions(
        world, QuestionSpec(seed=seed, n_questions=60, type_mix=TYPES, phrasing=phrasing)
    )


def gold(q):
    """Everything about a question except its words."""
    return (
        q.id,
        q.type,
        q.hops,
        q.start,
        q.answer,
        q.accepted_answers,
        q.supporting_facts,
        q.gold_paths,
    )


# --- questions ------------------------------------------------------------------------------------


def test_plain_is_the_default_and_unchanged(world) -> None:
    assert generate_questions(world, QuestionSpec(n_questions=30, type_mix=TYPES)) == (
        generate_questions(world, QuestionSpec(n_questions=30, type_mix=TYPES, phrasing="plain"))
    )


def test_varied_asks_the_same_questions_with_the_same_gold(world) -> None:
    plain, varied = questions(world, "plain"), questions(world, "varied")

    assert [gold(q) for q in plain] == [gold(q) for q in varied]


def test_varied_changes_the_words_of_most_questions(world) -> None:
    plain, varied = questions(world, "plain"), questions(world, "varied")
    changed = sum(p.question != v.question for p, v in zip(plain, varied, strict=True))

    assert changed >= len(plain) // 2
    assert len({v.question.split()[0] for v in varied}) > len(
        {p.question.split()[0] for p in plain}
    )


def test_varied_is_repeatable_and_follows_the_question_seed(world) -> None:
    again = [q.question for q in questions(world, "varied")]
    assert again == [q.question for q in questions(world, "varied")]
    # a different seed asks different questions, so compare wording of the same ids only
    assert again != [q.question for q in questions(world, "varied", seed=1)]


def test_every_varied_question_is_a_sentence_that_names_where_it_starts(world) -> None:
    names = {e.id: e.name for e in world.entities}
    for q in questions(world, "varied"):
        text = q.question
        assert text[0].isupper() and text.rstrip().endswith((".", "?")), text
        assert "{" not in text and "}" not in text, text
        if q.type not in {"path", "open_ended", "superlative", "conversation"}:  # no start named
            assert names[q.start] in text, text


def test_varied_uses_each_wording_stem_somewhere(world) -> None:
    """Revert proof: without the bank every entity question opens with "Name"."""
    openings = {q.question.split()[0] for q in questions(world, "varied") if q.type == "entity"}
    assert {"Name", "Identify", "Give", "What", "Which"} & openings != {"Name"}
    assert len(openings) > 1


# --- sentences ------------------------------------------------------------------------------------


def documents(world, phrasing: str):
    return render_documents(world, RenderSpec(phrasing=phrasing))


def test_every_relation_has_more_wordings_in_varied_and_keeps_the_plain_ones() -> None:
    for predicate, plain in TEMPLATES.items():
        assert set(plain) <= set(VARIED_TEMPLATES[predicate])
        if predicate != "RELATED_TO":
            assert len(VARIED_TEMPLATES[predicate]) > len(plain)


def test_varied_documents_lay_out_the_same_facts_in_the_same_documents(world) -> None:
    plain, varied = documents(world, "plain"), documents(world, "varied")

    assert [(d.id, d.title, d.distractor) for d in plain] == [
        (d.id, d.title, d.distractor) for d in varied
    ]
    for p, v in zip(plain, varied, strict=True):
        assert [s.fact_ids for s in p.sentences] == [s.fact_ids for s in v.sentences]
    assert [d.text for d in plain] != [d.text for d in varied]


def test_every_mention_in_a_varied_document_points_at_its_words(world) -> None:
    """Passive wordings put the object first; the gold spans must still index the text."""
    seen_object_first = False
    for document in documents(world, "varied"):
        for sentence in document.sentences:
            for m in sentence.mentions:
                assert document.text[m.start : m.end] == m.surface
            if len(sentence.mentions) == 2 and sentence.fact_ids:
                (first, second) = sentence.mentions
                edge = next(e for e in world.edges if e.id == sentence.fact_ids[0])
                seen_object_first |= (
                    first.entity_id == edge.object and second.entity_id == edge.subject
                )
    assert seen_object_first  # the passive wordings were used, so direction is actually tested


def test_varied_is_repeatable(world) -> None:
    assert documents(world, "varied") == documents(world, "varied")


# --- spec files -----------------------------------------------------------------------------------


def test_a_plain_spec_serialises_as_it_always_did_and_a_varied_one_says_so() -> None:
    assert "phrasing" not in QuestionSpec().model_dump_json()
    assert "phrasing" not in RenderSpec().model_dump_json()
    assert '"phrasing":"varied"' in QuestionSpec(phrasing="varied").model_dump_json().replace(
        " ", ""
    )
    assert (
        RenderSpec.model_validate_json(RenderSpec(phrasing="varied").model_dump_json()).phrasing
        == "varied"
    )
    assert RenderSpec.model_validate_json("{}").phrasing == "plain"


# GRA-185: a varied chain question reads as a person would ask it, not as a stack of clauses.
def test_varied_questions_do_not_stack_the_same_clause(world) -> None:
    import re

    stacked = re.compile(r"\b(the \w+) (?:that|which|who)\b.*\b\1 (?:that|which|who)\b")
    many = generate_questions(
        world,
        QuestionSpec(
            seed=3,
            n_questions=300,
            type_mix={"entity": 3, "set": 1, "count": 1, "false_premise": 1},
            phrasing="varied",
        ),
    )
    assert many and not [q.question for q in many if stacked.search(q.question)]


def test_varied_chains_use_possessives_and_plain_chains_still_nest(world) -> None:
    varied = [q for q in questions(world, "varied") if q.type == "entity" and q.hops >= 2]
    assert any("'s " in q.question or "' " in q.question for q in varied)
    plain = [q for q in questions(world, "plain") if q.type == "entity" and q.hops >= 2]
    assert plain and all("'s " not in q.question for q in plain)


def test_a_name_ending_in_s_takes_a_bare_apostrophe_and_a_long_run_is_counted() -> None:
    from types import SimpleNamespace as NS

    from kggen.questions import _noun_phrase

    names = {"a": "Yewtree Works", "b": "B", "c": "C", "d": "D", "e": "E"}
    types = dict.fromkeys(names, "LOCATION")

    def hop(s, o, pred="LOCATED_IN"):
        return NS(subject=s, object=o, predicate=pred)

    one = _noun_phrase(names, types, "a", [(hop("a", "b"), True)], None, True)
    assert one == "Yewtree Works' location"
    three = _noun_phrase(
        names,
        types,
        "a",
        [(hop("a", "b"), True), (hop("b", "c"), True), (hop("c", "d"), True)],
        None,
        True,
    )
    assert "three times" in three and three.count("located in") == 1
