"""Typed-answer scoring: each answer_kind has a scorer, and a refusal is judged apart (ADR-0013)."""

from __future__ import annotations

import dataclasses

import pytest
from kgbench.answers import Answer, evaluate_answers, score_answer
from kggen.questions import QUESTION_TYPES, QuestionSpec, generate_questions
from kggen.spec import WorldSpec
from kggen.world import generate


@pytest.fixture(scope="module")
def questions():
    world = generate(
        WorldSpec(
            n_entities=600,
            avg_degree=6,
            seed=3,
            churn_rate=0.25,
            retraction_rate=0.1,
            alias_rate=0.5,
        )
    )
    spec = QuestionSpec(
        n_questions=120,
        hop_mix={1: 1.0, 2: 1.0, 3: 1.0},
        type_mix={t: 1.0 for t in QUESTION_TYPES},
    )
    return generate_questions(world, spec)


def perfect(q) -> Answer:
    """What a flawless system says for ``q``, built from its gold."""
    if q.answer_kind in ("abstain", "false_premise"):
        return Answer("I cannot answer that: there is no information in the documents.")
    if q.answer_kind == "number":
        return Answer(f"There are {q.number} of them.")
    if q.answer_kind == "set":
        return Answer("They are " + ", ".join(names[0] for names in q.answer_names) + ".")
    if q.answer_kind == "path":
        return Answer("They are linked.", path=q.gold_paths[0])
    if q.answer_kind == "open_ended":
        return Answer(" ".join(f"{r['text']}." for r in q.required_facts))
    return Answer(f"The answer is {q.answer_name}.")


def of_kind(questions, kind):
    return [q for q in questions if q.answer_kind == kind]


def test_every_kind_of_question_is_generated_for_these_tests(questions) -> None:
    assert {q.answer_kind for q in questions} >= {
        "entity",
        "set",
        "number",
        "path",
        "abstain",
        "false_premise",
        "open_ended",
    }


def test_a_flawless_answer_to_every_question_scores_correct(questions) -> None:
    for q in questions:
        score = score_answer(q, perfect(q))
        assert score.correct and score.score == 1.0, (q.id, q.type, score.detail)


def test_declining_everything_is_right_only_for_questions_that_should_be_declined(
    questions,
) -> None:
    report = evaluate_answers(questions, {q.id: Answer("I cannot say.") for q in questions})
    assert report.correct_refusal_rate == 1.0 and report.false_refusal_rate == 1.0
    assert report.answerable.correct == 0 and report.unanswerable.n > 0


def test_answering_everything_never_declines_and_misses_the_unanswerable(questions) -> None:
    gold = {
        q.id: (
            Answer("It is Ada.") if q.answer_kind in ("abstain", "false_premise") else perfect(q)
        )
        for q in questions
    }
    report = evaluate_answers(questions, gold)
    assert report.false_refusal_rate == 0.0 and report.correct_refusal_rate == 0.0
    assert report.answerable.correct == report.answerable.n


def test_a_missing_answer_is_wrong_and_not_a_refusal(questions) -> None:
    report = evaluate_answers(questions, {})
    assert report.answerable.correct == 0 and report.correct_refusal_rate == 0.0
    assert report.false_refusal_rate == 0.0


def test_the_report_splits_scores_by_type(questions) -> None:
    report = evaluate_answers(questions, {q.id: perfect(q) for q in questions})
    assert set(report.by_type) == {q.type for q in questions}
    assert all(t.correct == t.n for t in report.by_type.values())
    assert report.to_dict()["by_type"]["set"]["n"] == report.by_type["set"].n


def test_a_different_entity_is_not_the_answer(questions) -> None:
    plain = [q for q in of_kind(questions, "entity") if q.type == "entity"]
    a, b = plain[0], next(q for q in plain if q.answer != plain[0].answer)
    assert not score_answer(a, f"It is {b.answer_name}.").correct
    assert score_answer(a, f"it is {a.answer_name.upper()}").correct  # case does not matter


def test_a_stale_value_given_as_current_is_wrong_unless_marked_as_past(questions) -> None:
    q = next(q for q in questions if q.type == "latest" and q.stale_values)
    stale = q.stale_values[0]
    assert not score_answer(q, f"{q.answer_name}. It is also {stale}.").correct
    assert score_answer(q, f"{q.answer_name}. It was formerly {stale}.").correct
    assert not score_answer(q, f"It is {stale}.").correct


def test_a_set_is_scored_by_f1(questions) -> None:
    q = next(q for q in of_kind(questions, "set") if len(q.answers) >= 3)
    first = [names[0] for names in q.answer_names]
    half = score_answer(q, ", ".join(first[:2]) + ".")
    assert not half.correct and 0 < half.score < 1
    padded = score_answer(q, ", ".join([*first, "Nobody Atall"]))
    assert not padded.correct and 0 < padded.score < 1
    assert score_answer(q, "none of these").score == 0.0
    by_id = score_answer(q, Answer(entities=q.answers))
    assert by_id.correct
    assert not score_answer(q, Answer(entities=(*q.answers, "E-nobody"))).correct


def test_a_set_name_with_and_in_it_is_not_split(questions) -> None:
    q = next(q for q in of_kind(questions, "set") if len(q.answers) >= 2)
    names = [names[0] for names in q.answer_names]
    odd = dataclasses.replace(
        q, answer_names=(("Smith and Sons", "S&S"), *q.answer_names[1:]), answers=q.answers
    )
    text = "; ".join(["Smith and Sons", *names[1:]])
    assert score_answer(odd, text).correct


def test_a_count_is_checked_against_the_number(questions) -> None:
    q = next(q for q in of_kind(questions, "number") if q.number >= 2)
    assert score_answer(q, f"{q.number}").correct
    assert not score_answer(q, f"There are {q.number + 1}.").correct
    assert not score_answer(q, "Quite a few.").correct
    assert score_answer(q, Answer(number=q.number)).correct
    word = {2: "two", 3: "three", 4: "four", 5: "five"}.get(q.number)
    if word:
        assert score_answer(q, f"There are {word}.").correct
    big = dataclasses.replace(q, number=1234)
    assert score_answer(big, "That makes 1,234 in all.").correct


def test_a_path_needs_a_whole_shortest_chain(questions) -> None:
    q = next(q for q in of_kind(questions, "path") if len(q.gold_paths[0]) >= 2)
    chain = q.gold_paths[0]
    assert score_answer(q, Answer(path=chain)).correct
    assert score_answer(q, Answer(path=tuple(reversed(chain)))).correct
    part = score_answer(q, Answer(path=chain[:1]))
    assert not part.correct and 0 < part.score < 1
    assert score_answer(q, Answer(path=("F-none",))).score == 0.0
    nothing = score_answer(q, "They are connected somehow.")
    assert not nothing.correct and nothing.score == 0.0 and "cited" in nothing.detail


def test_a_path_with_several_shortest_chains_accepts_any_of_them(questions) -> None:
    many = [q for q in of_kind(questions, "path") if len(q.gold_paths) > 1]
    if not many:
        pytest.skip("no path question with two shortest chains in this world")
    q = many[0]
    assert all(score_answer(q, Answer(path=c)).correct for c in q.gold_paths)


def test_open_ended_needs_every_required_fact_and_no_stale_claim(questions) -> None:
    q = next(q for q in of_kind(questions, "open_ended") if len(q.required_facts) >= 2)
    first = f"{q.required_facts[0]['text']}."
    partial = score_answer(q, first)
    assert not partial.correct and 0 < partial.score < 1
    assert score_answer(q, perfect(q)).correct


def test_a_decline_phrase_inside_a_right_answer_is_a_false_refusal_only_when_wrong(
    questions,
) -> None:
    q = next(q for q in questions if q.type == "latest")
    ok = score_answer(q, f"{q.answer_name}, which replaced an earlier one that is unknown.")
    assert ok.correct
    wrong = score_answer(q, "I cannot tell.")
    assert not wrong.correct and wrong.refused and "declined" in wrong.detail


def test_a_structured_refusal_overrides_the_wording(questions) -> None:
    q = next(q for q in of_kind(questions, "abstain"))
    assert score_answer(q, Answer("Probably Ada.", refused=True)).correct
    assert not score_answer(q, Answer("I cannot say.", refused=False)).correct
