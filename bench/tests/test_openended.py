"""The open-ended matcher: coverage of required facts, and stale claims the answer must not make."""

from __future__ import annotations

import pytest
from kgbench.openended import score_open_ended
from kggen.questions import QuestionSpec, generate_questions
from kggen.spec import WorldSpec
from kggen.world import generate


@pytest.fixture(scope="module")
def questions():
    world = generate(
        WorldSpec(
            n_entities=300,
            avg_degree=5,
            seed=2,
            churn_rate=0.2,
            retraction_rate=0.1,
            alias_rate=0.5,
        )
    )
    return generate_questions(
        world, QuestionSpec(n_questions=30, hop_mix={1: 1.0, 2: 1.0}, type_mix={"open_ended": 1.0})
    )


def _with_stale(questions):
    return next(q for q in questions if q.forbidden_claims)


def test_an_answer_stating_every_required_fact_gets_full_coverage(questions) -> None:
    for q in questions:
        answer = " ".join(f"{r['text']}." for r in q.required_facts)
        score = score_open_ended(q, answer)
        assert score.coverage == 1.0 and not score.missed and score.clean


def test_an_empty_answer_covers_nothing_and_claims_nothing(questions) -> None:
    score = score_open_ended(questions[0], "")
    assert (
        score.coverage == 0.0
        and score.clean
        and len(score.missed) == len(questions[0].required_facts)
    )


def test_coverage_is_weighted(questions) -> None:
    q = next(q for q in questions if q.hops == 2)
    weights = {r["fact_id"]: r["weight"] for r in q.required_facts}
    assert set(weights.values()) == {1.0, 0.5}
    context = [r for r in q.required_facts if r["weight"] == 0.5]
    answer = " ".join(f"{r['text']}." for r in q.required_facts if r["weight"] == 1.0)
    score = score_open_ended(q, answer)
    total = sum(weights.values())
    assert score.coverage == pytest.approx((total - 0.5 * len(context)) / total)


def test_a_fact_split_across_sentences_is_not_covered(questions) -> None:
    q = questions[0]
    r = q.required_facts[0]
    split = f"{r['subject_names'][0]} is mentioned here. Elsewhere, {r['object_names'][0]} appears."
    assert r["fact_id"] not in score_open_ended(q, split).covered


def test_aliases_and_case_count_but_substrings_do_not(questions) -> None:
    q = next(q for q in questions if any(len(r["object_names"]) > 1 for r in q.required_facts))
    r = next(r for r in q.required_facts if len(r["object_names"]) > 1)
    alias = f"{r['subject_names'][0].upper()} is tied to {r['object_names'][1].lower()}."
    assert r["fact_id"] in score_open_ended(q, alias).covered
    glued = f"{r['subject_names'][0]} relates to x{r['object_names'][0]}y."
    assert r["fact_id"] not in score_open_ended(q, glued).covered


def test_a_stale_claim_is_flagged_unless_the_answer_marks_it_as_past(questions) -> None:
    q = _with_stale(questions)
    claim = q.forbidden_claims[0]
    flat = f"{claim['text']}."
    assert claim["fact_id"] in score_open_ended(q, flat).forbidden
    for hedge in ("formerly", "no longer", "until recently"):
        past = f"{claim['subject_names'][0]} {hedge} had a tie with {claim['object_names'][0]}."
        assert claim["fact_id"] not in score_open_ended(q, past).forbidden
    assert not score_open_ended(q, flat).clean


def test_a_judge_decides_when_it_has_an_opinion_and_the_matcher_otherwise(questions) -> None:
    q = _with_stale(questions)
    fact = q.required_facts[0]
    never = lambda text, f: False  # noqa: E731
    assert score_open_ended(q, f"{fact['text']}.", judge=never).coverage == 0.0
    always = lambda text, f: True  # noqa: E731
    assert score_open_ended(q, "", judge=always).coverage == 1.0
    assert score_open_ended(q, "", judge=always).forbidden == tuple(
        c["fact_id"] for c in q.forbidden_claims
    )
    abstains = lambda text, f: None  # noqa: E731
    answer = f"{fact['text']}."
    assert score_open_ended(q, answer, judge=abstains) == score_open_ended(q, answer)


def test_other_question_types_are_rejected() -> None:
    world = generate(WorldSpec(n_entities=100, avg_degree=4, seed=1))
    (plain,) = generate_questions(world, QuestionSpec(n_questions=1))
    with pytest.raises(ValueError, match="not an open-ended one"):
        score_open_ended(plain, "anything")
