"""The optional LLM judge and its measured error rates (GRA-89)."""

from __future__ import annotations

import json

import pytest
from kgbench import judge
from kgbench.cli import _DATA, main
from kggen.llm import CachedChat, ResponseCache
from kggen.questions import QuestionSpec, generate_questions
from kggen.spec import load_preset
from kggen.world import generate

FACT = {
    "fact_id": "F1",
    "text": "Aa Works is part of Bb Group",
    "subject_names": ["Aa Works"],
    "object_names": ["Bb Group"],
    "weight": 1.0,
}


class Scripted:
    """A client that answers from a function and counts the calls."""

    model = "m"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, messages):
        self.calls += 1
        return self.reply(messages)


def chat(tmp_path, reply):
    client = Scripted(reply)
    return CachedChat(ResponseCache(tmp_path), client), client


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("YES", True), ("no.", False), (" Yes, it does", True), ("Maybe", None), ("", None)],
)
def test_verdict_parsing(tmp_path, reply, expected):
    c, _ = chat(tmp_path, lambda m: reply)
    assert judge.llm_judge(c)("some answer", FACT) is expected


def test_only_a_valid_verdict_is_cached(tmp_path):
    c, client = chat(tmp_path, lambda m: "Maybe")
    judge.llm_judge(c)("an answer", FACT)
    assert list(tmp_path.iterdir()) == []
    c, client = chat(tmp_path, lambda m: "YES")
    j = judge.llm_judge(c)
    j("an answer", FACT), j("an answer", FACT)
    assert client.calls == 1  # the second ask is a cache hit


def test_the_judge_declines_a_stale_claim(tmp_path):
    c, client = chat(tmp_path, lambda m: "YES")
    assert judge.llm_judge(c)("an answer", {**FACT, "reason": "superseded"}) is None
    assert client.calls == 0


@pytest.fixture(scope="module")
def cases():
    world = generate(load_preset("small"))
    spec = QuestionSpec(n_questions=12, type_mix={"open_ended": 1.0}, hop_mix={1: 1.0, 2: 1.0})
    return judge.build_cases(generate_questions(world, spec))


def test_the_committed_cases_are_the_ones_the_code_builds(cases):
    assert json.loads((_DATA / "judge_cases.json").read_text()) == cases


def test_the_matcher_is_fooled_where_the_cases_say_it_is(cases):
    report = judge.evaluate(cases, lambda a, f: None)  # the judge declines everything
    assert report.abstained == len(cases)
    assert report.matcher == report.judge
    assert report.by_kind["negated"]["matcher"][0] == 0  # names alone read a denial as covered
    assert report.by_kind["pronoun"]["matcher"][0] == 0  # and miss a fact stated about "it"
    assert report.by_kind["stated"]["matcher"][0] == report.by_kind["stated"]["matcher"][1]


def test_a_perfect_judge_has_no_errors(cases):
    truth = {(c["answer"], c["fact"]["fact_id"]): c["label"] for c in cases}
    report = judge.evaluate(cases, lambda a, f: truth[a, f["fact_id"]])
    assert (report.judge.fp, report.judge.fn) == (0, 0)
    assert report.matcher.fp > 0 and report.matcher.fn > 0


def test_replaying_the_committed_answers_reproduces_the_published_rates(capsys):
    assert main(["judge"]) == 0  # no endpoint: every answer must come from the cache
    published = json.loads((_DATA.parents[2] / "docs/results/judge.json").read_text())
    out = capsys.readouterr().out
    assert f"{published['judge']['fp']}/" in out and f"{published['judge']['fn']}/" in out


def test_hard_cases_keep_only_answers_that_name_the_right_entities(tmp_path, cases):
    from kgbench import judge_hard

    world = generate(load_preset("small"))
    spec = QuestionSpec(n_questions=12, type_mix={"open_ended": 1.0}, hop_mix={1: 1.0, 2: 1.0})
    questions = generate_questions(world, spec)

    # a writer that echoes its instruction names every entity the instruction mentions
    echo, _ = chat(tmp_path / "echo", lambda m: m[-1]["content"])
    made, rejected = judge_hard.build_hard_cases(world, questions, echo, facts=6)
    # the echo of a reversal quotes the true fact in its original order, so that kind is rejected
    assert {c["kind"] for c in made} == set(judge_hard.HARD_KINDS) - {"reversed"}
    assert all(c["label"] == judge_hard.HARD_KINDS[c["kind"]] for c in made)
    assert len(made) + rejected >= 6 * (len(judge_hard.HARD_KINDS) - 1)
    # accepted answers are cached and a rejected one is not: a re-run asks only for the latter
    again, client = chat(tmp_path / "echo", lambda m: "unused")
    judge_hard.build_hard_cases(world, questions, again, facts=6)
    assert client.calls == rejected

    # a writer that names nobody is rejected throughout, and caches nothing
    mute, _ = chat(tmp_path / "mute", lambda m: "Nothing of note.")
    none, rejected = judge_hard.build_hard_cases(world, questions, mute, facts=6)
    assert [c for c in none if c["kind"] != "other_only"] == []
    assert not list((tmp_path / "mute").glob("*.json")) or all(
        c["kind"] == "other_only" for c in none
    )


def test_a_symmetric_relation_is_never_reversed():
    from kgbench.judge_hard import _pool

    world = generate(load_preset("small"))
    spec = QuestionSpec(n_questions=12, type_mix={"open_ended": 1.0}, hop_mix={1: 1.0, 2: 1.0})
    pool = _pool(world, generate_questions(world, spec))
    assert pool and all(f["predicate"] != "AFFILIATED_WITH" for f in pool)


def test_replaying_the_committed_hard_answers_reproduces_the_published_rates(capsys):
    assert main(["judge", "--hard"]) == 0
    published = json.loads((_DATA.parents[2] / "docs/results/judge-hard.json").read_text())
    out = capsys.readouterr().out
    assert f"{published['judge']['fp']}/" in out and f"{published['judge']['fn']}/" in out
    assert published["cases"] == 99 and published["judge"]["fp"] <= 1
