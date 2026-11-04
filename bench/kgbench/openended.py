"""Score an open-ended answer against a question's required facts and forbidden claims (ADR-0013).

An open-ended question ("Summarize what is known about X") has no single answer. Its gold is the
facts a good answer covers, each with a weight, and the claims it must not make: facts that
were replaced or withdrawn. The matcher is offline and deterministic, and works on sentences:

* a required fact is *covered* when one sentence of the answer names its subject and its object
  (by name or alias, whole words, any case);
* a forbidden claim is *made* when one sentence names its subject and its stale object, unless the
  sentence marks it as past ("formerly", "no longer", "until", ...). The check is a heuristic and
  errs toward not flagging; it cannot tell a careful denial from an assertion in every phrasing.

A judge can replace the matcher for any fact: a callable ``judge(text, fact) -> bool | None`` is
asked first, and ``None`` falls back to the matcher. The bench ships no judge and makes no
provider call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kggen.questions import Question

from kgbench.text import PAST, mentions, sentences_of

Judge = Callable[[str, dict], "bool | None"]


def _together(sentences: list[str], fact: dict) -> list[str]:
    return [
        s
        for s in sentences
        if mentions(s, fact["subject_names"]) and mentions(s, fact["object_names"])
    ]


@dataclass(frozen=True)
class OpenEndedScore:
    coverage: float  # weight of the required facts covered, out of the total weight
    covered: tuple[str, ...]  # fact ids
    missed: tuple[str, ...]
    forbidden: tuple[str, ...]  # fact ids of forbidden claims the answer made

    @property
    def clean(self) -> bool:
        return not self.forbidden

    def to_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 4),
            "covered": list(self.covered),
            "missed": list(self.missed),
            "forbidden": list(self.forbidden),
            "clean": self.clean,
        }


def score_open_ended(question: Question, answer: str, judge: Judge | None = None) -> OpenEndedScore:
    """Coverage of ``question.required_facts`` and the forbidden claims ``answer`` makes."""
    if question.answer_kind != "open_ended":
        raise ValueError(f"{question.id} is a {question.type} question, not an open-ended one")
    sentences = sentences_of(answer)
    covered, missed, total, got = [], [], 0.0, 0.0
    for fact in question.required_facts:
        verdict = judge(answer, fact) if judge else None
        hit = bool(_together(sentences, fact)) if verdict is None else verdict
        total += fact["weight"]
        if hit:
            got += fact["weight"]
            covered.append(fact["fact_id"])
        else:
            missed.append(fact["fact_id"])
    made = []
    for claim in question.forbidden_claims:
        verdict = judge(answer, claim) if judge else None
        if verdict is None:
            verdict = any(not PAST.search(s) for s in _together(sentences, claim))
        if verdict:
            made.append(claim["fact_id"])
    return OpenEndedScore(got / total if total else 0.0, tuple(covered), tuple(missed), tuple(made))
