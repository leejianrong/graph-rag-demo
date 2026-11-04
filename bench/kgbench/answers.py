"""Score a system's answer to a typed question (ADR-0013), one scorer per ``answer_kind``.

A system answers in text, and may add structure it knows: the entity ids it names, a number, a
refusal, the facts it cites. Structure is used when given; otherwise it is read from the text,
offline and deterministically, with the limits noted on each kind:

* ``entity``: right when the answer names the gold entity (any name or alias, whole words) and
  does not give a ``stale_values`` name as current (a name in a sentence with a past marker such
  as "formerly" is not a claim).
* ``set``: F1 of the listed entities against the gold set. From text, recall is the gold entities
  named, and precision is the share of listed items that name a gold entity.
* ``number``: the number in the answer equals the gold count (first number in the text).
* ``abstain`` and ``false_premise``: right when the answer declines or rejects the premise. A
  refusal is read from stock phrases ("cannot", "no information", "not true", ...).
* ``path``: right when the cited facts are exactly one gold chain. Free text cannot be scored, so
  an answer without cited facts scores zero.
* ``open_ended``: scored by :mod:`kgbench.openended`; right when every required fact is covered
  and no forbidden claim is made.

The report keeps answerable and unanswerable questions apart, with the correct-refusal and the
false-refusal rates side by side, so a system cannot look good by refusing everything.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from kggen.questions import Question

from kgbench.openended import score_open_ended
from kgbench.text import PAST, mentions, sentences_of

_DECLINE = re.compile(
    r"\b(cannot|can't|can not|unable to|do not know|don't know|no information|no record|"
    r"no evidence|no data|no mention|not mentioned|not stated|not found|not available|"
    r"not known|unknown|no such|there is no|there are no|not true|is false|incorrect|"
    r"false premise|premise is|was withdrawn|was retracted|withdrawn|retracted)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?![\w])")
_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_WORD_NUMBER = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)
_SPLIT = re.compile(r"[,;\n]|\band\b", re.IGNORECASE)


@dataclass(frozen=True)
class Answer:
    """What a system said. Only ``text`` is needed; the rest is read from it when absent."""

    text: str = ""
    entities: tuple[str, ...] | None = None  # entity ids the system names
    number: float | None = None
    refused: bool | None = None
    path: tuple[str, ...] | None = None  # fact ids the system cites


@dataclass(frozen=True)
class AnswerScore:
    question_id: str
    type: str
    answerable: bool  # a gold answer exists (the correct reply is not a refusal)
    correct: bool
    score: float  # 0 to 1: F1 for a set, coverage for open-ended, overlap for a path
    refused: bool  # the system declined
    detail: str = ""


def _refused(answer: Answer) -> bool:
    return answer.refused if answer.refused is not None else bool(_DECLINE.search(answer.text))


def _number(answer: Answer) -> float | None:
    if answer.number is not None:
        return answer.number
    found = [(m.start(), float(m.group(1).replace(",", ""))) for m in _NUMBER.finditer(answer.text)]
    found += [
        (m.start(), float(_WORDS[m.group(1).lower()])) for m in _WORD_NUMBER.finditer(answer.text)
    ]
    return min(found)[1] if found else None


def _gold_names(question: Question) -> tuple[tuple[str, ...], ...]:
    return question.answer_names or (
        (question.accepted_answers,) if question.accepted_answers else ()
    )


def _stale_claimed(question: Question, text: str) -> list[str]:
    return [
        name
        for name in question.stale_values
        if any(mentions(s, [name]) and not PAST.search(s) for s in sentences_of(text))
    ]


def _entity(question: Question, answer: Answer) -> tuple[bool, float, str]:
    if answer.entities is not None:
        hit = bool(set(answer.entities) & set(question.answers))
        return hit and len(set(answer.entities)) == 1, float(hit), ""
    names = [n for group in _gold_names(question) for n in group]
    named = mentions(answer.text, names)
    stale = _stale_claimed(question, answer.text)
    ok = named and not stale
    return ok, float(ok), f"gives the stale value {stale[0]}" if stale else ""


def _set(question: Question, answer: Answer) -> tuple[bool, float, str]:
    gold = set(question.answers)
    if answer.entities is not None:
        said = set(answer.entities)
        right = len(said & gold)
        precision, recall = right / len(said) if said else 0.0, right / len(gold) if gold else 0.0
    else:
        groups = _gold_names(question)
        recall = sum(mentions(answer.text, g) for g in groups) / len(groups) if groups else 0.0
        text = answer.text
        for i, name in enumerate(sorted({n for g in groups for n in g}, key=len, reverse=True)):
            text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", f"\x00{i}\x00", text, flags=re.I)
        items = [p for p in (x.strip() for x in _SPLIT.split(text)) if p]
        precision = sum("\x00" in p for p in items) / len(items) if items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1 == 1.0, f1, f"precision {precision:.2f}, recall {recall:.2f}"


def _path(question: Question, answer: Answer) -> tuple[bool, float, str]:
    if not answer.path:
        return False, 0.0, "no cited facts to check against the gold chains"
    cited = set(answer.path)
    best = max((len(cited & set(c)) / len(c) for c in question.gold_paths if c), default=0.0)
    exact = any(cited == set(c) for c in question.gold_paths)
    return exact, best, "" if exact else "the cited facts are not a shortest chain"


def score_answer(question: Question, answer: Answer | str) -> AnswerScore:
    """Score one answer (plain text is accepted) by the question's ``answer_kind``."""
    if isinstance(answer, str):
        answer = Answer(text=answer)
    kind = question.answer_kind
    refused = _refused(answer)
    answerable = kind not in ("abstain", "false_premise")
    if kind in ("abstain", "false_premise"):
        ok, score, detail = (
            refused,
            float(refused),
            "" if refused else "answered instead of declining",
        )
    elif kind == "number":
        got = _number(answer)
        ok = got is not None and question.number is not None and got == question.number
        score, detail = float(ok), f"said {got:g}" if got is not None else "no number given"
    elif kind == "set":
        ok, score, detail = _set(question, answer)
    elif kind == "path":
        ok, score, detail = _path(question, answer)
    elif kind == "open_ended":
        result = score_open_ended(question, answer.text)
        ok, score = result.coverage == 1.0 and result.clean, result.coverage
        detail = f"makes a stale claim: {result.forbidden[0]}" if result.forbidden else ""
    else:
        ok, score, detail = _entity(question, answer)
    if answerable and not ok and refused:
        detail = detail or "declined a question that has an answer"
    return AnswerScore(question.id, question.type, answerable, ok, score, refused, detail)


@dataclass(frozen=True)
class TypeScore:
    n: int
    correct: int
    mean_score: float


@dataclass(frozen=True)
class AnswerReport:
    by_type: dict[str, TypeScore]
    answerable: TypeScore  # questions with an answer: how many were answered right
    unanswerable: TypeScore  # questions to decline: how many were declined
    false_refusal_rate: float  # answerable questions the system declined
    scores: tuple[AnswerScore, ...]

    @property
    def correct_refusal_rate(self) -> float:
        return self.unanswerable.correct / self.unanswerable.n if self.unanswerable.n else 0.0

    def to_dict(self) -> dict:
        def one(t: TypeScore) -> dict:
            return {"n": t.n, "correct": t.correct, "mean_score": round(t.mean_score, 4)}

        return {
            "by_type": {k: one(v) for k, v in sorted(self.by_type.items())},
            "answerable": one(self.answerable),
            "unanswerable": one(self.unanswerable),
            "correct_refusal_rate": round(self.correct_refusal_rate, 4),
            "false_refusal_rate": round(self.false_refusal_rate, 4),
        }


def _summary(scores: Sequence[AnswerScore]) -> TypeScore:
    n = len(scores)
    return TypeScore(
        n, sum(s.correct for s in scores), sum(s.score for s in scores) / n if n else 0.0
    )


def evaluate_answers(
    questions: Iterable[Question], answers: Mapping[str, Answer | str]
) -> AnswerReport:
    """Score every question; one with no answer counts as answered wrongly, not declined."""
    scores = tuple(score_answer(q, answers.get(q.id, Answer())) for q in questions)
    grouped: dict[str, list[AnswerScore]] = defaultdict(list)
    for s in scores:
        grouped[s.type].append(s)
    can = [s for s in scores if s.answerable]
    cannot = [s for s in scores if not s.answerable]
    return AnswerReport(
        {t: _summary(g) for t, g in grouped.items()},
        _summary(can),
        _summary(cannot),
        sum(s.refused for s in can) / len(can) if can else 0.0,
        scores,
    )
