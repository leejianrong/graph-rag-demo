"""An optional LLM judge for open-ended answers, and its measured error rate (GRA-89).

``score_open_ended`` accepts a ``judge(text, fact) -> bool | None``. ``llm_judge`` builds one that
asks a chat model whether an answer states a required fact. Nothing here runs unless a caller
asks for it, and the judge is never trusted unmeasured: ``evaluate`` runs it over labelled cases
and reports its false positive and false negative rates next to the offline matcher's.

The cases are built from a world's own facts, so every label is true by construction, not
because a model said so. Each case is one answer that is meant to cover (or not cover) one
required fact:

* ``stated``: the fact in plain words (covered);
* ``pronoun``: the subject is named in one sentence and the fact is stated about "it" in the
  next (covered; the sentence matcher cannot see it);
* ``negated``: the fact is denied (not covered; the matcher is fooled by the names);
* ``co_mention``: both entities appear but no relation is stated (not covered; same trap);
* ``wrong_object``: the fact with another entity in the object's place (not covered);
* ``absent``: a different fact of the same question (not covered).

Stale claims (``forbidden_claims``) are not judged: the judge declines them and the matcher
decides, because "states X as currently true" needs its own measured prompt.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from kggen.llm import CachedChat

from kgbench.openended import Judge, covers

KINDS = ("stated", "pronoun", "negated", "co_mention", "wrong_object", "absent")
_SYSTEM = "You check a written answer against one fact. Reply with the single word YES or NO."
_ASK = (
    "Fact: {fact}\n\nAnswer:\n{answer}\n\n"
    "Does the answer state this fact, in the same words or a paraphrase? Reply NO if the fact is "
    "absent, denied, said to be untrue or unknown, or if the answer only mentions the names "
    "without stating the relation. Reply with YES or NO."
)


def _messages(answer: str, fact: dict) -> list[dict[str, str]]:
    prompt = _ASK.format(fact=fact["text"], answer=answer)
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]


def _verdict(reply: str) -> bool | None:
    word = reply.strip().upper().split(maxsplit=1)[0].strip(".,!") if reply.strip() else ""
    return {"YES": True, "NO": False}.get(word)


def llm_judge(chat: CachedChat) -> Judge:
    """A judge that asks ``chat``; ``None`` (use the matcher) for a stale claim or a reply that is
    neither YES nor NO. Only a valid verdict is stored in the cache."""

    def judge(answer: str, fact: dict) -> bool | None:
        if "reason" in fact:  # a forbidden claim: not measured, so not judged
            return None
        messages = _messages(answer, fact)
        reply = chat.ask(messages)
        verdict = _verdict(reply)
        if verdict is not None and chat.cached(messages) is None:
            chat.store(messages, reply)
        return verdict

    return judge


def build_cases(questions: Sequence, facts_per_kind: int = 12) -> list[dict]:
    """Labelled cases from the required facts of open-ended questions."""
    pool, seen = [], set()
    for q in questions:
        for fact in q.required_facts:
            subject = fact["subject_names"][0]
            if fact["fact_id"] not in seen and fact["text"].startswith(subject):
                seen.add(fact["fact_id"])
                pool.append(fact)
    cases = []
    for i, fact in enumerate(pool[:facts_per_kind]):
        other = pool[(i + 1) % len(pool)]
        subject, obj, text = fact["subject_names"][0], fact["object_names"][0], fact["text"]
        answers = {
            "stated": (text + ".", True),
            "pronoun": (f"{subject} appears in the records. It{text[len(subject) :]}.", True),
            "negated": (f"It is not true that {text}.", False),
            "co_mention": (
                f"{subject} and {obj} both appear in the records, but how they relate is unknown.",
                False,
            ),
            "wrong_object": (text.replace(obj, other["object_names"][0]) + ".", False),
            "absent": (other["text"] + ".", False),
        }
        for kind, (answer, label) in answers.items():
            if not label and kind in ("wrong_object", "absent") and covers(answer, fact):
                continue  # these are controls the names must not satisfy; a clash would mislabel
            keep = ("fact_id", "text", "subject_names", "object_names", "weight")
            cases.append(
                {
                    "id": f"{fact['fact_id']}-{kind}",
                    "kind": kind,
                    "fact": {k: fact[k] for k in keep},
                    "answer": answer,
                    "label": label,
                }
            )
    return cases


@dataclass(frozen=True)
class Rates:
    """Confusion counts for one scorer over labelled cases."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def false_negative_rate(self) -> float:
        return self.fn / (self.fn + self.tp) if self.fn + self.tp else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


def _rates(pairs: Sequence[tuple[bool, bool]]) -> Rates:
    c = Counter(pairs)  # (label, predicted)
    return Rates(c[True, True], c[False, True], c[True, False], c[False, False])


@dataclass(frozen=True)
class JudgeReport:
    n: int
    abstained: int  # judge gave no usable verdict, so the matcher decided
    matcher: Rates
    judge: Rates
    by_kind: dict[str, dict[str, tuple[int, int]]]  # kind -> scorer -> (correct, total)

    def to_dict(self) -> dict:
        return {
            "cases": self.n,
            "abstained": self.abstained,
            "matcher": self.matcher.to_dict(),
            "judge": self.judge.to_dict(),
            "by_kind": {k: {s: list(v) for s, v in d.items()} for k, d in self.by_kind.items()},
        }

    def format_table(self) -> str:
        rows = ["| scorer | false positive rate | false negative rate |", "|---|---|---|"]
        for name, r in (("offline matcher", self.matcher), ("LLM judge", self.judge)):
            rows.append(
                f"| {name} | {r.false_positive_rate:.0%} ({r.fp}/{r.fp + r.tn}) "
                f"| {r.false_negative_rate:.0%} ({r.fn}/{r.fn + r.tp}) |"
            )
        kinds = ["", "| case kind | label | matcher right | judge right |", "|---|---|---|---|"]
        for kind in KINDS:
            if kind in self.by_kind:
                m, j = self.by_kind[kind]["matcher"], self.by_kind[kind]["judge"]
                label = "covered" if kind in ("stated", "pronoun") else "not covered"
                kinds.append(f"| {kind} | {label} | {m[0]}/{m[1]} | {j[0]}/{j[1]} |")
        note = f"\n{self.n} cases; the judge gave no usable verdict on {self.abstained}."
        return "\n".join([*rows, *kinds]) + note


def evaluate(
    cases: Sequence[dict], judge: Callable[[str, dict], bool | None], workers: int = 8
) -> JudgeReport:
    """Score every case with the matcher and the judge. Calls run in parallel because a
    recording endpoint holds each one until it is answered."""
    with ThreadPoolExecutor(workers) as pool:
        verdicts = list(pool.map(lambda c: judge(c["answer"], c["fact"]), cases))
    matched = [covers(c["answer"], c["fact"]) for c in cases]
    judged = [m if v is None else v for m, v in zip(matched, verdicts, strict=True)]
    labels = [c["label"] for c in cases]
    by_kind: dict[str, dict[str, tuple[int, int]]] = {}
    for kind in KINDS:
        idx = [i for i, c in enumerate(cases) if c["kind"] == kind]
        if idx:
            by_kind[kind] = {
                name: (sum(got[i] == labels[i] for i in idx), len(idx))
                for name, got in (("matcher", matched), ("judge", judged))
            }
    return JudgeReport(
        n=len(cases),
        abstained=sum(v is None for v in verdicts),
        matcher=_rates(list(zip(labels, matched, strict=True))),
        judge=_rates(list(zip(labels, judged, strict=True))),
        by_kind=by_kind,
    )
