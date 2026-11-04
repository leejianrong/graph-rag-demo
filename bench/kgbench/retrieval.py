"""Score retrieval on its own: supporting-fact recall@k and path recovery, per hop count (ADR-0011).

Retrieval is the claim, so it is scored apart from answer accuracy. A retriever (graph-rag, BM25,
dense) returns ranked sentences, each with the ids of the facts it expresses, so every system is
scored on the same gold: a question's ``supporting_facts``, the facts along its path.

* **recall@k**: the share of a question's gold facts expressed in the top ``k`` sentences.
* **path@k**: whether *all* of them are, i.e. the whole path can be read from the top ``k``.

Both are reported per hop count, which is where a graph should beat keyword or dense retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from kggen.questions import Question

DEFAULT_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class RetrievedSentence:
    doc: str
    text: str
    fact_ids: tuple[str, ...]  # the world-model facts this sentence expresses (may be empty)


class Retriever(Protocol):
    """What every system under comparison implements: question in, ranked sentences out."""

    def retrieve(self, question: str, k: int) -> Sequence[RetrievedSentence]: ...


@dataclass(frozen=True)
class HopScore:
    questions: int
    recall_at: dict[int, float]  # k -> mean fraction of gold facts in the top k sentences
    path_at: dict[int, float]  # k -> share of questions whose whole path is in the top k

    def to_dict(self) -> dict:
        return {
            "questions": self.questions,
            "recall_at": {k: round(v, 4) for k, v in self.recall_at.items()},
            "path_at": {k: round(v, 4) for k, v in self.path_at.items()},
        }


@dataclass(frozen=True)
class RetrievalReport:
    ks: tuple[int, ...]
    overall: HopScore
    by_hops: dict[int, HopScore]

    def to_dict(self) -> dict:
        return {
            "ks": list(self.ks),
            "overall": self.overall.to_dict(),
            "by_hops": {h: s.to_dict() for h, s in sorted(self.by_hops.items())},
        }

    def format_table(self) -> str:
        """One row per hop count; each ``k`` has a recall column and a whole-path column."""
        title = "Retrieval vs gold: recall@k and whole-path@k"
        head = "  hops  n    " + "  ".join(f"R@{k:<2} path@{k:<2}" for k in self.ks)
        lines = [title, "-" * len(title), head]
        rows = [("all", self.overall)] + [(str(h), s) for h, s in sorted(self.by_hops.items())]
        for name, s in rows:
            cells = "  ".join(f"{s.recall_at[k]:.2f}  {s.path_at[k]:.2f}  " for k in self.ks)
            lines.append(f"  {name:<5} {s.questions:<4} {cells}")
        return "\n".join(lines)


def facts_in_top(ranking: Sequence[RetrievedSentence], k: int) -> set[str]:
    """The fact ids expressed by the first ``k`` sentences."""
    return {f for sentence in ranking[:k] for f in sentence.fact_ids}


def score_retrieval(
    questions: Iterable[Question],
    rankings: Mapping[str, Sequence[RetrievedSentence]],
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalReport:
    """Score ``rankings`` (question id -> ranked sentences). A question with no ranking scores 0."""
    ks = tuple(sorted(set(ks)))
    per_hops: dict[int, list[Question]] = defaultdict(list)
    for q in questions:
        per_hops[q.hops].append(q)

    def summarize(group: list[Question]) -> HopScore:
        recall = {k: 0.0 for k in ks}
        path = {k: 0.0 for k in ks}
        for q in group:
            gold = set(q.supporting_facts)
            ranking = rankings.get(q.id, ())
            for k in ks:
                found = facts_in_top(ranking, k) & gold
                recall[k] += len(found) / len(gold) if gold else 0.0
                path[k] += float(bool(gold) and found == gold)
        n = len(group)
        return HopScore(
            n, {k: v / n for k, v in recall.items()}, {k: v / n for k, v in path.items()}
        )

    everything = [q for group in per_hops.values() for q in group]
    by_hops = {h: summarize(group) for h, group in per_hops.items()}
    return RetrievalReport(ks, summarize(everything) if everything else _empty(ks), by_hops)


def _empty(ks: Sequence[int]) -> HopScore:
    return HopScore(0, {k: 0.0 for k in ks}, {k: 0.0 for k in ks})


def evaluate(
    retriever: Retriever, questions: Sequence[Question], ks: Sequence[int] = DEFAULT_KS
) -> RetrievalReport:
    """Run ``retriever`` on every question (asking for the deepest ``k``) and score it."""
    depth = max(ks)
    return score_retrieval(
        questions, {q.id: list(retriever.retrieve(q.question, depth)) for q in questions}, ks
    )
