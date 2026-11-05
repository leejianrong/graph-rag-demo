"""How much of a multi-hop score can be had without following the chain? (GRA-123)

MuSiQue is built so that a model answering from one hop alone does much worse than one that
chains. This module asks the same of our questions, with no LLM, three ways, per hop count:

* ``prior``: the expected accuracy of guessing, with no documents, among the entities that can
  fill the answer's slot of the final hop's relation (``1 / candidates``). It is the share a
  reader who knows only the schema gets;
* ``last fact``: a retriever's top-k holds the sentence stating the *final* fact. That sentence
  names the answer, so a reader needs nothing else to answer;
* ``answer text``: the answer's name or an alias appears in the top-k sentences.

Each is also counted when the whole path is *not* retrieved, which is the shortcut: the answer
is in hand without the chain. ``chance`` is the same measure for k random sentences (exact, not
sampled), so a number can be compared with what luck gives.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import comb

from kggen.render import Document

from kgbench.baselines import BM25Retriever, DenseRetriever, sentences_of
from kgbench.charts import grouped_bars, wilson
from kgbench.retrieval import Retriever, facts_in_top

SYSTEMS = ("BM25", "Dense")


def _final_fact(question: Mapping, edges: Mapping[str, Mapping]) -> Mapping | None:
    """The last fact of the first gold path, when the answer is one of its two ends."""
    edge = edges[question["gold_paths"][0][-1]]
    return edge if question["answer"] in (edge["subject"], edge["object"]) else None


def prior(question: Mapping, world: Mapping) -> float:
    """Expected accuracy of a uniform guess among entities that fill the answer's slot."""
    edges = {e["id"]: e for e in world["edges"]}
    last = _final_fact(question, edges)
    if last is None:
        return 0.0
    slot = "subject" if last["subject"] == question["answer"] else "object"
    pool = {e[slot] for e in world["edges"] if e["predicate"] == last["predicate"]}
    return 1 / len(pool)


def _chance(matching: int, total: int, k: int) -> float:
    """The probability that k sentences drawn at random include at least one of ``matching``."""
    return 1 - comb(total - matching, k) / comb(total, k) if total >= k else 1.0


def _names(question: Mapping) -> re.Pattern[str]:
    names = sorted(question["accepted_answers"], key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)


def measure(
    questions: Sequence[Mapping],
    world: Mapping,
    documents: Sequence[Document],
    retrievers: Mapping[str, Retriever],
    k: int = 5,
) -> dict[str, dict[int, dict[str, int | float]]]:
    """Counts per system and hop count: questions, ``path``, ``last``, ``answer``, the last two
    without the path (``last_only``, ``answer_only``), their ``chance`` expectations and the
    ``prior``."""
    edges = {e["id"]: e for e in world["edges"]}
    sentences = sentences_of(documents)
    out: dict = {s: defaultdict(lambda: defaultdict(float)) for s in retrievers}
    for q in questions:
        final = _final_fact(q, edges)
        if final is None:
            continue
        gold, pattern = set(q["supporting_facts"]), _names(q)
        in_text = sum(1 for s in sentences if pattern.search(s.text))
        with_fact = sum(1 for s in sentences if final["id"] in s.fact_ids)
        for name, retriever in retrievers.items():
            top = retriever.retrieve(q["question"], k)
            path = gold <= facts_in_top(top, k)
            last = final["id"] in facts_in_top(top, k)
            answer = any(pattern.search(s.text) for s in top)
            row = out[name][q["hops"]]
            row["n"] += 1
            row["path"] += path
            row["last"] += last
            row["last_only"] += last and not path
            row["answer"] += answer
            row["answer_only"] += answer and not path
            row["chance_last"] += _chance(with_fact, len(sentences), k)
            row["chance_answer"] += _chance(in_text, len(sentences), k)
            row["prior"] += prior(q, world)
    return {s: {h: dict(r) for h, r in sorted(rows.items())} for s, rows in out.items()}


def build_retrievers(documents: Sequence[Document], embedder) -> dict[str, Retriever]:
    return {"BM25": BM25Retriever(documents), "Dense": DenseRetriever(documents, embedder)}


def _pct(value: float, n: float) -> str:
    low, high = wilson(round(value), int(n))
    return f"{value / n:.0%} ({int(round(value))}/{int(n)}, {low:.0%}–{high:.0%})"


def format_table(result: Mapping[str, Mapping[int, Mapping[str, float]]], k: int) -> str:
    """One row per system and hop count; intervals are 95% Wilson."""
    lines = [
        f"| system | hops | n | prior | path@{k} | last fact@{k} | last fact, no path | "
        f"answer@{k} | answer, no path | chance (last fact / answer) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, rows in result.items():
        for hops, r in rows.items():
            n = r["n"]
            lines.append(
                f"| {system} | {hops} | {int(n)} | {r['prior'] / n:.1%} | {_pct(r['path'], n)} | "
                f"{_pct(r['last'], n)} | {_pct(r['last_only'], n)} | {_pct(r['answer'], n)} | "
                f"{_pct(r['answer_only'], n)} | "
                f"{r['chance_last'] / n:.1%} / {r['chance_answer'] / n:.1%} |"
            )
    return "\n".join(lines)


def chart_svg(
    result: Mapping[str, Mapping[int, Mapping[str, float]]], system: str, title: str
) -> str:
    """Whole path, last fact and answer text found in the top k, with the chance level."""
    rows = result[system]
    hops = list(rows)
    keys = (
        ("whole path", "path"),
        ("last fact", "last"),
        ("answer text", "answer"),
        ("chance: answer text", "chance_answer"),
    )
    series: dict[str, Sequence[float]] = {
        label: [rows[h][key] / rows[h]["n"] for h in hops] for label, key in keys
    }
    intervals: dict[str, Sequence[tuple[float, float]]] = {
        label: [
            (v, v) if key.startswith("chance") else wilson(round(rows[h][key]), int(rows[h]["n"]))
            for h, v in zip(hops, series[label], strict=True)
        ]
        for label, key in keys
    }
    return grouped_bars(
        title,
        [f"{h} hop" for h in hops],
        series,
        ylabel="share of questions",
        intervals=intervals,
        counts=[f"n={int(rows[h]['n'])}" for h in hops],
    )
