"""Can a reader answer from one hop? The MuSiQue test with an LLM reader (GRA-131).

MuSiQue keeps a question only if a model reading a single hop cannot answer it. GRA-123 bounded
the shortcut with retrieval alone; here a reader answers each question from two passages and the
answers are graded:

* ``last``: only the sentence(s) stating the *final* fact of the gold path. It names the answer
  but not how the question's start reaches it, so a correct answer here needed no chain;
* ``chain``: the sentences of every fact on the path, which is what a perfect retriever would hand
  over, so it bounds what any pipeline can get from this reader.

The reader may say ``unknown``. A correct answer from ``last`` on a question of two or more hops is
a shortcut. Opt-in: nothing runs without an endpoint or a committed cache.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat
from kggen.render import Document

from kgbench.charts import grouped_bars, wilson

INPUTS = ("last", "chain")
OUTCOMES = ("correct", "unknown", "wrong")
_SYSTEM = "You answer questions from a passage. Reply with only the answer."
_ASK = (
    "Answer the question using only the passage. Reply with just the name, or with the single "
    "word unknown if the passage does not tell you the answer.\n\nPassage:\n{passage}\n\n"
    "Question: {question}\nAnswer:"
)


def passages(question: Mapping, documents: Sequence[Document]) -> dict[str, tuple[str, bool]]:
    """Per input, the passage and whether it already holds every fact on the path."""
    by_fact: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        for sentence in doc.sentences:
            for fact in sentence.fact_ids:
                by_fact[fact].append(sentence.text)
    wanted = {"last": [question["gold_paths"][0][-1]], "chain": question["supporting_facts"]}
    out = {}
    for name, facts in wanted.items():
        texts = list(dict.fromkeys(t for f in facts for t in by_fact[f]))
        expressed = {
            f for doc in documents for s in doc.sentences if s.text in texts for f in s.fact_ids
        }
        out[name] = (" ".join(texts), set(question["supporting_facts"]) <= expressed)
    return out


def grade(reply: str, question: Mapping) -> str:
    """``correct`` if the reply names an accepted answer, ``unknown`` if it declines."""
    for name in question["accepted_answers"]:
        if re.search(rf"\b{re.escape(name)}\b", reply, re.IGNORECASE):
            return "correct"
    return "unknown" if reply.strip().lower().startswith("unknown") else "wrong"


def ask(chat: CachedChat, question: Mapping, passage: str) -> str | None:
    """One reader reply; ``None`` when it is empty (never cached)."""
    prompt = _ASK.format(passage=passage, question=question["question"])
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    reply = chat.ask(messages)
    if reply.strip() and chat.cached(messages) is None:
        chat.store(messages, reply)
    return reply if reply.strip() else None


def evaluate(
    chat: CachedChat,
    questions: Sequence[Mapping],
    documents: Sequence[Document],
    workers: int = 4,
) -> list[dict]:
    """One row per question and input: ``outcome`` (or ``None`` for an empty reply) and whether
    the passage already held the whole path."""

    def run(q: Mapping) -> list[dict]:
        rows = []
        for name, (passage, whole) in passages(q, documents).items():
            reply = ask(chat, q, passage)
            rows.append(
                {
                    "id": q["id"],
                    "hops": q["hops"],
                    "shared_name": q["shared_name"],
                    "input": name,
                    "whole_path": whole,
                    "reply": reply,
                    "outcome": grade(reply, q) if reply else None,
                }
            )
        return rows

    with ThreadPoolExecutor(workers) as pool:
        return [row for rows in pool.map(run, questions) for row in rows]


def _cell(rows: Sequence[Mapping], outcome: str) -> str:
    hit = sum(1 for r in rows if r["outcome"] == outcome)
    low, high = wilson(hit, len(rows))
    return f"{hit / len(rows):.0%} ({hit}/{len(rows)}, {low:.0%}–{high:.0%})" if rows else "n/a"


def format_table(rows: Sequence[Mapping]) -> str:
    """Per hop count, and per ``shared_name`` flag: the share of ``correct``, ``unknown`` and
    ``wrong`` replies for each input (intervals are 95% Wilson). An empty reply counts as none."""
    lines = [
        "| group | n | last: correct | last: unknown | last: wrong | chain: correct | "
        "chain: unknown | chain: wrong |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups = {"all": list(rows)}
    for key, label in (("hops", "hops"), ("shared_name", "shared name")):
        for value in sorted({r[key] for r in rows}):
            groups[f"{label} {value}"] = [r for r in rows if r[key] == value]
    for name, part in groups.items():
        cells = []
        for inp in INPUTS:
            mine = [r for r in part if r["input"] == inp]
            cells += [_cell(mine, o) for o in OUTCOMES]
        lines.append(f"| {name} | {len(part) // len(INPUTS)} | " + " | ".join(cells) + " |")
    empty = sum(1 for r in rows if r["outcome"] is None)
    if empty:
        lines.append(f"\n{empty} replies were empty and count in no column.")
    return "\n".join(lines)


def shortcuts(rows: Sequence[Mapping]) -> str:
    """Multi-hop questions answered right from the last fact alone, and how many of those had a
    last passage that held the whole path anyway (merged sentences), so they are not shortcuts."""
    last = [r for r in rows if r["input"] == "last" and r["hops"] > 1]
    right = [r for r in last if r["outcome"] == "correct"]
    whole = sum(1 for r in right if r["whole_path"])
    low, high = wilson(len(right) - whole, len(last))
    return (
        f"Of {len(last)} questions of two or more hops, {len(right)} were answered correctly from "
        f"the last fact alone; {whole} of those passages already held the whole path, leaving "
        f"{len(right) - whole} true shortcuts ({(len(right) - whole) / len(last):.1%}, "
        f"{low:.0%}–{high:.0%})."
        if last
        else "No multi-hop questions."
    )


def chart_svg(rows: Sequence[Mapping], title: str) -> str:
    """Correct answers per hop count from the last fact and from the whole path."""
    hops = sorted({r["hops"] for r in rows})
    series: dict[str, Sequence[float]] = {}
    intervals: dict[str, Sequence[tuple[float, float]]] = {}
    for inp in INPUTS:
        parts = [[r for r in rows if r["input"] == inp and r["hops"] == h] for h in hops]
        hits = [sum(1 for r in p if r["outcome"] == "correct") for p in parts]
        series[f"{inp}"] = [h / len(p) for h, p in zip(hits, parts, strict=True)]
        intervals[f"{inp}"] = [wilson(h, len(p)) for h, p in zip(hits, parts, strict=True)]
    return grouped_bars(
        title,
        [f"{h} hop" for h in hops],
        series,
        ylabel="share answered correctly",
        intervals=intervals,
        counts=[
            f"n={sum(1 for r in rows if r['hops'] == h and r['input'] == 'last')}" for h in hops
        ],
    )
