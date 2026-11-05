"""Filter generated questions with three LLM ratings, after the Hugging Face RAG cookbook (GRA-121).

Each question gets a 1-5 "total rating" on three criteria, each asked separately, rationale first:

* ``groundedness``: can the question be answered unambiguously from its context? The context is
  the documents holding the question's supporting facts, so this checks the text, not the graph;
* ``relevance``: would someone querying the knowledge base plausibly ask it?
* ``standalone``: does it make sense without further context?

The cookbook keeps a question when all three are at least 4. Nothing runs unless a caller asks
(an endpoint, or a committed cache for a replay). The ratings are one model's opinion, so the
report gives pass rates with intervals and leaves the dataset itself unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat

from kgbench.charts import grouped_bars, wilson

CRITERIA = ("groundedness", "relevance", "standalone")
KEEP_AT = 4

_FORMAT = (
    "Provide your answer as follows:\n\nAnswer:::\nEvaluation: (your rationale for the rating, "
    "as a text)\nTotal rating: (your rating, as a number between 1 and 5)\n\n"
    "You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer."
)
_ASKS = {
    "groundedness": (
        "You will be given a context and a question.\nYour task is to provide a 'total rating' "
        "scoring how well one can answer the given question unambiguously with the given "
        "context.\nGive your answer on a scale of 1 to 5, where 1 means that the question is "
        "not answerable at all given the context, and 5 means that the question is clearly and "
        "unambiguously answerable with the context.\n\n" + _FORMAT + "\n\nNow here are the "
        "question and context.\n\nQuestion: {question}\nContext: {context}\nAnswer:::"
    ),
    "relevance": (
        "You will be given a question.\nYour task is to provide a 'total rating' representing "
        "how useful this question can be to someone querying a knowledge base built from "
        "documents about companies, people, products and places.\nGive your answer on a scale "
        "of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the "
        "question is extremely useful.\n\n" + _FORMAT + "\n\nNow here is the question.\n\n"
        "Question: {question}\nAnswer:::"
    ),
    "standalone": (
        "You will be given a question.\nYour task is to provide a 'total rating' representing "
        "how context-independent this question is.\nGive your answer on a scale of 1 to 5, "
        "where 1 means that the question depends on additional information to be understood, "
        "and 5 means that the question makes sense by itself.\nFor instance, if the question "
        "refers to a particular setting, like 'in the context' or 'in the document', the "
        "rating must be 1. The questions can contain obscure technical nouns or acronyms and "
        "still be a 5: it must simply be clear to an operator with access to documentation "
        "what the question is about.\n\n" + _FORMAT + "\n\nNow here is the question.\n\n"
        "Question: {question}\nAnswer:::"
    ),
}
_SYSTEM = "You are a careful rater. Follow the requested answer format exactly."
_RATING = re.compile(r"Total rating:\s*\**\s*([1-5])\b")


def context_for(question: Mapping, documents: Sequence[Mapping]) -> str:
    """The documents that hold any of the question's supporting facts, in document order."""
    wanted = set(question["supporting_facts"])
    held = [d for d in documents if any(wanted & set(s["fact_ids"]) for s in d["sentences"])]
    return "\n\n".join(f"{d['title']}\n{d['text']}" for d in held)


def rating(chat: CachedChat, criterion: str, question: Mapping, context: str) -> int | None:
    """One 1-5 rating, or ``None`` when the reply has no valid one (never cached)."""
    prompt = _ASKS[criterion].format(question=question["question"], context=context)
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    reply = chat.ask(messages)
    found = _RATING.search(reply)
    if found and chat.cached(messages) is None:
        chat.store(messages, reply)
    return int(found.group(1)) if found else None


def critique(
    chat: CachedChat,
    questions: Sequence[Mapping],
    documents: Sequence[Mapping],
    workers: int = 4,
) -> list[dict]:
    """Rate every question on every criterion; a missing rating is ``None``."""

    def rate(q: Mapping) -> dict:
        context = context_for(q, documents)
        scores = {c: rating(chat, c, q, context) for c in CRITERIA}
        return {"id": q["id"], "hops": q["hops"], "type": q["type"], **scores}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(rate, questions))


def kept(row: Mapping) -> bool:
    return all((row[c] or 0) >= KEEP_AT for c in CRITERIA)


def text_ok(row: Mapping) -> bool:
    """Groundedness and standalone only: relevance asks about usefulness, not the text."""
    return all((row[c] or 0) >= KEEP_AT for c in ("groundedness", "standalone"))


def tag_shared_names(
    rows: Sequence[Mapping], questions: Sequence[Mapping], world: Mapping
) -> list[dict]:
    """Add ``shared_name``: does the question's path touch an entity whose name or alias another
    entity also carries? Text alone cannot tell such entities apart."""
    owners: dict[str, set[str]] = defaultdict(set)
    for e in world["entities"]:
        for name in (e["name"], *e["aliases"]):
            owners[name].add(e["id"])
    clash = {i for ids in owners.values() if len(ids) > 1 for i in ids}
    edges = {e["id"]: e for e in world["edges"]}
    by_id = {q["id"]: q for q in questions}

    def touches(q: Mapping) -> bool:
        ends = {
            end for f in q["supporting_facts"] for end in (edges[f]["subject"], edges[f]["object"])
        }
        return bool(ends & clash)

    return [{**r, "shared_name": "yes" if touches(by_id[r["id"]]) else "no"} for r in rows]


def _groups(rows: Sequence[Mapping], key: str) -> dict:
    groups: dict = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(sorted(groups.items()))


def _share(rows: Sequence[Mapping], ok) -> tuple[int, int]:
    return sum(1 for r in rows if ok(r)), len(rows)


def _cell(share: tuple[int, int]) -> str:
    low, high = wilson(*share)
    return f"{share[0] / share[1]:.0%} ({share[0]}/{share[1]}, {low:.0%}–{high:.0%})"


def format_table(rows: Sequence[Mapping]) -> str:
    """Pass rates, with 95% Wilson intervals, per hop count and per question type."""
    head = "| group | n | " + " | ".join(f"{c} ≥ {KEEP_AT}" for c in CRITERIA)
    head += " | groundedness and standalone | all three |"
    lines = [head, "|---|---|" + "---|" * (len(CRITERIA) + 2)]
    sections = [
        ("all", {"all": list(rows)}),
        ("hops", _groups(rows, "hops")),
        ("type", _groups(rows, "type")),
    ]
    if all("shared_name" in r for r in rows):
        sections.append(("shared name", _groups(rows, "shared_name")))
    for label, groups in sections:
        for name, part in groups.items():
            cells = [_cell(_share(part, lambda r, c=c: (r[c] or 0) >= KEEP_AT)) for c in CRITERIA]
            tag = name if label == "all" else f"{label} {name}"
            lines.append(
                f"| {tag} | {len(part)} | "
                + " | ".join(cells)
                + f" | {_cell(_share(part, text_ok))} | {_cell(_share(part, kept))} |"
            )
    missing = sum(1 for r in rows for c in CRITERIA if r[c] is None)
    if missing:
        lines.append(f"\n{missing} ratings had no valid number and count as failing.")
    return "\n".join(lines)


def chart_svg(rows: Sequence[Mapping], title: str) -> str:
    """Share of questions kept per hop count, for each criterion and for all three."""
    groups = _groups(rows, "hops")
    names = {c: lambda r, c=c: (r[c] or 0) >= KEEP_AT for c in CRITERIA} | {
        "groundedness and standalone": text_ok,
        "all three": kept,
    }
    shares = {n: [_share(g, ok) for g in groups.values()] for n, ok in names.items()}
    return grouped_bars(
        title,
        [f"{h} hop" for h in groups],
        {n: [a / b for a, b in s] for n, s in shares.items()},
        ylabel=f"share rated ≥ {KEEP_AT}",
        intervals={n: [wilson(a, b) for a, b in s] for n, s in shares.items()},
        counts=[f"n={len(g)}" for g in groups.values()],
    )
