"""The Hugging Face cookbook's 1-5 correctness judge, next to our yes/no one (GRA-126).

The cookbook grades an answer against a reference on a 1-5 scale (feedback first, then
``[RESULT] n``), and reports ``(score - 1) / 4``. Here the reference is a required fact and the
answer is one of the labelled judge cases, whose labels are true by construction. Two questions:
does a 1-5 score separate covering from non-covering answers at least as well as yes/no, and where
do the two judges disagree? Opt-in; nothing runs without an endpoint or a committed cache.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat

from kgbench.charts import grouped_bars, wilson

_SYSTEM = "You are a fair evaluator language model."
_ASK = (
    "###Task Description:\nAn instruction (might include an Input inside it), a response to "
    "evaluate, a reference answer that gets a score of 5, and a score rubric representing a "
    "evaluation criteria are given.\n"
    "1. Write a detailed feedback that assess the quality of the response strictly based on the "
    "given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 and 5. You should "
    "refer to the score rubric.\n"
    '3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} '
    '[RESULT] {{an integer number between 1 and 5}}"\n'
    "4. Please do not generate any other opening, closing, and explanations. Be sure to include "
    "[RESULT] in your output.\n\n"
    "###The instruction to evaluate:\nState the fact given in the reference answer.\n\n"
    "###Response to evaluate:\n{answer}\n\n"
    "###Reference Answer (Score 5):\n{fact}\n\n"
    "###Score Rubrics:\n[Is the response correct, accurate, and factual based on the reference "
    "answer?]\n"
    "Score 1: The response is completely incorrect, inaccurate, and/or not factual.\n"
    "Score 2: The response is mostly incorrect, inaccurate, and/or not factual.\n"
    "Score 3: The response is somewhat correct, accurate, and/or factual.\n"
    "Score 4: The response is mostly correct, accurate, and factual.\n"
    "Score 5: The response is completely correct, accurate, and factual.\n\n"
    "###Feedback:"
)
_RESULT = re.compile(r"\[RESULT\]\s*([1-5])\b")
KEEP_AT = 4


def score(chat: CachedChat, answer: str, fact: Mapping) -> int | None:
    """A 1-5 score, or ``None`` when the reply has none (never cached)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _ASK.format(answer=answer, fact=fact["text"])},
    ]
    reply = chat.ask(messages)
    found = _RESULT.search(reply)
    if found and chat.cached(messages) is None:
        chat.store(messages, reply)
    return int(found.group(1)) if found else None


def compare(
    cases: Sequence[Mapping],
    chat: CachedChat,
    binary: Callable[[str, dict], bool | None],
    workers: int = 4,
) -> list[dict]:
    """Per case: the label, the yes/no verdict (``None`` if it abstained) and the 1-5 score."""

    def row(case: Mapping) -> dict:
        return {
            "id": case["id"],
            "kind": case["kind"],
            "label": case["label"],
            "binary": binary(case["answer"], dict(case["fact"])),
            "score": score(chat, case["answer"], case["fact"]),
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(row, cases))


def _says_covered(row: Mapping) -> bool:
    return (row["score"] or 0) >= KEEP_AT


def format_table(rows: Sequence[Mapping]) -> str:
    """Accuracy of each judge against the labels, by case kind, and where they disagree."""

    def cell(hit: int, total: int) -> str:
        low, high = wilson(hit, total)
        return f"{hit / total:.0%} ({hit}/{total}, {low:.0%}–{high:.0%})"

    lines = [
        f"| kind | label | n | yes/no judge right | 1-5 judge right (score ≥ {KEEP_AT}) | "
        "mean (score − 1) / 4 | judges disagree |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, bool], list[Mapping]] = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["label"])].append(r)
    groups[("all", True)] = [r for r in rows if r["label"]]
    groups[("all", False)] = [r for r in rows if not r["label"]]
    for (kind, label), part in sorted(groups.items(), key=lambda kv: (kv[0][0] == "all", kv[0])):
        n = len(part)
        scored = [r for r in part if r["score"]]
        mean = sum((r["score"] - 1) / 4 for r in scored) / len(scored) if scored else 0.0
        lines.append(
            f"| {kind} | {'covers' if label else 'does not'} | {n} | "
            f"{cell(sum(r['binary'] == label for r in part), n)} | "
            f"{cell(sum(_says_covered(r) == label for r in part), n)} | {mean:.2f} | "
            f"{sum(r['binary'] != _says_covered(r) for r in part)} |"
        )
    return "\n".join(lines)


def distribution(rows: Sequence[Mapping]) -> str:
    """How many answers got each score, for answers that do and do not cover the fact."""
    lines = [
        "| label | " + " | ".join(f"score {s}" for s in range(1, 6)) + " |",
        "|---|" + "---|" * 5,
    ]
    for label in (True, False):
        part = [r for r in rows if r["label"] == label]
        counts = [sum(r["score"] == s for r in part) for s in range(1, 6)]
        lines.append(
            f"| {'covers' if label else 'does not cover'} (n={len(part)}) | "
            + " | ".join(map(str, counts))
            + " |"
        )
    return "\n".join(lines)


def chart_svg(rows: Sequence[Mapping], title: str) -> str:
    """Accuracy of both judges per label, with 95% intervals."""
    cats, yes_no, scaled, ivs_a, ivs_b, counts = [], [], [], [], [], []
    for label, name in ((True, "covers"), (False, "does not cover")):
        part = [r for r in rows if r["label"] == label]
        a = sum(r["binary"] == label for r in part)
        b = sum(_says_covered(r) == label for r in part)
        cats.append(name)
        yes_no.append(a / len(part))
        scaled.append(b / len(part))
        ivs_a.append(wilson(a, len(part)))
        ivs_b.append(wilson(b, len(part)))
        counts.append(f"n={len(part)}")
    return grouped_bars(
        title,
        cats,
        {"yes/no judge": yes_no, f"1-5 judge, ≥{KEEP_AT}": scaled},
        ylabel="share judged correctly",
        intervals={"yes/no judge": ivs_a, f"1-5 judge, ≥{KEEP_AT}": ivs_b},
        counts=counts,
    )
