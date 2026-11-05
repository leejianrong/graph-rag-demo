"""How confusable is the corpus? Alias, look-alike and distractor statistics (GRA-125).

Three things make retrieval and linking harder without changing a question: entities that are
named by an alias, entities whose names are almost the same, and documents that mention the
entities a question is about without holding its facts. This counts them, from the world and the
text, with no model:

* per world: entities with aliases, entities whose name another entity shares, and pairs of
  different names that are almost the same (``difflib`` ratio of 0.85 or more, same type);
* per text: mentions that use an alias, and documents flagged as distractors;
* per question, by hop count: documents that do *not* hold a supporting fact but name an entity
  on its path (by plain string search, as a retriever sees them), and how many of those are
  distractor documents.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from itertools import combinations

from kggen.render import Document

NEAR = 0.85


def world_stats(world: Mapping) -> dict[str, float]:
    entities = world["entities"]
    owners: dict[str, set[str]] = defaultdict(set)
    for e in entities:
        for label in (e["name"], *e["aliases"]):
            owners[label.casefold()].add(e["id"])
    shared = {i for ids in owners.values() if len(ids) > 1 for i in ids}
    names: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in entities:
        names[(e["type"], e["name"].casefold())].append(e["id"])
    distinct = sorted(names)
    near = sum(
        1
        for (ta, a), (tb, b) in combinations(distinct, 2)
        if ta == tb and SequenceMatcher(None, a, b).ratio() >= NEAR
    )
    n = len(entities)
    return {
        "entities": n,
        "aliased": sum(bool(e["aliases"]) for e in entities) / n,
        "shared_name": len(shared) / n,
        "near_duplicate_pairs": near,
    }


def text_stats(documents: Sequence[Document], world: Mapping) -> dict[str, float]:
    canonical = {e["id"]: e["name"] for e in world["entities"]}
    mentions = [m for d in documents for s in d.sentences for m in s.mentions]
    return {
        "documents": len(documents),
        "distractor_documents": sum(d.distractor for d in documents) / len(documents),
        "mentions": len(mentions),
        "alias_mentions": sum(m.surface != canonical[m.entity_id] for m in mentions)
        / len(mentions),
    }


def _patterns(world: Mapping) -> dict[str, re.Pattern[str]]:
    return {
        e["id"]: re.compile(
            r"\b(?:"
            + "|".join(map(re.escape, sorted({e["name"], *e["aliases"]}, key=len, reverse=True)))
            + r")\b",
            re.IGNORECASE,
        )
        for e in world["entities"]
    }


def question_stats(
    questions: Sequence[Mapping], world: Mapping, documents: Sequence[Document]
) -> dict[int, dict[str, float]]:
    """Per hop count: mean non-gold documents that name a path entity, and the mean and share
    of questions for which some of them are distractor documents."""
    patterns = _patterns(world)
    edges = {e["id"]: e for e in world["edges"]}
    holds = {d.id: {f for s in d.sentences for f in s.fact_ids} for d in documents}
    out: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for q in questions:
        gold = set(q["supporting_facts"])
        path = {x for f in gold for x in (edges[f]["subject"], edges[f]["object"])}
        confusing = [
            d
            for d in documents
            if not holds[d.id] & gold and any(patterns[i].search(d.text) for i in path)
        ]
        row = out[q["hops"]]
        row["n"] += 1
        row["confusing"] += len(confusing)
        row["distractors"] += sum(d.distractor for d in confusing)
        row["any_distractor"] += any(d.distractor for d in confusing)
    return {h: dict(r) for h, r in sorted(out.items())}


def format_report(
    world: Mapping[str, float],
    texts: Mapping[str, tuple[Mapping[str, float], Mapping[int, Mapping[str, float]]]],
) -> str:
    lines = [
        "| world | entities | with aliases | name shared with another entity | "
        "near-duplicate name pairs |",
        "|---|---|---|---|---|",
        f"| | {world['entities']:.0f} | {world['aliased']:.0%} | {world['shared_name']:.0%} | "
        f"{world['near_duplicate_pairs']:.0f} |",
        "",
        "| text | documents | distractor documents | mentions | mentions using an alias |",
        "|---|---|---|---|---|",
    ]
    for name, (stats, _) in texts.items():
        lines.append(
            f"| {name} | {stats['documents']:.0f} | {stats['distractor_documents']:.0%} | "
            f"{stats['mentions']:.0f} | {stats['alias_mentions']:.0%} |"
        )
    lines += [
        "",
        "Per question, documents that do not hold a supporting fact but name a path entity:",
        "",
        "| text | hops | questions | mean such documents | of which distractors (mean) | "
        "questions with a distractor among them |",
        "|---|---|---|---|---|---|",
    ]
    for name, (_, per_hop) in texts.items():
        for hops, r in per_hop.items():
            lines.append(
                f"| {name} | {hops} | {r['n']:.0f} | {r['confusing'] / r['n']:.1f} | "
                f"{r['distractors'] / r['n']:.2f} | {r['any_distractor'] / r['n']:.0%} |"
            )
    return "\n".join(lines)
