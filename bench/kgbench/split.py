"""Retrieval and extraction scores split by shared names (GRA-130).

A name is shared when two entities carry it (as a name or an alias, casefolded). Text that
mentions it cannot say which entity it means, so the shared names are where a retriever or an
extractor is most likely to go wrong. The generator already marks the questions whose path touches
such an entity (``shared_name``); this module applies the same test to facts and mentions and
reports each score for the two groups apart:

* retrieval: whole-path@k and recall@k per system, by the question's ``shared_name`` flag, with
  the hop count held fixed because flagged questions are not spread evenly over hop counts;
* extraction: mention recall and strict triple recall, by whether the entity (or either end of the
  fact) shares a name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from kggen.world import World

from kgbench.charts import wilson
from kgbench.extraction import Extraction, Gold, align
from kgbench.retrieval import Retriever, facts_in_top


def shared_ids(world: World) -> frozenset[str]:
    """Entities with a name or alias that another entity also carries."""
    owners: dict[str, set[str]] = defaultdict(set)
    for entity in world.entities:
        for label in (entity.name, *entity.aliases):
            owners[label.casefold()].add(entity.id)
    return frozenset(i for ids in owners.values() if len(ids) > 1 for i in ids)


def retrieval(
    questions: Sequence[Mapping], retrievers: Mapping[str, Retriever], k: int
) -> dict[str, dict[tuple[int, bool], dict[str, float]]]:
    """Per system and ``(hops, shared_name)``: ``n``, ``path`` and ``recall`` sums at ``k``."""
    out: dict = {name: defaultdict(lambda: defaultdict(float)) for name in retrievers}
    for q in questions:
        gold = set(q["supporting_facts"])
        for name, retriever in retrievers.items():
            found = facts_in_top(retriever.retrieve(q["question"], k), k) & gold
            cell = out[name][(q["hops"], q["shared_name"])]
            cell["n"] += 1
            cell["path"] += found == gold
            cell["recall"] += len(found) / len(gold)
    return {s: {key: dict(v) for key, v in sorted(cells.items())} for s, cells in out.items()}


def extraction(
    gold: Gold, extracted: Extraction, shared: frozenset[str]
) -> dict[str, dict[bool, tuple[int, int]]]:
    """``(found, total)`` for mentions and for strict triples, keyed by whether they are shared."""
    spans = {(m.doc, m.start, m.end) for m in extracted.mentions}
    aligned = align(gold, extracted)
    found = {
        m
        for m in ((aligned.get(s), p, aligned.get(o)) for s, p, o in set(extracted.triples))
        if m in gold.triples
    }
    out: dict = {"mentions": defaultdict(lambda: [0, 0]), "triples": defaultdict(lambda: [0, 0])}
    for span, entity in gold.mentions.items():
        cell = out["mentions"][entity in shared]
        cell[0] += span in spans
        cell[1] += 1
    for triple in gold.triples:
        cell = out["triples"][triple[0] in shared or triple[2] in shared]
        cell[0] += triple in found
        cell[1] += 1
    return {kind: {flag: (a, b) for flag, (a, b) in sorted(c.items())} for kind, c in out.items()}


def _cell(found: float, n: float) -> str:
    if not n:
        return "n/a"
    low, high = wilson(round(found), int(n))
    return f"{found / n:.0%} ({int(round(found))}/{int(n)}, {low:.0%}–{high:.0%})"


def format_retrieval(
    result: Mapping[str, Mapping[tuple[int, bool], Mapping[str, float]]], k: int
) -> str:
    """One row per system and hop count (and ``all``): path@k for questions without and with a
    shared name, then the gap in points. Intervals are 95% Wilson."""
    lines = [
        f"| system | hops | no shared name: path@{k} | shared name: path@{k} | gap (points) |",
        "|---|---|---|---|---|",
    ]
    for system, cells in result.items():
        hops = sorted({h for h, _ in cells})
        for label, wanted in (("all", None), *((str(h), h) for h in hops)):
            part = {
                flag: [c for (h, f), c in cells.items() if f == flag and wanted in (None, h)]
                for flag in (False, True)
            }
            n = {f: sum(c["n"] for c in cs) for f, cs in part.items()}
            hit = {f: sum(c["path"] for c in cs) for f, cs in part.items()}
            gap = (
                f"{(hit[True] / n[True] - hit[False] / n[False]) * 100:+.0f}"
                if n[True] and n[False]
                else "n/a"
            )
            lines.append(
                f"| {system} | {label} | {_cell(hit[False], n[False])} | "
                f"{_cell(hit[True], n[True])} | {gap} |"
            )
    return "\n".join(lines)


def format_extraction(result: Mapping[str, Mapping[str, Mapping[bool, tuple[int, int]]]]) -> str:
    """One row per pipeline: mention and triple recall, not shared against shared."""
    lines = [
        "| pipeline | mention recall, not shared | mention recall, shared | "
        "triple recall, no shared end | triple recall, a shared end |",
        "|---|---|---|---|---|",
    ]
    for name, kinds in result.items():
        cells = [
            _cell(*kinds[kind].get(flag, (0, 0)))
            for kind in ("mentions", "triples")
            for flag in (False, True)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
