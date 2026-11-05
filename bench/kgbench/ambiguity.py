"""Does a reader who knows only names get one answer? (GRA-124)

The generator checks that a question's answer is unique among *entities*. A reader of the text
sees names, not entity ids, and two entities can share a name or an alias. So each chain
question is re-answered with names as the identity: whatever carries any name of an entity
on the path counts as that entity. The answers that come out beyond the gold's are answers the
text cannot rule out. Only chain questions (entity, set, count) have a path to re-walk;
compare and superlative questions are not analysed here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from kgbench.charts import wilson

CHAINS = ("entity", "set", "count")


def _classes(world: Mapping) -> dict[str, frozenset[str]]:
    """For each entity, every entity that shares a name or alias with it (itself included),
    followed transitively."""
    parent = {e["id"]: e["id"] for e in world["entities"]}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    holders: dict[str, str] = {}
    for e in world["entities"]:
        for label in (e["name"], *e["aliases"]):
            other = holders.setdefault(label.casefold(), e["id"])
            parent[find(e["id"])] = find(other)
    groups: dict[str, set[str]] = defaultdict(set)
    for i in parent:
        groups[find(i)].add(i)
    return {i: frozenset(groups[find(i)]) for i in parent}


def _steps(
    question: Mapping, edges: Mapping[str, Mapping], types: Mapping[str, str]
) -> list[tuple[str, bool, str]]:
    """The chain as (predicate, followed forward, type of the entity reached) along the first
    gold path. The question names each step's type ("the event that ..."), so a reader uses it."""
    here, steps = question["start"], []
    for fact in question["gold_paths"][0]:
        edge = edges[fact]
        forward = here == edge["subject"]
        here = edge["object"] if forward else edge["subject"]
        steps.append((edge["predicate"], forward, types[here]))
    return steps


def read_by_names(question: Mapping, world: Mapping, classes: Mapping) -> set[frozenset[str]]:
    """The answer classes (entities that share names count as one) a name-only reader reaches."""
    edges = {e["id"]: e for e in world["edges"]}
    types = {e["id"]: e["type"] for e in world["entities"]}
    live = [e for e in world["edges"] if e.get("valid_to") is None]
    frontier = set(classes[question["start"]])
    for predicate, forward, kind in _steps(question, edges, types):
        near, far = ("subject", "object") if forward else ("object", "subject")
        reached = {e[far] for e in live if e["predicate"] == predicate and e[near] in frontier}
        frontier = {m for i in reached for m in classes[i] if types[m] == kind}
    return {classes[i] for i in frontier}


def measure(questions: Sequence[Mapping], world: Mapping) -> dict[str, dict]:
    """Per question type: how many have answers beyond the gold's once names are the identity."""
    classes = _classes(world)
    out: dict[str, dict] = {}
    for q in questions:
        if q["type"] not in CHAINS or not q["gold_paths"]:
            continue
        gold = {classes[a] for a in q["answers"]}
        extra = read_by_names(q, world, classes) - gold
        row = out.setdefault(
            q["type"], {"n": 0, "ambiguous": 0, "extra": 0, "flagged": 0, "flagged_ambiguous": 0}
        )
        row["n"] += 1
        row["ambiguous"] += bool(extra)
        row["extra"] += len(extra)
        row["flagged"] += q["shared_name"]
        row["flagged_ambiguous"] += q["shared_name"] and bool(extra)
    return out


def format_table(result: Mapping[str, Mapping]) -> str:
    def cell(hit: int, total: int) -> str:
        low, high = wilson(hit, total)
        return f"{hit / total:.0%} ({hit}/{total}, {low:.0%}–{high:.0%})" if total else "n/a"

    lines = [
        "| type | n | answers beyond the gold (names as identity) | extra answers per question | "
        "shared_name set | of those, with extra answers |",
        "|---|---|---|---|---|---|",
    ]
    for kind, r in result.items():
        lines.append(
            f"| {kind} | {r['n']} | {cell(r['ambiguous'], r['n'])} | {r['extra'] / r['n']:.2f} | "
            f"{cell(r['flagged'], r['n'])} | {cell(r['flagged_ambiguous'], r['flagged'])} |"
        )
    return "\n".join(lines)
