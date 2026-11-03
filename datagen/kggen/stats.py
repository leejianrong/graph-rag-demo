"""Realized statistics of a generated world, and a check of each knob against its spec.

A complexity knob nobody verifies is a claim, not a control. ``world_stats`` measures what
the generator actually produced; ``check_against_spec`` lines each knob's target up with the
realized value and says whether it is within tolerance.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

from kggen.spec import VAGUE_PREDICATE
from kggen.world import World

PATH_SOURCES = 50  # BFS sources sampled for the path-length histogram
# Below this many entities per community the generator often has no same-community target of
# the right type, so community_mixing is not a controllable knob and is not checked.
MIN_COMMUNITY_SIZE = 25


@dataclass(frozen=True)
class Check:
    """One knob: what the spec asked for, what the world has, and whether they agree."""

    knob: str
    target: float
    realized: float
    tolerance: float
    ok: bool


def _base_edges(world: World):
    """Facts true at the start year among real entities (before churn, without distractors)."""
    distractors = {e.id for e in world.entities if e.distractor}
    return [
        e
        for e in world.edges
        if e.valid_from == world.spec.start_year
        and e.subject not in distractors
        and e.object not in distractors
    ]


def _adjacency(ids: list[str], edges) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        adj[e.subject].append(e.object)
        adj[e.object].append(e.subject)
    return adj


def _components(adj: dict[str, list[str]]) -> list[int]:
    seen: set[str] = set()
    sizes: list[int] = []
    for start in adj:
        if start in seen:
            continue
        seen.add(start)
        queue, size = deque([start]), 0
        while queue:
            node = queue.popleft()
            size += 1
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _path_histogram(adj: dict[str, list[str]], seed: int) -> dict[int, int]:
    """Shortest-path lengths from a seeded sample of sources, over reachable pairs."""
    rng = random.Random(f"{seed}:paths")
    nodes = sorted(adj)
    sources = rng.sample(nodes, min(PATH_SOURCES, len(nodes)))
    histogram: Counter[int] = Counter()
    for source in sources:
        dist = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for nxt in adj[node]:
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        histogram.update(d for d in dist.values() if d > 0)
    return dict(sorted(histogram.items()))


def world_stats(world: World) -> dict:
    """Measure the generated world. Structure is measured on the original (start-year) graph."""
    real = [e for e in world.entities if not e.distractor]
    base = _base_edges(world)
    adj = _adjacency([e.id for e in real], base)
    degree = Counter({i: len(v) for i, v in adj.items()})
    successors = sum(e.valid_from > world.spec.start_year for e in world.edges)
    ended = sum(e.valid_to is not None for e in world.edges)
    components = _components(adj)
    names = Counter(e.name for e in real)
    return {
        "entities": len(real),
        "distractors": len(world.entities) - len(real),
        "facts": len(world.edges),
        "original_facts": len(base),
        "current_facts": len(world.current_edges()),
        "type_counts": dict(sorted(Counter(e.type for e in real).items())),
        "degree": {
            "mean": 2 * len(base) / len(real),
            "max": max(degree.values(), default=0),
            "histogram": dict(sorted(Counter(degree.values()).items())),
        },
        "components": {
            "count": len(components),
            "largest": components[0],
            "sizes": components[:10],
        },
        "communities": dict(sorted(Counter(e.community for e in real).items())),
        "path_lengths": _path_histogram(adj, world.spec.seed),
        "ambiguity": {
            "aliased_entities": sum(bool(e.aliases) for e in real),
            "aliases": sum(len(e.aliases) for e in real),
            "shared_names": len(real) - len(names),
            "vague_facts": sum(e.predicate == VAGUE_PREDICATE for e in base),
        },
        "churn": {"replaced": successors, "retracted": ended - successors},
    }


def check_against_spec(world: World, stats: dict | None = None) -> list[Check]:
    """Each knob's target next to its realized value, with a tolerance and a pass mark."""
    spec = world.spec
    stats = stats or world_stats(world)
    n = spec.n_entities
    base = _base_edges(world)
    n_base = len(base)
    checks: list[Check] = []

    def add(knob: str, target: float, realized: float, tolerance: float) -> None:
        checks.append(Check(knob, target, realized, tolerance, abs(realized - target) <= tolerance))

    add("n_entities", n, stats["entities"], 0)
    add("avg_degree", spec.avg_degree, stats["degree"]["mean"], 0.05 * spec.avg_degree + 2 / n)
    total_weight = sum(spec.type_mix.values())
    worst = max(
        abs(stats["type_counts"].get(t, 0) / n - w / total_weight) for t, w in spec.type_mix.items()
    )
    add("type_mix", 0.0, worst, 1 / n + 1e-9)
    if spec.n_communities > 1 and n / spec.n_communities >= MIN_COMMUNITY_SIZE:
        community = {e.id: e.community for e in world.entities}
        typed = [e for e in base if e.predicate != VAGUE_PREDICATE]
        crossing = sum(community[e.subject] != community[e.object] for e in typed) / len(typed)
        p = spec.community_mixing
        # Binomial noise on the edge count, plus a little for cycle-closing edges.
        add("community_mixing", p, crossing, 0.05 + 3 * (p * (1 - p) / len(typed)) ** 0.5)
    add("alias_rate", spec.alias_rate, stats["ambiguity"]["aliased_entities"] / n, 1 / n)
    add("homonym_rate", spec.homonym_rate, stats["ambiguity"]["shared_names"] / n, 1 / n)
    add("distractor_rate", spec.distractor_rate, stats["distractors"] / n, 1 / n)
    add("noise_rate", spec.noise_rate, stats["ambiguity"]["vague_facts"] / n_base, 1 / n_base)
    add("churn_rate", spec.churn_rate, stats["churn"]["replaced"] / n_base, 1 / n_base)
    add("retraction_rate", spec.retraction_rate, stats["churn"]["retracted"] / n_base, 1 / n_base)
    return checks


def report(world: World) -> dict:
    """The full report written beside a world: measurements plus the spec checks."""
    stats = world_stats(world)
    return {
        "stats": stats,
        "checks": [vars(c) for c in check_against_spec(world, stats)],
    }
