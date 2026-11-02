"""Generate a ground-truth world graph from a ``WorldSpec`` (ADR-0011).

The world is the benchmark's gold: documents, questions and scores are derived from it,
never from graph-rag's own extraction. Generation is deterministic: the same spec (which
includes the seed) always produces the same world, byte for byte once serialised.
"""

from __future__ import annotations

import itertools
import json
import random
from bisect import bisect_right
from dataclasses import dataclass

from kgbench.names import NameFactory
from kgbench.spec import WorldSpec


class WorldGenerationError(ValueError):
    """The spec cannot be realised (for example it asks for more edges than can be placed)."""


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str
    community: int


@dataclass(frozen=True)
class Edge:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class World:
    """A generated ground-truth graph plus the spec that produced it."""

    spec: WorldSpec
    entities: tuple[Entity, ...]
    edges: tuple[Edge, ...]

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.model_dump(),
            "entities": [vars(e) for e in self.entities],
            "edges": [vars(e) for e in self.edges],
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys and a trailing newline, so equal worlds are byte-equal."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=1) + "\n"


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` across keys in proportion to ``weights`` (largest remainder, exact sum)."""
    positive = {k: w for k, w in weights.items() if w > 0}
    scale = sum(positive.values())
    raw = {k: total * w / scale for k, w in positive.items()}
    counts = {k: int(v) for k, v in raw.items()}
    leftover = total - sum(counts.values())
    for key in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))[:leftover]:
        counts[key] += 1
    return counts


class _Pool:
    """Weighted sampling over a fixed list of entity indexes."""

    def __init__(self, members: list[int], weight: list[float]) -> None:
        self.members = members
        self._cum = list(itertools.accumulate(weight[i] for i in members))

    def sample(self, rng: random.Random) -> int:
        point = rng.random() * self._cum[-1]
        return self.members[min(bisect_right(self._cum, point), len(self.members) - 1)]


def generate(spec: WorldSpec) -> World:
    """Build the world described by ``spec``."""
    rng = random.Random(spec.seed)
    n = spec.n_entities

    counts = apportion(n, spec.type_mix)
    types = [t for t in sorted(counts) for _ in range(counts[t])]
    rng.shuffle(types)
    community = [i % spec.n_communities for i in range(n)]
    rng.shuffle(community)

    names = NameFactory(rng)
    width = len(str(n - 1))
    entities = tuple(
        Entity(
            id=f"E{i:0{width}d}", name=names.new(types[i]), type=types[i], community=community[i]
        )
        for i in range(n)
    )

    # Hub-ness: a random ranking turned into Zipf weights (skew 0 gives uniform weights).
    ranking = list(range(n))
    rng.shuffle(ranking)
    weight = [1.0] * n
    for rank, index in enumerate(ranking):
        weight[index] = (rank + 1) ** -spec.degree_skew

    by_type: dict[str, list[int]] = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    pools: dict[tuple[str, int | None], _Pool] = {}

    def pool(entity_type: str, comm: int | None) -> _Pool | None:
        key = (entity_type, comm)
        if key not in pools:
            members = [
                i for i in by_type.get(entity_type, []) if comm is None or community[i] == comm
            ]
            if not members:
                return None
            pools[key] = _Pool(members, weight)
        return pools[key]

    feasible = {
        p: tuple(pair for pair in pairs if pair[0] in by_type and pair[1] in by_type)
        for p, pairs in spec.feasible_pairs().items()
    }
    feasible = {p: pairs for p, pairs in feasible.items() if pairs}
    if not feasible:
        raise WorldGenerationError("no relation can connect the generated entity types")
    pred_names = sorted(feasible)
    pred_weights = [spec.predicate_mix[p] for p in pred_names]
    by_pair: dict[tuple[str, str], tuple[list[str], list[float]]] = {}
    for p in pred_names:
        for pair in feasible[p]:
            preds, ws = by_pair.setdefault(pair, ([], []))
            preds.append(p)
            ws.append(spec.predicate_mix[p])

    seen: set[tuple[int, str, int]] = set()
    edges: list[tuple[int, str, int]] = []
    out_edges: dict[int, list[tuple[str, int]]] = {}

    def add(s: int, p: str, o: int) -> bool:
        if s == o or (s, p, o) in seen:
            return False
        seen.add((s, p, o))
        edges.append((s, p, o))
        out_edges.setdefault(s, []).append((p, o))
        return True

    def draw_base_edge() -> bool:
        predicate = rng.choices(pred_names, weights=pred_weights)[0]
        s_type, o_type = rng.choice(feasible[predicate])
        subject = pool(s_type, None).sample(rng)  # type: ignore[union-attr]
        home = community[subject]
        target_pool = None
        if spec.n_communities > 1 and rng.random() < spec.community_mixing:
            global_pool = pool(o_type, None)
            for _ in range(20):
                candidate = global_pool.sample(rng)  # type: ignore[union-attr]
                if community[candidate] != home:
                    return add(subject, predicate, candidate)
        else:
            target_pool = pool(o_type, home)
        target_pool = target_pool or pool(o_type, None)
        return add(subject, predicate, target_pool.sample(rng))  # type: ignore[union-attr]

    def draw_closing_edge() -> bool:
        a, _, b = edges[rng.randrange(len(edges))]
        onward = out_edges.get(b)
        if not onward:
            return False
        _, c = onward[rng.randrange(len(onward))]
        options = by_pair.get((types[c], types[a]))
        if c == a or not options:
            return False
        predicate = rng.choices(options[0], weights=options[1])[0]
        return add(c, predicate, a)

    n_edges = max(1, round(n * spec.avg_degree / 2))
    n_closing = round(spec.cycle_rate * n_edges)
    budget = 200 * n_edges + 1000

    while len(edges) < n_edges - n_closing and budget > 0:
        draw_base_edge()
        budget -= 1
    while len(edges) < n_edges and n_closing and budget > 0:
        draw_closing_edge()
        budget -= 1
    while len(edges) < n_edges and budget > 0:  # closing edges that could not be placed
        draw_base_edge()
        budget -= 1
    if len(edges) < n_edges:
        raise WorldGenerationError(
            f"could only place {len(edges)} of {n_edges} edges; lower avg_degree or add entities"
        )

    ordered = sorted((entities[s].id, p, entities[o].id) for s, p, o in edges)
    return World(spec=spec, entities=entities, edges=tuple(Edge(*e) for e in ordered))
