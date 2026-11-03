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

from kggen.names import NameFactory, alias_forms
from kggen.spec import VAGUE_PREDICATE, WorldSpec


class WorldGenerationError(ValueError):
    """The spec cannot be realised (for example it asks for more edges than can be placed)."""


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    type: str
    community: int
    aliases: tuple[str, ...] = ()
    distractor: bool = False


@dataclass(frozen=True)
class Edge:
    """One fact. ``id`` is the stable world-model id that questions and documents cite."""

    id: str
    subject: str
    predicate: str
    object: str
    valid_from: int = 0
    valid_to: int | None = None  # None: still true in the last year


@dataclass(frozen=True)
class World:
    """A generated ground-truth graph plus the spec that produced it."""

    spec: WorldSpec
    entities: tuple[Entity, ...]
    edges: tuple[Edge, ...]

    def edges_as_of(self, year: int) -> tuple[Edge, ...]:
        """The facts true in ``year``: started by then and not yet ended."""
        return tuple(
            e
            for e in self.edges
            if e.valid_from <= year and (e.valid_to is None or year < e.valid_to)
        )

    def current_edges(self) -> tuple[Edge, ...]:
        """The expected final state: every fact that was never ended."""
        return tuple(e for e in self.edges if e.valid_to is None)

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
    n_distractors = round(spec.distractor_rate * n)
    width = len(str(n + n_distractors - 1))
    name_of = [names.new(types[i]) for i in range(n)]

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
    n_noise = min(round(spec.noise_rate * n_edges), n_edges - 1)
    n_core = n_edges - n_noise
    n_closing = round(spec.cycle_rate * n_core)
    budget = 200 * n_edges + 1000

    while len(edges) < n_core - n_closing and budget > 0:
        draw_base_edge()
        budget -= 1
    while len(edges) < n_core and n_closing and budget > 0:
        draw_closing_edge()
        budget -= 1
    while len(edges) < n_core and budget > 0:  # closing edges that could not be placed
        draw_base_edge()
        budget -= 1
    while len(edges) < n_edges and budget > 0:  # vague links between any two entities
        add(rng.randrange(n), VAGUE_PREDICATE, rng.randrange(n))
        budget -= 1
    if len(edges) < n_edges:
        raise WorldGenerationError(
            f"could only place {len(edges)} of {n_edges} edges; lower avg_degree or add entities"
        )

    dated = _churn(spec, edges, by_type, types)
    ids = [f"E{i:0{width}d}" for i in range(n + n_distractors)]
    entities = _entities(spec, ids, name_of, types, community, n_distractors)
    distractor_edges = _distractor_edges(spec, entities[n:])
    ordered = sorted(
        [(ids[s], p, ids[o], vf, vt) for s, p, o, vf, vt in dated] + distractor_edges,
        key=lambda e: (e[0], e[1], e[2], e[3]),
    )
    fact_width = len(str(len(ordered) - 1))
    facts = tuple(Edge(f"F{i:0{fact_width}d}", *e) for i, e in enumerate(ordered))
    return World(spec=spec, entities=entities, edges=facts)


def _churn(
    spec: WorldSpec,
    edges: list[tuple[int, str, int]],
    by_type: dict[str, list[int]],
    types: list[str],
) -> list[tuple[int, str, int, int, int | None]]:
    """Date every edge; some are replaced by a new object and some retracted."""
    rng = random.Random(f"{spec.seed}:churn")
    start = spec.start_year
    dated: list[tuple[int, str, int, int, int | None]] = [
        (s, p, o, start, None) for s, p, o in edges
    ]
    n_replace = round(spec.churn_rate * len(edges))
    n_retract = round(spec.retraction_rate * len(edges))
    if not n_replace + n_retract:
        return dated
    taken = {(s, p, o) for s, p, o in edges}
    order = list(range(len(edges)))
    rng.shuffle(order)
    replaced = retracted = 0
    for i in order:
        if replaced >= n_replace and retracted >= n_retract:
            break
        s, p, o, _, _ = dated[i]
        year = rng.randint(start + 1, spec.end_year)
        if replaced < n_replace:
            options = [c for c in by_type[types[o]] if c not in (s, o) and (s, p, c) not in taken]
            if options:
                new = rng.choice(options)
                taken.add((s, p, new))
                dated[i] = (s, p, o, start, year)
                dated.append((s, p, new, year, None))
                replaced += 1
                continue
        if retracted < n_retract:
            dated[i] = (s, p, o, start, year)
            retracted += 1
    if replaced < n_replace or retracted < n_retract:
        raise WorldGenerationError(
            "could not place all churn and retractions; lower churn_rate or retraction_rate"
        )
    return dated


def _entities(
    spec: WorldSpec,
    ids: list[str],
    name_of: list[str],
    types: list[str],
    community: list[int],
    n_distractors: int,
) -> tuple[Entity, ...]:
    """Build the entities, then apply homonyms, aliases and look-alike distractors."""
    rng = random.Random(f"{spec.seed}:ambiguity")
    n = len(types)
    names = list(name_of)

    # Homonyms: some entities take over another entity's exact name.
    homonyms = rng.sample(range(n), round(spec.homonym_rate * n))
    taken = set(homonyms)
    twins = [i for i in range(n) if i not in taken]
    for i in homonyms:
        same = [t for t in twins if types[t] == types[i]]
        other = [t for t in twins if types[t] != types[i]]
        pool = same if (rng.random() < spec.homonym_same_type and same) or not other else other
        names[i] = names[rng.choice(pool)]

    # Aliases: a share of entities get alternative names that no entity already uses.
    used_names = set(names)
    aliases: dict[int, tuple[str, ...]] = {}
    wanted = round(spec.alias_rate * n)
    for i in rng.sample(range(n), n):
        if len(aliases) >= wanted:
            break
        forms = [a for a in alias_forms(names[i], types[i]) if a not in used_names]
        if forms:
            aliases[i] = tuple(forms[: rng.randint(1, spec.aliases_per_entity)])

    real = [Entity(ids[i], names[i], types[i], community[i], aliases.get(i, ())) for i in range(n)]

    # Distractors: new entities whose names resemble real ones.
    factory = NameFactory(rng, used_names)
    out = list(real)
    for k in range(n_distractors):
        model = real[rng.randrange(n)]
        name = factory.look_alike(model.name, model.type)
        out.append(Entity(ids[n + k], name, model.type, model.community, (), True))
    return tuple(out)


def _distractor_edges(
    spec: WorldSpec, distractors: tuple[Entity, ...]
) -> list[tuple[str, str, str, int, int | None]]:
    """One edge from each distractor to another, so distractor documents have facts."""
    rng = random.Random(f"{spec.seed}:distractor-edges")
    feasible = spec.feasible_pairs()
    out: list[tuple[str, str, str, int, int | None]] = []
    seen: set[tuple[str, str, str]] = set()
    for d in distractors:
        options = [
            (p, o)
            for o in distractors
            if o.id != d.id
            for p, pairs in feasible.items()
            if (d.type, o.type) in pairs
        ]
        if not options:
            continue
        p, o = rng.choice(options)
        if (d.id, p, o.id) not in seen:
            seen.add((d.id, p, o.id))
            out.append((d.id, p, o.id, spec.start_year, None))
    return out
