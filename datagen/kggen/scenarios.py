"""Update, delete and reorder scenarios over a world, with the expected final graph.

A memory store must handle facts that change, facts that are forgotten, and facts that arrive in any
order. A scenario starts from the world's current facts, adds one ``add`` event per fact, then
``update`` events (a fact's object changes, so the old fact is replaced by a new one) and ``delete``
events (a fact is forgotten). ``expected`` is the graph after every event, and ``orders`` are seeded
arrival orders of the events that keep each update or delete after the add it targets. Because an
update or delete touches a fact no other event touches, every order must end in the same graph:
``apply_events`` is the reference for that, and what a store is scored against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kggen.spec import PREDICATE_SIGNATURES
from kggen.world import Edge, World

Kind = Literal["add", "update", "delete"]
_TRIES = 30  # draws to find a new object before an update is turned into a delete


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    n_updates: int = Field(default=5, ge=0)
    n_deletes: int = Field(default=5, ge=0)
    n_orders: int = Field(default=3, ge=1, description="Arrival orders to generate.")


@dataclass(frozen=True)
class Event:
    kind: Kind
    fact: Edge  # add: the fact; update: the new fact; delete: the fact being removed
    replaces: str | None = None  # update: the id of the fact this one supersedes

    def to_dict(self) -> dict:
        return {"kind": self.kind, "fact": vars(self.fact), "replaces": self.replaces}


@dataclass(frozen=True)
class Scenario:
    events: tuple[Event, ...]  # canonical order: all adds, then updates, then deletes
    expected: tuple[Edge, ...]  # the graph once every event has been applied, ordered by id
    orders: tuple[tuple[int, ...], ...]  # permutations of event indices; each is causally valid

    def to_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self.events],
            "expected": [vars(e) for e in self.expected],
            "orders": [list(o) for o in self.orders],
        }


def apply_events(events: list[Event] | tuple[Event, ...]) -> tuple[Edge, ...]:
    """The reference semantics: replay events in the given order, return the facts left."""
    facts: dict[str, Edge] = {}
    for event in events:
        if event.kind == "add":
            facts[event.fact.id] = event.fact
        elif event.kind == "update":
            del facts[event.replaces]  # type: ignore[arg-type]
            facts[event.fact.id] = event.fact
        else:
            del facts[event.fact.id]
    return tuple(sorted(facts.values(), key=lambda e: e.id))


def generate_scenario(world: World, spec: ScenarioSpec | None = None) -> Scenario:
    spec = spec or ScenarioSpec()
    rng = random.Random(f"{world.spec.seed}:{spec.seed}:scenario")
    entities = {e.id: e for e in world.entities}
    real = [
        e
        for e in world.current_edges()
        if not entities[e.subject].distractor and not entities[e.object].distractor
    ]
    taken = {(e.subject, e.predicate, e.object) for e in world.edges}
    by_type: dict[str, list[str]] = {}
    for entity in world.entities:
        if not entity.distractor:
            by_type.setdefault(entity.type, []).append(entity.id)

    targets = rng.sample(real, min(len(real), spec.n_updates + spec.n_deletes))
    updates: list[Event] = []
    deletes: list[Event] = []
    for edge in targets:
        if len(updates) < spec.n_updates and (
            new := _retarget(edge, entities, by_type, taken, rng)
        ):
            taken.add((new.subject, new.predicate, new.object))
            updates.append(Event("update", new, replaces=edge.id))
        elif len(deletes) < spec.n_deletes:
            deletes.append(Event("delete", edge))
    adds = [Event("add", e) for e in world.current_edges()]
    events = tuple(adds + updates + deletes)

    add_index = {a.fact.id: i for i, a in enumerate(adds)}
    needs = {  # index of the add each update/delete must follow
        i: add_index[ev.replaces or ev.fact.id] for i, ev in enumerate(events) if ev.kind != "add"
    }
    orders = tuple(_causal_shuffle(len(events), needs, rng) for _ in range(spec.n_orders))
    return Scenario(events, apply_events(events), orders)


def _retarget(
    edge: Edge,
    entities: dict,
    by_type: dict[str, list[str]],
    taken: set[tuple[str, str, str]],
    rng: random.Random,
) -> Edge | None:
    """A copy of ``edge`` pointing at a different, type-valid object, or None if none exists."""
    subject_type = entities[edge.subject].type
    types = sorted(
        {d for s, d in PREDICATE_SIGNATURES.get(edge.predicate, ()) if s == subject_type}
    )
    pools = [by_type[t] for t in types if by_type.get(t)]
    if not pools:
        return None
    for _ in range(_TRIES):  # bounded random draws keep this linear in the number of updates
        o = rng.choice(rng.choice(pools))
        if o not in (edge.subject, edge.object) and (edge.subject, edge.predicate, o) not in taken:
            return Edge(f"{edge.id}~1", edge.subject, edge.predicate, o)
    return None


def _causal_shuffle(n: int, needs: dict[int, int], rng: random.Random) -> tuple[int, ...]:
    """A random order of ``range(n)`` in which every index follows the one it needs (linear)."""
    waiting: dict[int, list[int]] = {}
    for later, earlier in needs.items():
        waiting.setdefault(earlier, []).append(later)
    ready = [i for i in range(n) if i not in needs]
    order: list[int] = []
    while ready:
        j = rng.randrange(len(ready))
        ready[j], ready[-1] = ready[-1], ready[j]
        pick = ready.pop()
        order.append(pick)
        ready.extend(waiting.get(pick, ()))
    return tuple(order)
