"""Does a reader who sees only names get one answer? (GRA-183)

The generator makes a chain question's answer unique among *entities*. A reader of the text sees
names, not entity ids, and two entities can share a name or an alias. So the chain is re-answered
with names as the identity: whatever carries any name of an entity on the path counts as that
entity, and each step keeps the type the question names ("the event that ..."). An answer beyond
the gold's is one the text cannot rule out. The question's ``shared_name`` flag is only a warning
that a path touches a shared name; most flagged questions still have one answer, so this is the
test that says whether the gold is reliable for a grader that matches names.

Only chain questions (entity, set, count) have a path to re-walk. ``kgbench.ambiguity`` measures
the same thing over a finished dataset; this copy lets the generator record it, or avoid it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from kggen.world import World


class NameReader:
    """A reader who knows names but not entity ids, over one world's current facts."""

    def __init__(self, world: World) -> None:
        self._types = {e.id: e.type for e in world.entities}
        self._edges = {e.id: e for e in world.edges}
        self._live = world.current_edges()
        self._class = self._classes(world)

    @staticmethod
    def _classes(world: World) -> dict[str, frozenset[str]]:
        """For each entity, every entity that shares a name or alias with it, transitively."""
        parent = {e.id: e.id for e in world.entities}

        def find(i: str) -> str:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        holders: dict[str, str] = {}
        for e in world.entities:
            for label in (e.name, *e.aliases):
                other = holders.setdefault(label.casefold(), e.id)
                parent[find(e.id)] = find(other)
        groups: dict[str, set[str]] = defaultdict(set)
        for i in parent:
            groups[find(i)].add(i)
        return {i: frozenset(groups[find(i)]) for i in parent}

    def reached(self, start: str, facts: Sequence[str]) -> set[frozenset[str]]:
        """The answer classes a name-only reader reaches by following ``facts`` from ``start``."""
        here, steps = start, []
        for fact in facts:
            edge = self._edges[fact]
            forward = here == edge.subject
            here = edge.object if forward else edge.subject
            steps.append((edge.predicate, forward, self._types[here]))
        frontier = set(self._class[start])
        for predicate, forward, kind in steps:
            reach = {
                (e.object if forward else e.subject)
                for e in self._live
                if e.predicate == predicate and (e.subject if forward else e.object) in frontier
            }
            frontier = {m for i in reach for m in self._class[i] if self._types[m] == kind}
        return {self._class[i] for i in frontier}

    def ambiguous(self, start: str, facts: Sequence[str], answers: Sequence[str]) -> bool:
        """True if names give an answer beyond the gold ``answers`` along the chain ``facts``."""
        return bool(self.reached(start, facts) - {self._class[a] for a in answers})
