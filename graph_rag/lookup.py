"""Look an entity up by name or id: its aliases and the facts it takes part in (GRA-37, ADR-0014).

Built over the ports that already exist: name blocking in the entity store finds candidates, and
the graph store lists each one's edges. Nothing here reads the vector, so an answer is plain data.
"""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel

from graph_rag.models import CanonicalEntity, CuratedType, EdgeProvenance
from graph_rag.normalize import normalize_name
from graph_rag.ports import EntityStore, GraphStore
from graph_rag.scope import Scope

__all__ = ["EntityDetail", "EntityLookup", "EntityRef", "Neighbour"]


class EntityRef(BaseModel):
    """An entity as an API reader sees it: no vector."""

    canonical_id: str
    name: str
    type: CuratedType
    aliases: list[str] = []

    @classmethod
    def of(cls, entity: CanonicalEntity) -> EntityRef:
        return cls(
            canonical_id=entity.canonical_id,
            name=entity.name,
            type=entity.type,
            aliases=entity.aliases,
        )


class Neighbour(BaseModel):
    """One fact the entity takes part in, with the entity on its other side."""

    entity: EntityRef
    predicate: str
    direction: str  # "outgoing": the looked-up entity is the subject
    valid_from: str | None = None
    valid_to: str | None = None  # set when a newer fact ended this one (GRA-43)
    provenance: EdgeProvenance


class EntityDetail(EntityRef):
    """An entity with its neighbours."""

    neighbours: list[Neighbour] = []


class EntityLookup:
    """Finds entities in one scope and describes them."""

    def __init__(self, entity_store: EntityStore, graph_store: GraphStore) -> None:
        self._entities = entity_store
        self._graph = graph_store

    def find(
        self,
        scope: Scope,
        *,
        name: str | None = None,
        entity_id: str | None = None,
        limit: int = 20,
    ) -> list[EntityDetail]:
        """Entities matching a name (or alias) or an id, best match first.

        A name matches ignoring case and punctuation and can match several entities of different
        types. Those whose own name matches come before those matched through an alias. At most
        ``limit`` neighbours are returned per entity.
        """
        if entity_id is not None:
            entity = self._entities.get(scope, entity_id)
            found = [entity] if entity else []
        else:
            key = normalize_name(name or "")
            found = [
                e
                for kind in get_args(CuratedType)
                for e in self._entities.block_candidates(
                    scope, entity_type=kind, normalized_name=key
                )
            ]
            found.sort(key=lambda e: (normalize_name(e.name) != key, e.canonical_id))
        return [self._detail(scope, entity, limit) for entity in found]

    def _detail(self, scope: Scope, entity: CanonicalEntity, limit: int) -> EntityDetail:
        neighbours: list[Neighbour] = []
        for edge in sorted(
            self._graph.get_node_edges(scope, entity.canonical_id),
            key=lambda t: (t.valid_to is not None, t.predicate, t.subject_id, t.object_id),
        ):
            outgoing = edge.subject_id == entity.canonical_id
            other_id = edge.object_id if outgoing else edge.subject_id
            other = self._graph.get_node(scope, other_id) or self._entities.get(scope, other_id)
            if other is None:
                continue
            neighbours.append(
                Neighbour(
                    entity=EntityRef.of(other),
                    predicate=edge.predicate,
                    direction="outgoing" if outgoing else "incoming",
                    valid_from=edge.valid_from,
                    valid_to=edge.valid_to,
                    provenance=edge.provenance,
                )
            )
            if len(neighbours) == limit:
                break
        return EntityDetail(**EntityRef.of(entity).model_dump(), neighbours=neighbours)
