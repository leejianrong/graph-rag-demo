"""Browse the entity graph: ``GET /v1/entities`` and ``GET /v1/entities/{id}`` (GRA-142, ADR-0014).

For the explorer UI. Search pages through a scope's entities by name or alias, newest first; the
detail call returns one entity with the facts on each side of it, its aliases and the documents
that mention it. Everything is read through the store ports under the caller's scope, so one
tenant (or namespace) never sees another's entities. Nothing here returns a vector or document
text: a document is its id and source name, a neighbour's provenance is what ``lookup`` already
exposes.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from graph_rag.agent_api import AgentError, _scope
from graph_rag.lookup import EntityRef, Neighbour
from graph_rag.models import CanonicalEntity, CuratedType, EntitySource, Triple

if TYPE_CHECKING:
    from graph_rag.ports import DocumentStore, EntityStore, GraphStore
    from graph_rag.scope import Scope

__all__ = [
    "EntityExplorer",
    "EntityNeighbourhood",
    "EntitySearchResponse",
    "EntitySummary",
    "add_entity_routes",
]

_MAX_LIMIT = 100


class EntitySummary(EntityRef):
    """An entity as a search result: its id, name, type, aliases and when it first appeared."""

    first_seen: str | None = None  # UTC ISO-8601; ``None`` for one stored before it was kept


class EntitySearchResponse(BaseModel):
    entities: list[EntitySummary]
    next_cursor: str | None = (
        None  # pass it back as ``cursor`` for the next page; ``None`` at the end
    )


class EntityNeighbourhood(EntitySummary):
    """One entity with both sides of its edges and the documents that mention it.

    ``neighbours`` lists up to ``neighbours_limit`` outgoing edges (this entity is the subject)
    and as many incoming ones, outgoing first; the totals say how many exist. ``sources`` is
    most-mentioned first and cut at ``sources_limit``.
    """

    neighbours: list[Neighbour] = []
    outgoing_total: int = 0
    incoming_total: int = 0
    sources: list[EntitySource] = []
    sources_total: int = 0


def _encode(first_seen: str, canonical_id: str) -> str:
    raw = json.dumps([first_seen, canonical_id]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(cursor: str) -> tuple[str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        seen, last = json.loads(base64.urlsafe_b64decode(padded))
        if isinstance(seen, str) and isinstance(last, str):
            return seen, last
    except (binascii.Error, ValueError, TypeError):
        pass
    raise AgentError(
        "invalid_request", "cursor: not a cursor this API issued.", {"field": "cursor"}
    )


class EntityExplorer:
    """Searches entities and describes one with its neighbours and sources."""

    def __init__(
        self, entity_store: EntityStore, graph_store: GraphStore, document_store: DocumentStore
    ) -> None:
        self._entities = entity_store
        self._graph = graph_store
        self._documents = document_store

    def search(
        self,
        scope: Scope,
        *,
        query: str | None,
        entity_type: CuratedType | None,
        limit: int,
        cursor: str | None,
    ) -> EntitySearchResponse:
        """One page of entities matching ``query`` in ``scope``, newest first."""
        after = _decode(cursor) if cursor else None
        rows = self._entities.search(
            scope, query=query, entity_type=entity_type, limit=limit + 1, after=after
        )
        page = rows[:limit]
        more = _encode(page[-1][1], page[-1][0].canonical_id) if len(rows) > limit else None
        return EntitySearchResponse(
            entities=[self._summary(entity, seen) for entity, seen in page], next_cursor=more
        )

    def detail(
        self, scope: Scope, entity_id: str, *, neighbours_limit: int, sources_limit: int
    ) -> EntityNeighbourhood | None:
        """The entity with its neighbours and sources, or ``None`` if the scope has no such id."""
        entity = self._entities.get(scope, entity_id)
        if entity is None:
            return None
        outgoing: list[Triple] = []
        incoming: list[Triple] = []
        for edge in sorted(
            self._graph.get_node_edges(scope, entity_id),
            key=lambda t: (
                t.valid_to is not None,
                t.predicate,
                t.subject_id,
                t.object_id,
                t.provenance.source_doc_id,
                t.provenance.sentence_index,
            ),
        ):
            (outgoing if edge.subject_id == entity_id else incoming).append(edge)
        neighbours = [
            n
            for edges, direction in ((outgoing, "outgoing"), (incoming, "incoming"))
            for n in self._neighbours(scope, entity_id, edges[:neighbours_limit], direction)
        ]
        sources = self._documents.entity_sources(scope, entity_id)
        return EntityNeighbourhood(
            **self._summary(entity, self._entities.first_seen(scope, entity_id)).model_dump(),
            neighbours=neighbours,
            outgoing_total=len(outgoing),
            incoming_total=len(incoming),
            sources=sources[:sources_limit],
            sources_total=len(sources),
        )

    def _neighbours(
        self, scope: Scope, entity_id: str, edges: list[Triple], direction: str
    ) -> list[Neighbour]:
        found = []
        for edge in edges:
            other_id = edge.object_id if direction == "outgoing" else edge.subject_id
            other = self._graph.get_node(scope, other_id) or self._entities.get(scope, other_id)
            if other is None:
                continue
            found.append(
                Neighbour(
                    entity=EntityRef.of(other),
                    predicate=edge.predicate,
                    direction=direction,
                    valid_from=edge.valid_from,
                    valid_to=edge.valid_to,
                    provenance=edge.provenance,
                )
            )
        return found

    @staticmethod
    def _summary(entity: CanonicalEntity, first_seen: str | None) -> EntitySummary:
        return EntitySummary(**EntityRef.of(entity).model_dump(), first_seen=first_seen or None)


def add_entity_routes(
    router: APIRouter, explorer: EntityExplorer | None, tenant_of: Callable[[Request], str]
) -> None:
    """Serve ``GET /v1/entities`` and ``GET /v1/entities/{entity_id}`` on the ``/v1`` router."""

    def ready() -> EntityExplorer:
        if explorer is None:
            raise AgentError("unavailable", "Entity search is not configured on this service.")
        return explorer

    @router.get("/entities")
    def search_entities(
        q: Annotated[str | None, Query(max_length=200, description="A name or alias.")] = None,
        entity_type: Annotated[
            CuratedType | None, Query(alias="type", description="Only entities of this type.")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = 20,
        cursor: Annotated[str | None, Query(description="`next_cursor` of the last page.")] = None,
        namespace: Annotated[str | None, Query(min_length=1)] = None,
        tenant: str = Depends(tenant_of),
    ) -> EntitySearchResponse:
        """Search entities by name or alias, newest first, one page at a time."""
        return ready().search(
            _scope(tenant, namespace), query=q, entity_type=entity_type, limit=limit, cursor=cursor
        )

    @router.get("/entities/{entity_id}")
    def entity_detail(
        entity_id: str,
        neighbours_limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = 25,
        sources_limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = 20,
        namespace: Annotated[str | None, Query(min_length=1)] = None,
        tenant: str = Depends(tenant_of),
    ) -> EntityNeighbourhood:
        """One entity with its aliases, neighbours (both directions) and source documents."""
        found = ready().detail(
            _scope(tenant, namespace),
            entity_id,
            neighbours_limit=neighbours_limit,
            sources_limit=sources_limit,
        )
        if found is None:
            raise AgentError("not_found", f"No entity {entity_id}.")
        return found
