"""Curate entities by hand: ``POST /v1/entities/{keep_id}/merge`` (GRA-144, ADR-0014 addendum).

Entity linking makes mistakes a person must correct. A merge folds one or more entities into a
survivor the way GRA-30 folds duplicates: the survivor gains their names as aliases, their edges
move onto it (``GraphStore.merge_nodes``), the documents' links to them name the survivor, and
they leave the entity store. What was folded is stored first as a :class:`MergeRecord`, so a split
(GRA-145) can restore it and a retry can finish a merge that stopped half-way.

The three systems have no shared transaction. The record is written ``pending``; each step after
it is safe to run again; the record is marked ``done`` last. A crash leaves a ``pending`` record
and some of the steps applied. Any later merge request that touches the survivor (or one of the
dropped ids) first finishes every pending record it finds, so the retry heals it.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from graph_rag.agent_api import AgentError, _scope
from graph_rag.entities_api import EntitySummary
from graph_rag.logging import get_logger
from graph_rag.merge_records import DocumentLink, DroppedEntity, MergeRecord
from graph_rag.normalize import normalize_name
from graph_rag.timeutil import utc_now_iso

if TYPE_CHECKING:
    from graph_rag.merge_records import MergeRecordStore
    from graph_rag.models import CanonicalEntity
    from graph_rag.ports import DocumentStore, EntityStore, GraphStore
    from graph_rag.scope import Scope

__all__ = ["EntityCurator", "MergeRequest", "MergeResponse", "add_curation_routes"]

_logger = get_logger(__name__)
_MAX_MERGE = 10


class MergeRequest(BaseModel):
    merge_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        max_length=_MAX_MERGE,
        description=f"Entities to fold into the one in the path: 1 to {_MAX_MERGE}, no repeats.",
    )
    namespace: str | None = Field(default=None, min_length=1)

    @field_validator("merge_ids")
    @classmethod
    def _distinct(cls, ids: list[str]) -> list[str]:
        if len(set(ids)) != len(ids):
            raise ValueError("must not repeat an id")
        return ids


class MergeResponse(BaseModel):
    """What a merge did. ``entity`` is the survivor as it is now."""

    merge_id: str = Field(
        description="The record of this merge; for a repeat, the record of the earlier one."
    )
    merged: list[str] = Field(description="Ids this call folded into the survivor.")
    already_merged: list[str] = Field(description="Ids an earlier call had folded in already.")
    entity: EntitySummary


class EntityCurator:
    """Merges entities in one scope and records what it merged."""

    def __init__(
        self,
        entity_store: EntityStore,
        graph_store: GraphStore,
        document_store: DocumentStore,
        merges: MergeRecordStore,
    ) -> None:
        self._entities = entity_store
        self._graph = graph_store
        self._documents = document_store
        self._merges = merges
        self._lock = threading.Lock()  # one merge at a time in this process

    def merge(self, scope: Scope, keep_id: str, merge_ids: list[str]) -> MergeResponse:
        """Fold ``merge_ids`` into ``keep_id``; see the module docstring for the steps."""
        if keep_id in merge_ids:
            raise AgentError(
                "invalid_request",
                "merge_ids must not contain the entity being kept.",
                {"field": "merge_ids"},
            )
        with self._lock:
            if self._entities.get(scope, keep_id) is None:
                raise AgentError("not_found", f"No entity {keep_id}.")
            self._heal(scope, keep_id, merge_ids)
            keep = self._entities.get(scope, keep_id)
            if keep is None:  # healing another merge cannot remove the survivor; be explicit
                raise AgentError("not_found", f"No entity {keep_id}.")
            fresh, already = self._classify(scope, keep, merge_ids)
            if not fresh:
                return self._response(scope, keep_id, already[0][1].merge_id, [], already)
            record = MergeRecord(
                merge_id=uuid.uuid4().hex,
                keep_id=keep_id,
                status="pending",
                created_at=utc_now_iso(),
                keep_aliases_before=list(keep.aliases),
                dropped=self._snapshot(scope, fresh),
            )
            self._merges.put(scope, record)  # first: everything below can be re-run from this
            self._apply(scope, record)
            _logger.info("merged entities merge_id=%s dropped=%d", record.merge_id, len(fresh))
            merged = [e.canonical_id for e in fresh]
            return self._response(scope, keep_id, record.merge_id, merged, already)

    def _heal(self, scope: Scope, keep_id: str, merge_ids: list[str]) -> None:
        """Finish every ``pending`` record that involves the survivor or a requested id."""
        records = {r.merge_id: r for r in self._merges.for_entity(scope, keep_id)}
        for merge_id in merge_ids:
            records.update({r.merge_id: r for r in self._merges.containing(scope, merge_id)})
        for record in sorted(records.values(), key=lambda r: (r.created_at, r.merge_id)):
            if record.status == "pending":
                _logger.warning("finishing an interrupted merge merge_id=%s", record.merge_id)
                self._apply(scope, record)

    def _classify(
        self, scope: Scope, keep: CanonicalEntity, merge_ids: list[str]
    ) -> tuple[list[CanonicalEntity], list[tuple[str, MergeRecord]]]:
        """Split ``merge_ids`` into entities still to merge and ids already merged into ``keep``.

        Raises before anything is written: ``not_found`` for an id this scope does not hold (an
        id merged into a different entity says so in ``details.merged_into``), ``conflict`` for
        an entity of another type, which entity linking never unifies either.
        """
        fresh: list[CanonicalEntity] = []
        already: list[tuple[str, MergeRecord]] = []
        for merge_id in merge_ids:
            entity = self._entities.get(scope, merge_id)
            if entity is not None:
                fresh.append(entity)
                continue
            done = [r for r in self._merges.containing(scope, merge_id) if r.status == "done"]
            if done and done[-1].keep_id == keep.canonical_id:
                already.append((merge_id, done[-1]))
                continue
            raise AgentError(
                "not_found",
                f"No entity {merge_id}.",
                {"merged_into": done[-1].keep_id} if done else None,
            )
        mismatched = [e.canonical_id for e in fresh if e.type != keep.type]
        if mismatched:
            raise AgentError(
                "conflict",
                f"Only entities of the same type merge; {keep.canonical_id} is {keep.type}.",
                {"reason": "type_mismatch", "entity_ids": mismatched},
            )
        return fresh, already

    def _snapshot(self, scope: Scope, entities: list[CanonicalEntity]) -> list[DroppedEntity]:
        """Each entity as it is now, with its edges and the document links that name it."""
        ids = {e.canonical_id for e in entities}
        links: dict[str, list[DocumentLink]] = {i: [] for i in ids}
        for record in self._documents.all(scope):
            for link in record.el_result:
                if link.canonical_id in ids:
                    links[link.canonical_id].append(
                        DocumentLink(document_id=record.document_id, link=link)
                    )
        return [
            DroppedEntity(
                canonical_id=e.canonical_id,
                name=e.name,
                type=e.type,
                aliases=list(e.aliases),
                first_seen=self._entities.first_seen(scope, e.canonical_id),
                vector=e.vector,
                edges=self._graph.get_node_edges(scope, e.canonical_id),
                document_links=links[e.canonical_id],
            )
            for e in entities
        ]

    def _apply(self, scope: Scope, record: MergeRecord) -> None:
        """Run the merge steps in an order that is safe to repeat, then mark the record done.

        Aliases first (so the survivor answers to the dropped names before they vanish), then the
        graph, then the documents' links, then the dropped entities leave the entity store.
        """
        keep = self._entities.get(scope, record.keep_id)
        if keep is None:
            raise AgentError("internal", f"The survivor {record.keep_id} of a merge is gone.")
        drop_ids = [d.canonical_id for d in record.dropped]
        aliases = list(keep.aliases)
        known = {normalize_name(keep.name), *(normalize_name(a) for a in aliases)}
        for dropped in record.dropped:
            for surface in (dropped.name, *dropped.aliases):
                if normalize_name(surface) and normalize_name(surface) not in known:
                    known.add(normalize_name(surface))
                    aliases.append(surface)
        survivor = keep.model_copy(update={"aliases": aliases})
        if aliases != keep.aliases:
            self._entities.upsert(scope, survivor)
        self._graph.upsert_entities(scope, [survivor.model_copy(update={"vector": None})])
        self._graph.merge_nodes(scope, record.keep_id, drop_ids)
        gone = set(drop_ids)
        for document in self._documents.all(scope):
            if any(link.canonical_id in gone for link in document.el_result):
                document.el_result = [
                    link.model_copy(update={"canonical_id": record.keep_id})
                    if link.canonical_id in gone
                    else link
                    for link in document.el_result
                ]
                self._documents.upsert(scope, document)
        for drop_id in drop_ids:
            self._entities.delete(scope, drop_id)
        self._merges.put(
            scope, record.model_copy(update={"status": "done", "completed_at": utc_now_iso()})
        )

    def _response(
        self,
        scope: Scope,
        keep_id: str,
        merge_id: str,
        merged: list[str],
        already: list[tuple[str, MergeRecord]],
    ) -> MergeResponse:
        entity = self._entities.get(scope, keep_id)
        assert entity is not None  # the survivor is checked at the start and never removed
        return MergeResponse(
            merge_id=merge_id,
            merged=merged,
            already_merged=[i for i, _ in already],
            entity=EntitySummary(
                canonical_id=entity.canonical_id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                first_seen=self._entities.first_seen(scope, entity.canonical_id) or None,
            ),
        )


def add_curation_routes(
    router: APIRouter, curator: EntityCurator | None, tenant_of: Callable[[Request], str]
) -> None:
    """Serve ``POST /v1/entities/{keep_id}/merge`` on the ``/v1`` router."""

    @router.post("/entities/{keep_id}/merge")
    def merge_entities(
        keep_id: str, body: MergeRequest, tenant: str = Depends(tenant_of)
    ) -> MergeResponse:
        """Fold the entities in `merge_ids` into this one, recording what was merged."""
        if curator is None:
            raise AgentError("unavailable", "Entity curation is not configured on this service.")
        return curator.merge(_scope(tenant, body.namespace), keep_id, body.merge_ids)
