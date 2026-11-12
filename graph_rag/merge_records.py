"""What a manual entity merge folded together, kept so a later split can undo it (GRA-144).

A merge touches three systems with no shared transaction (the entity store, the graph and the
documents' links), so the record is written first, in the ``pending`` state, holding everything a
retry or a split needs: each dropped entity exactly as it was, the edges that touched it and the
document links that named it. :class:`MergeRecordStore` keeps records per scope.

The record is the only copy of a dropped entity once the merge is done. A split rebuilds from it
alone: put each ``DroppedEntity`` back in the entity and graph stores, write its ``edges``, and
point its ``document_links`` at it again. A split (GRA-145) records which dropped ids it has
restored in ``split_ids``; ``status`` is ``splitting`` while one is in flight and ``split`` once
every dropped entity is back (a partial split leaves the record ``done``).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from graph_rag.models import CuratedType, EntityLink, Triple
from graph_rag.scope import Scope

__all__ = [
    "DocumentLink",
    "DroppedEntity",
    "InMemoryMergeRecordStore",
    "MergeRecord",
    "MergeRecordStore",
]

MergeStatus = Literal["pending", "done", "splitting", "split"]


class DocumentLink(BaseModel):
    """One entity link of one document, as stored before the merge (it names the dropped id)."""

    document_id: str
    link: EntityLink


class DroppedEntity(BaseModel):
    """An entity folded into another, with everything that belonged to it."""

    canonical_id: str
    name: str
    type: CuratedType
    aliases: list[str] = Field(default_factory=list)
    first_seen: str | None = None  # UTC ISO-8601; the entity store stamps it, a split restores it
    vector: list[float] | None = None
    edges: list[Triple] = Field(default_factory=list)  # every edge touching it, as stored
    document_links: list[DocumentLink] = Field(default_factory=list)


class MergeRecord(BaseModel):
    """One merge: the surviving entity and the entities dropped into it.

    ``keep_aliases_before`` is the survivor's alias list before the merge, so a split can take
    the dropped names back out without losing aliases the survivor earned from later documents.
    """

    merge_id: str
    keep_id: str
    status: MergeStatus
    created_at: str
    completed_at: str | None = None
    keep_aliases_before: list[str] = Field(default_factory=list)
    dropped: list[DroppedEntity] = Field(default_factory=list)
    # Survivor edges the merge found identical to a moved one and so did not duplicate (GRA-145):
    # a split must leave these in place when it takes the moved copy back out.
    shadowed_edges: list[Triple] = Field(default_factory=list)
    split_ids: list[str] = Field(default_factory=list)  # dropped ids a split has restored
    splitting_ids: list[str] = Field(default_factory=list)  # the ids of the split in flight
    split_at: str | None = None  # when the last split finished

    def holds(self, entity_id: str) -> bool:
        """Whether ``entity_id`` is folded into the survivor by this finished merge right now."""
        return (
            self.status in ("done", "splitting")
            and entity_id not in self.split_ids
            and any(d.canonical_id == entity_id for d in self.dropped)
        )


class MergeRecordStore(Protocol):
    """Merge records, per scope. A scope sees only its own."""

    def put(self, scope: Scope, record: MergeRecord) -> None:
        """Insert or overwrite the record keyed by ``record.merge_id``."""
        ...

    def get(self, scope: Scope, merge_id: str) -> MergeRecord | None:
        """The record, or ``None`` if this scope holds none by that id."""
        ...

    def for_entity(self, scope: Scope, keep_id: str) -> list[MergeRecord]:
        """Every record whose survivor is ``keep_id``, oldest first (``created_at``, then id)."""
        ...

    def containing(self, scope: Scope, dropped_id: str) -> list[MergeRecord]:
        """Every record that dropped ``dropped_id``, oldest first."""
        ...

    def all(self, scope: Scope) -> list[MergeRecord]:
        """Every record in ``scope`` (operator use: export and delete)."""
        ...

    def scopes(self, tenant: str) -> list[Scope]:
        """The scopes of ``tenant`` that hold records (operator use)."""
        ...

    def delete(self, scope: Scope, merge_id: str) -> None:
        """Remove the record; a no-op if it is not there."""
        ...


def _order(record: MergeRecord) -> tuple[str, str]:
    return (record.created_at, record.merge_id)


class InMemoryMergeRecordStore:
    """:class:`MergeRecordStore` in a dict, safe to share between request threads."""

    def __init__(self) -> None:
        self._records: dict[Scope, dict[str, MergeRecord]] = defaultdict(dict)
        self._lock = threading.Lock()

    def put(self, scope: Scope, record: MergeRecord) -> None:
        with self._lock:
            self._records[scope][record.merge_id] = record.model_copy(deep=True)

    def get(self, scope: Scope, merge_id: str) -> MergeRecord | None:
        with self._lock:
            found = self._records[scope].get(merge_id)
            return found.model_copy(deep=True) if found else None

    def for_entity(self, scope: Scope, keep_id: str) -> list[MergeRecord]:
        return [r for r in self.all(scope) if r.keep_id == keep_id]

    def containing(self, scope: Scope, dropped_id: str) -> list[MergeRecord]:
        return [r for r in self.all(scope) if any(d.canonical_id == dropped_id for d in r.dropped)]

    def all(self, scope: Scope) -> list[MergeRecord]:
        with self._lock:
            return sorted(
                (r.model_copy(deep=True) for r in self._records[scope].values()), key=_order
            )

    def scopes(self, tenant: str) -> list[Scope]:
        with self._lock:
            found = [s for s, records in self._records.items() if s.tenant == tenant and records]
        return sorted(found, key=lambda s: (s.tenant, s.namespace or ""))

    def delete(self, scope: Scope, merge_id: str) -> None:
        with self._lock:
            self._records[scope].pop(merge_id, None)
