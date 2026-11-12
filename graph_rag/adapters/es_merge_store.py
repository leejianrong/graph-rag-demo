"""Elasticsearch adapter for the ``MergeRecordStore`` (GRA-144).

One document per ``(scope, merge_id)``, in its own index. Only the fields a query needs are
mapped (``scope``, ``tenant``, ``keep_id``, ``dropped_ids``, ``status``, ``created_at``); the whole
record rides in ``_source`` and is read back from there. Writes refresh, so a read right after a
write sees it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elasticsearch import Elasticsearch, NotFoundError, helpers

from graph_rag.adapters.es_client import es_client
from graph_rag.adapters.scoping import es_in_scope, es_scope_filter, scope_fields, scope_of_key
from graph_rag.merge_records import MergeRecord

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.scope import Scope

__all__ = ["EsMergeRecordStore"]

_MAPPING: dict[str, Any] = {
    "dynamic": False,
    "properties": {
        "merge_id": {"type": "keyword"},
        "tenant": {"type": "keyword"},
        "scope": {"type": "keyword"},
        "keep_id": {"type": "keyword"},
        "dropped_ids": {"type": "keyword"},
        "status": {"type": "keyword"},
        "created_at": {"type": "keyword"},
    },
}


class EsMergeRecordStore:
    """Merge records in an Elasticsearch index."""

    def __init__(self, client: Elasticsearch, index: str) -> None:
        self._client = client
        self._index = index

    @classmethod
    def from_settings(cls, settings: Settings) -> EsMergeRecordStore:
        return cls(es_client(settings), settings.merges_index)

    def ensure_index(self) -> None:
        """Create the index if it is missing."""
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(index=self._index, mappings=_MAPPING)

    @staticmethod
    def _id(scope: Scope, merge_id: str) -> str:
        return f"{scope.key}|{merge_id}"  # neither a tenant nor a namespace holds a "|"

    def put(self, scope: Scope, record: MergeRecord) -> None:
        document = record.model_dump(mode="json")
        document["dropped_ids"] = [d.canonical_id for d in record.dropped]
        document.update(scope_fields(scope))
        self._client.index(
            index=self._index, id=self._id(scope, record.merge_id), document=document, refresh=True
        )

    def get(self, scope: Scope, merge_id: str) -> MergeRecord | None:
        try:
            found = self._client.get(index=self._index, id=self._id(scope, merge_id))
        except NotFoundError:
            return None
        if not es_in_scope(found["_source"], scope):
            return None
        return MergeRecord.model_validate(found["_source"])

    def _find(self, scope: Scope, *clauses: dict[str, Any]) -> list[MergeRecord]:
        query = {"bool": {"filter": [es_scope_filter(scope), *clauses]}}
        try:
            hits = list(helpers.scan(self._client, index=self._index, query={"query": query}))
        except NotFoundError:
            return []
        records = [MergeRecord.model_validate(hit["_source"]) for hit in hits]
        return sorted(records, key=lambda r: (r.created_at, r.merge_id))

    def for_entity(self, scope: Scope, keep_id: str) -> list[MergeRecord]:
        return self._find(scope, {"term": {"keep_id": keep_id}})

    def containing(self, scope: Scope, dropped_id: str) -> list[MergeRecord]:
        return self._find(scope, {"term": {"dropped_ids": dropped_id}})

    def all(self, scope: Scope) -> list[MergeRecord]:
        return self._find(scope)

    def scopes(self, tenant: str) -> list[Scope]:
        try:
            response = self._client.search(
                index=self._index,
                size=0,
                query={"term": {"tenant": tenant}},
                aggs={"scopes": {"terms": {"field": "scope", "size": 10_000}}},
            )
        except NotFoundError:
            return []
        keys = [b["key"] for b in response["aggregations"]["scopes"]["buckets"]]
        return sorted((scope_of_key(tenant, k) for k in keys), key=lambda s: s.key)

    def delete(self, scope: Scope, merge_id: str) -> None:
        if self.get(scope, merge_id) is None:
            return
        try:
            self._client.delete(index=self._index, id=self._id(scope, merge_id), refresh=True)
        except NotFoundError:
            return
