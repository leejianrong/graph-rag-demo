"""Elasticsearch adapter for the ``JobStore`` (GRA-31).

With the ingest worker in its own process (``INGEST_MODE``), the API that reads job status and the
worker that writes it no longer share memory, so jobs live in Elasticsearch, which both already
use. One document per ``(scope, document_id)``; reads are by id, which is real-time, so no refresh
is needed. A job has a single writer at a time (the API queues it, then one worker runs it), so
read-modify-write is enough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch, NotFoundError

from graph_rag.adapters.es_client import es_client
from graph_rag.jobs import Job, JobState, _now

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.scope import Scope

__all__ = ["EsJobStore"]


class EsJobStore:
    """Jobs in an Elasticsearch index, shared by every process that points at it."""

    def __init__(self, client: Elasticsearch, index: str) -> None:
        self._client = client
        self._index = index

    @classmethod
    def from_settings(cls, settings: Settings) -> EsJobStore:
        return cls(es_client(settings), settings.jobs_index)

    def ensure_index(self) -> None:
        """Create the index if it is missing; the fields are not searched, only fetched by id."""
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(index=self._index, mappings={"dynamic": False})

    @staticmethod
    def _id(scope: Scope, document_id: str) -> str:
        return f"{scope.key}|{document_id}"  # neither a tenant nor a namespace holds a "|"

    def _put(self, scope: Scope, job: Job) -> None:
        self._client.index(
            index=self._index, id=self._id(scope, job.document_id), document=job.model_dump()
        )

    def queued(self, scope: Scope, document_id: str, object_key: str) -> None:
        now = _now()
        self._put(
            scope,
            Job(
                document_id=document_id,
                object_key=object_key,
                state="queued",
                queued_at=now,
                updated_at=now,
            ),
        )

    def update(
        self,
        scope: Scope,
        document_id: str,
        state: JobState,
        error: str | None = None,
        stage: str | None = None,
        entities: int | None = None,
        edges: int | None = None,
    ) -> None:
        job = self.get(scope, document_id)
        if job is None:  # a trigger that did not come through the API
            now = _now()
            job = Job(
                document_id=document_id,
                object_key="",
                state=state,
                queued_at=now,
                updated_at=now,
            )
        self._put(
            scope,
            job.model_copy(
                update={
                    "state": state,
                    "updated_at": _now(),
                    "error": error,
                    "stages": [*job.stages, stage] if stage else job.stages,
                    "entities": entities,
                    "edges": edges,
                }
            ),
        )

    def get(self, scope: Scope, document_id: str) -> Job | None:
        try:
            found = self._client.get(index=self._index, id=self._id(scope, document_id))
        except NotFoundError:
            return None
        return Job.model_validate(found["_source"])
