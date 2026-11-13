"""Where an ingested document is in the pipeline: queued, running, done or failed (GRA-32).

A job is identified by its ``document_id``, which ``/ingest`` and ``/v1/remember`` return, so the
id a caller already holds is the one they poll. Jobs are kept per scope: one tenant cannot see,
or learn the existence of, another's.

:class:`InMemoryJobStore` lives in the process that runs the pipeline. It is lost on restart and
cannot be shared between an API process and a separate worker (GRA-31 will need a shared store).
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, RootModel

from graph_rag.scope import Scope

__all__ = ["InMemoryJobStore", "Job", "JobState", "JobStore", "JobStreamEvent", "StageEvent"]

JobState = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    """One document's progress."""

    document_id: str
    object_key: str
    state: JobState
    queued_at: str
    updated_at: str
    error: str | None = None  # set when state is "failed": which stage, and the error type
    stages: list[str] = []  # the pipeline stages entered so far, in order (GRA-33)


class StageEvent(BaseModel):
    """The ``data`` of a ``stage`` event on ``GET /jobs/{id}/events``."""

    stage: str


class JobStreamEvent(RootModel[StageEvent | Job]):
    """The ``data`` of one server-sent event: a :class:`StageEvent` for ``stage``, a :class:`Job`
    for the final ``done`` or ``failed``. Only documents the stream in the OpenAPI file."""


class JobStore(Protocol):
    """Records and reads job state."""

    def queued(self, scope: Scope, document_id: str, object_key: str) -> None:
        """A document was accepted; it (re)starts as ``queued``."""
        ...

    def update(
        self,
        scope: Scope,
        document_id: str,
        state: JobState,
        error: str | None = None,
        stage: str | None = None,
    ) -> None:
        """Move a document to ``state``, noting the ``stage`` just entered, if any.

        Unknown documents are recorded too.
        """
        ...

    def get(self, scope: Scope, document_id: str) -> Job | None:
        """The job, or ``None`` if this scope never queued the document."""
        ...


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class InMemoryJobStore:
    """Jobs in a dict, safe to share between the API thread and the consumer thread."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[Scope, str], Job] = {}
        self._lock = threading.Lock()

    def queued(self, scope: Scope, document_id: str, object_key: str) -> None:
        now = _now()
        with self._lock:
            self._jobs[(scope, document_id)] = Job(
                document_id=document_id,
                object_key=object_key,
                state="queued",
                queued_at=now,
                updated_at=now,
            )

    def update(
        self,
        scope: Scope,
        document_id: str,
        state: JobState,
        error: str | None = None,
        stage: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get((scope, document_id))
            if job is None:  # a trigger that did not come through the API (for example Kafka)
                job = Job(
                    document_id=document_id,
                    object_key="",
                    state=state,
                    queued_at=_now(),
                    updated_at=_now(),
                )
            self._jobs[(scope, document_id)] = job.model_copy(
                update={
                    "state": state,
                    "updated_at": _now(),
                    "error": error,
                    "stages": [*job.stages, stage] if stage else job.stages,
                }
            )

    def get(self, scope: Scope, document_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get((scope, document_id))
