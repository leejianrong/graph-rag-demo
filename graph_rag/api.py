"""FastAPI app exposing ``POST /ingest`` (U1/N1) and ``GET /health``.

The ingest endpoint is a **thin** entry point (ADR-0001): it stores the uploaded
bytes via the ``ObjectStore`` port, computes the deterministic ``document_id``, and
publishes an :class:`~graph_rag.models.IngestTrigger` via the ``TriggerPublisher``
port. It does **not** run the pipeline — the Kafka consumer + orchestrator do that
downstream.

Dependencies are injected through :func:`create_app` so the real composition root
(main.py) passes live adapters while tests pass in-memory fakes; the endpoint never
constructs an adapter itself (ADR-0010).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, UploadFile

from graph_rag.config import Settings, get_settings
from graph_rag.ids import document_id
from graph_rag.logging import get_logger
from graph_rag.models import IngestTrigger
from graph_rag.ports import ObjectStore, TriggerPublisher

__all__ = ["create_app"]

_logger = get_logger(__name__)


def create_app(
    object_store: ObjectStore,
    publisher: TriggerPublisher,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the FastAPI app with its ports injected.

    Args:
        object_store: Where uploaded bytes are stored (MinIO in prod, fake in tests).
        publisher: Where the ingest trigger is published (Kafka in prod, fake in tests).
        settings: Runtime settings; falls back to :func:`~graph_rag.config.get_settings`.

    Returns:
        A configured :class:`fastapi.FastAPI` app exposing ``POST /ingest`` and
        ``GET /health``.
    """
    resolved_settings = settings or get_settings()
    bucket = resolved_settings.minio_bucket

    app = FastAPI(title="Graph RAG Demo — Ingest API")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe (dev-playbook observability floor)."""
        return {"status": "ok"}

    @app.post("/ingest")
    async def ingest(file: Annotated[UploadFile, File()]) -> dict[str, str]:
        """Store an uploaded document and publish its ingest trigger.

        The object key is the uploaded filename; the bucket comes from settings.
        Returns the deterministic ``document_id`` plus the resolved location. Does
        NOT run the pipeline — that happens downstream off the published trigger.
        """
        object_key = file.filename or "upload"
        data = await file.read()

        object_store.put(bucket, object_key, data)
        doc_id = document_id(bucket, object_key)
        publisher.publish(IngestTrigger(bucket=bucket, object_key=object_key))

        _logger.info(
            "ingested document_id=%s bucket=%s object_key=%s bytes=%d",
            doc_id,
            bucket,
            object_key,
            len(data),
        )
        return {"document_id": doc_id, "bucket": bucket, "object_key": object_key}

    return app
