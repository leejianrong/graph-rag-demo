"""Fast E2E for ``POST /ingest`` — fakes only, no Docker ($0 gate).

Drives the FastAPI app through the port seam (ADR-0010) with the in-memory fakes:
asserts the endpoint stores the bytes, returns the deterministic ``document_id``,
and publishes exactly one trigger. NOT marked ``contract`` — part of the fast gate.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.fakes import InMemoryObjectStore, InMemoryTriggerPublisher
from graph_rag.ids import document_id

pytestmark = pytest.mark.e2e


def _client(store: InMemoryObjectStore, publisher: InMemoryTriggerPublisher) -> TestClient:
    """Build a TestClient over the app wired to the given fakes (default settings)."""
    return TestClient(create_app(store, publisher, settings=Settings()))


def test_health_ok() -> None:
    """GET /health returns the liveness payload."""
    client = _client(InMemoryObjectStore(), InMemoryTriggerPublisher())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_stores_bytes_and_publishes_trigger() -> None:
    """POST /ingest stores the upload, returns the doc id, publishes one trigger."""
    store = InMemoryObjectStore()
    publisher = InMemoryTriggerPublisher()
    client = _client(store, publisher)

    content = b"# Hello\nGraph RAG walking skeleton.\n"
    filename = "a.md"
    bucket = Settings().minio_bucket

    response = client.post("/ingest", files={"file": (filename, content, "text/markdown")})

    # 1. Response contract: 200 + deterministic document_id + resolved location.
    assert response.status_code == 200
    body = response.json()
    expected_id = document_id(bucket, filename)
    assert body == {
        "document_id": expected_id,
        "bucket": bucket,
        "object_key": filename,
    }

    # 2. The bytes landed in the object store under (bucket, filename).
    assert store.get(bucket, filename) == content

    # 3. Exactly one trigger published, with the right bucket/object_key.
    assert len(publisher.published) == 1
    trigger = publisher.published[0]
    assert trigger.bucket == bucket
    assert trigger.object_key == filename


def test_ingest_does_not_run_pipeline() -> None:
    """The endpoint only stores + publishes; it must not touch downstream stores.

    Proven structurally: the app is wired with ONLY an ObjectStore and a
    TriggerPublisher, so there is no path from /ingest into the pipeline here.
    We assert the trigger carries just {bucket, object_key} (ADR-0001).
    """
    store = InMemoryObjectStore()
    publisher = InMemoryTriggerPublisher()
    client = _client(store, publisher)

    client.post("/ingest", files={"file": ("doc.txt", b"body", "text/plain")})

    trigger = publisher.published[0]
    assert trigger.model_dump().keys() == {"bucket", "object_key"}
