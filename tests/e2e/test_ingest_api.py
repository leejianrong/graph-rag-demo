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
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
    InMemoryTriggerPublisher,
)
from graph_rag.ids import document_id
from graph_rag.models import IngestTrigger
from graph_rag.orchestrator import Orchestrator
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.wiring import HeuristicKgBuildStage, HeuristicNerStage

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
        "status": "accepted",
    }

    # 2. The bytes landed in the object store under (bucket, filename).
    assert store.get(DEFAULT_SCOPE, bucket, filename) == content

    # 3. Exactly one trigger published, with the right bucket/object_key.
    assert len(publisher.published) == 1
    trigger = publisher.published[0]
    assert trigger.bucket == bucket
    assert trigger.object_key == filename


def test_ingest_does_not_run_pipeline() -> None:
    """The endpoint only stores + publishes; it must not touch downstream stores.

    Proven structurally: the app is wired with ONLY an ObjectStore and a
    TriggerPublisher, so there is no path from /ingest into the pipeline here.
    We assert the trigger carries just the location and the scope (ADR-0001, ADR-0015).
    """
    store = InMemoryObjectStore()
    publisher = InMemoryTriggerPublisher()
    client = _client(store, publisher)

    client.post("/ingest", files={"file": ("doc.txt", b"body", "text/plain")})

    trigger = publisher.published[0]
    assert trigger.model_dump().keys() == {"bucket", "object_key", "tenant", "namespace"}


def test_a_second_identical_ingest_is_unchanged_and_publishes_nothing() -> None:
    """The same source with the same bytes is not stored or triggered again (GRA-26)."""
    store, publisher = InMemoryObjectStore(), InMemoryTriggerPublisher()
    client = _client(store, publisher)
    files = {"file": ("a.md", b"same bytes", "text/markdown")}

    first = client.post("/ingest", files=files)
    second = client.post("/ingest", files=files)

    assert first.json()["status"] == "accepted" and second.json()["status"] == "unchanged"
    assert second.json()["document_id"] == first.json()["document_id"]
    assert len(publisher.published) == 1


def test_changed_bytes_for_the_same_source_are_ingested_again() -> None:
    store, publisher = InMemoryObjectStore(), InMemoryTriggerPublisher()
    client = _client(store, publisher)
    client.post("/ingest", files={"file": ("a.md", b"first", "text/markdown")})
    second = client.post("/ingest", files={"file": ("a.md", b"second", "text/markdown")})

    assert second.json()["status"] == "accepted" and len(publisher.published) == 2
    assert store.get(DEFAULT_SCOPE, Settings().minio_bucket, "a.md") == b"second"


def test_the_source_url_keys_the_ingest_instead_of_the_filename() -> None:
    store, publisher = InMemoryObjectStore(), InMemoryTriggerPublisher()
    client = _client(store, publisher)
    url = "https://example.com/notes/a"
    files = {"file": ("upload.txt", b"body", "text/plain")}

    body = client.post("/ingest", files=files, data={"source_url": url}).json()
    again = client.post("/ingest", files=files, data={"source_url": url}).json()
    other = client.post("/ingest", files=files).json()  # no source_url: keyed by the filename

    assert body["object_key"] == url == again["object_key"] and again["status"] == "unchanged"
    assert other["object_key"] == "upload.txt" and other["status"] == "accepted"
    assert body["document_id"] == document_id(Settings().minio_bucket, url)
    assert len(publisher.published) == 2


def test_force_ingests_identical_bytes_again() -> None:
    """A source whose first attempt failed can be pushed through again."""
    store, publisher = InMemoryObjectStore(), InMemoryTriggerPublisher()
    client = _client(store, publisher)
    files = {"file": ("a.md", b"same", "text/markdown")}
    client.post("/ingest", files=files)
    forced = client.post("/ingest", params={"force": "true"}, files=files)

    assert forced.json()["status"] == "accepted" and len(publisher.published) == 2


class _RunsPipeline:
    """A publisher that runs the pipeline at once, standing in for Kafka and its consumer."""

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator
        self.published = 0

    def publish(self, trigger: IngestTrigger) -> None:
        self.published += 1
        self._orchestrator.process_document(trigger)


def test_a_second_identical_ingest_costs_no_llm_call_and_leaves_the_graph_alone() -> None:
    """Through the whole offline pipeline: ingest twice, and the second does no work (GRA-26)."""
    store, graph, llm = InMemoryObjectStore(), InMemoryGraphStore(), FakeLLMClient()
    entities, embedder = InMemoryEntityStore(), FakeEmbedder()
    orchestrator = Orchestrator(
        object_store=store,
        document_store=InMemoryDocumentStore(),
        ner_stage=HeuristicNerStage(),
        coref_stage=LLMCorefStage(llm),
        entity_linking_stage=EntityLinkingStage(entities, embedder, threshold=-1.0, knn_top_k=0),
        graph_store=graph,
        kg_build_stage=HeuristicKgBuildStage(),
    )
    publisher = _RunsPipeline(orchestrator)
    client = TestClient(create_app(store, publisher, settings=Settings()))
    text = b"Ada Lovelace works for Acme Corp. Acme Corp is based in London."
    files = {"file": ("notes.txt", text, "text/plain")}

    assert client.post("/ingest", files=files).json()["status"] == "accepted"
    calls, nodes, edges = (
        llm.calls,
        graph.node_count(DEFAULT_SCOPE),
        graph.edge_count(DEFAULT_SCOPE),
    )
    assert calls > 0 and nodes > 0 and edges > 0  # the first ingest did real work

    assert client.post("/ingest", files=files).json()["status"] == "unchanged"
    assert llm.calls == calls
    assert (graph.node_count(DEFAULT_SCOPE), graph.edge_count(DEFAULT_SCOPE)) == (nodes, edges)
    assert publisher.published == 1
