"""Ingest job status: queued, running, done, failed (GRA-32)."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.fakes import InMemoryObjectStore
from graph_rag.jobs import InMemoryJobStore
from graph_rag.models import IngestTrigger
from graph_rag.orchestrator import Orchestrator
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.stages.ner import NerResult
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        retriever=components.retriever,
        jobs=components.job_store,
    )
    return TestClient(app), components, queue


def test_a_document_moves_from_queued_to_done(stack) -> None:
    client, components, queue = stack
    stored = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()
    job = client.get(f"/jobs/{stored['document_id']}").json()
    assert job["state"] == "queued" and job["object_key"] == "n" and job["error"] is None

    drain(components, queue)
    done = client.get(f"/jobs/{stored['document_id']}").json()
    assert done["state"] == "done" and done["error"] is None
    assert done["queued_at"] <= done["updated_at"]
    assert done["entities"] == components.graph_store.node_count(DEFAULT_SCOPE)
    assert done["edges"] == components.graph_store.edge_count(DEFAULT_SCOPE)
    assert job["entities"] is None and job["edges"] is None  # not before it is done


def test_the_job_is_running_while_a_stage_works() -> None:
    from graph_rag.ids import document_id

    store, object_store = InMemoryJobStore(), InMemoryObjectStore()
    components = build_offline_components()
    doc = document_id("b", "k.txt", DEFAULT_SCOPE)
    seen: list[str] = []

    class Probe:
        def analyze(self, text: str) -> NerResult:
            job = store.get(DEFAULT_SCOPE, doc)
            seen.append(job.state if job else "none")
            return NerResult(mentions=[], sentences=[])

    object_store.put(DEFAULT_SCOPE, "b", "k.txt", b"hello")
    store.queued(DEFAULT_SCOPE, doc, "k.txt")
    shell = Orchestrator(
        object_store=object_store,
        document_store=components.document_store,
        ner_stage=Probe(),
        coref_stage=components.orchestrator._coref_stage,  # noqa: SLF001
        job_store=store,
    )
    shell.process_document(IngestTrigger(bucket="b", object_key="k.txt"))
    assert seen == ["running"]
    final = store.get(DEFAULT_SCOPE, doc)
    assert final is not None and final.state == "done"


def test_a_failure_names_the_stage_and_error_type_but_not_the_message(stack) -> None:
    client, components, queue = stack

    class Boom:
        def analyze(self, text: str) -> NerResult:
            raise RuntimeError("connection to db.internal:5432 refused")

    components.orchestrator._ner_stage = Boom()  # noqa: SLF001
    stored = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()
    assert components.orchestrator.process_document(queue.triggers[0]) is None
    job = client.get(f"/jobs/{stored['document_id']}").json()
    assert job["state"] == "failed"
    assert job["error"] == "entity recognition failed: RuntimeError"
    assert "db.internal" not in str(job)


def test_a_missing_object_fails_in_the_read_stage(stack) -> None:
    client, components, _ = stack
    from graph_rag.ids import document_id

    doc = document_id(components.bucket, "ghost.txt", DEFAULT_SCOPE)
    components.orchestrator.process_document(
        IngestTrigger(bucket=components.bucket, object_key="ghost.txt")
    )
    job = client.get(f"/jobs/{doc}").json()
    assert job["state"] == "failed" and job["error"] == "read failed: FileNotFoundError"


def test_an_unknown_id_is_404_and_so_is_an_unconfigured_service(stack) -> None:
    client, components, queue = stack
    assert client.get("/jobs/nope").status_code == 404
    bare = TestClient(create_app(components.object_store, queue, Settings()))
    assert bare.get("/jobs/nope").status_code == 503


def test_resubmitting_unchanged_bytes_leaves_the_finished_job_alone(stack) -> None:
    client, components, queue = stack
    first = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()
    drain(components, queue)
    again = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()
    assert again["status"] == "unchanged"
    assert client.get(f"/jobs/{first['document_id']}").json()["state"] == "done"
    forced = client.post("/v1/remember", json={"text": TEXT, "source_id": "n", "force": True})
    assert forced.status_code == 202
    assert client.get(f"/jobs/{first['document_id']}").json()["state"] == "queued"


def test_a_tenant_sees_only_its_own_jobs_and_needs_a_key() -> None:
    components = build_offline_components()
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True),
        jobs=components.job_store,
        keys=keys,
    )
    client = TestClient(app)
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    doc = client.post(
        "/v1/remember",
        json={"text": TEXT, "source_id": "n"},
        headers={"Authorization": f"Bearer {a}"},
    ).json()["document_id"]

    assert client.get(f"/jobs/{doc}").status_code == 401
    assert (
        client.get(f"/jobs/{doc}", headers={"Authorization": f"Bearer {a}"}).json()["state"]
        == "queued"
    )
    assert client.get(f"/jobs/{doc}", headers={"Authorization": f"Bearer {b}"}).status_code == 404
    assert components.job_store.get(Scope("tenant-b"), doc) is None


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_the_event_stream_has_one_event_per_stage_and_closes_on_done(stack) -> None:
    client, components, queue = stack
    doc = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()["document_id"]
    drain(components, queue)
    with client.stream("GET", f"/jobs/{doc}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events("".join(response.iter_text()))
    assert [name for name, _ in events] == ["stage"] * 5 + ["done"]
    assert [d["stage"] for _, d in events[:5]] == [
        "read",
        "entity recognition",
        "coreference",
        "entity linking",
        "knowledge graph",
    ]
    assert events[-1][1]["state"] == "done"


def test_the_event_stream_closes_on_failed_with_the_stage(stack) -> None:
    client, components, queue = stack

    class Boom:
        def analyze(self, text: str) -> NerResult:
            raise RuntimeError("secret")

    components.orchestrator._ner_stage = Boom()  # noqa: SLF001
    doc = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()["document_id"]
    components.orchestrator.process_document(queue.triggers[0])
    with client.stream("GET", f"/jobs/{doc}/events") as response:
        events = _events("".join(response.iter_text()))
    assert [n for n, _ in events] == ["stage", "stage", "failed"]
    assert events[-1][1]["error"] == "entity recognition failed: RuntimeError"


def test_the_event_stream_follows_a_job_that_is_still_running(stack) -> None:
    client, components, queue = stack
    doc = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()["document_id"]
    worker = threading.Timer(0.2, lambda: drain(components, queue))
    worker.start()
    with client.stream("GET", f"/jobs/{doc}/events") as response:
        events = _events("".join(response.iter_text()))
    worker.join()
    assert events[-1][0] == "done" and len(events) == 6


def test_the_event_stream_is_404_for_another_tenant_and_503_unconfigured() -> None:
    components = build_offline_components()
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True),
        jobs=components.job_store,
        keys=keys,
    )
    client = TestClient(app)
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    doc = client.post(
        "/v1/remember",
        json={"text": TEXT, "source_id": "n"},
        headers={"Authorization": f"Bearer {a}"},
    ).json()["document_id"]
    assert client.get(f"/jobs/{doc}/events").status_code == 401
    other = client.get(f"/jobs/{doc}/events", headers={"Authorization": f"Bearer {b}"})
    assert other.status_code == 404
    bare = TestClient(create_app(components.object_store, queue, Settings()))
    assert bare.get("/jobs/x/events").status_code == 503


# --- GRA-152: posting an unchanged document whose job failed retries it ---------------------------


class _Boom:
    def analyze(self, text: str) -> NerResult:
        raise RuntimeError("down")


def _fail_once(client, components, queue, **post) -> str:
    """Ingest with a failing NER stage, leave the job failed, then restore the real stage."""
    real = components.orchestrator._ner_stage  # noqa: SLF001
    components.orchestrator._ner_stage = _Boom()  # noqa: SLF001
    first = client.post("/ingest", **post).json()
    components.orchestrator.process_document(queue.triggers.pop())
    assert first["status"] == "accepted"
    components.orchestrator._ner_stage = real  # noqa: SLF001
    return first["document_id"]


def _file(text: str = TEXT) -> dict:
    return {"files": {"file": ("a.txt", text.encode(), "text/plain")}}


def test_posting_a_failed_document_again_retries_it_to_done(stack) -> None:
    client, components, queue = stack
    doc = _fail_once(client, components, queue, **_file())
    assert client.get(f"/jobs/{doc}").json()["state"] == "failed"
    again = client.post("/ingest", **_file()).json()
    assert again["status"] == "accepted" and again["document_id"] == doc
    queued = client.get(f"/jobs/{doc}").json()
    assert queued["state"] == "queued" and queued["error"] is None and queued["stages"] == []
    assert len(queue.triggers) == 1
    drain(components, queue)
    assert client.get(f"/jobs/{doc}").json()["state"] == "done"
    assert client.post("/ingest", **_file()).json()["status"] == "unchanged"
    assert queue.triggers == []


def test_a_failed_url_source_is_retried_too() -> None:
    from graph_rag.sources import FetchedPage
    from tests.unit.test_ingest_sources import FakeFetcher

    components = build_offline_components()
    queue = Queue()
    fetcher = FakeFetcher(FetchedPage(TEXT.encode(), "text/plain"))
    client = TestClient(
        create_app(
            components.object_store,
            queue,
            Settings(),
            retriever=components.retriever,
            jobs=components.job_store,
            fetcher=fetcher,
        )
    )
    doc = _fail_once(client, components, queue, data={"url": "https://example.com/a"})
    again = client.post("/ingest", data={"url": "https://example.com/a"}).json()
    assert again["status"] == "accepted" and again["document_id"] == doc
    drain(components, queue)
    assert client.get(f"/jobs/{doc}").json()["state"] == "done"


def test_queued_and_running_documents_are_not_queued_twice(stack) -> None:
    client, components, queue = stack
    doc = client.post("/ingest", **_file()).json()["document_id"]
    assert client.post("/ingest", **_file()).json()["status"] == "unchanged"  # queued
    components.job_store.update(DEFAULT_SCOPE, doc, "running", stage="read")
    assert client.post("/ingest", **_file()).json()["status"] == "unchanged"  # running
    assert len(queue.triggers) == 1


def _metered_client():
    from graph_rag.usage import InMemoryUsageStore, UsageMeter

    components = build_offline_components()
    queue = Queue()
    meter = UsageMeter(InMemoryUsageStore())
    client = TestClient(
        create_app(
            components.object_store,
            queue,
            Settings(),
            retriever=components.retriever,
            jobs=components.job_store,
            meter=meter,
        )
    )
    return client, components, queue, meter


def test_force_still_ingests_identical_bytes_and_counts() -> None:
    client, _, _, meter = _metered_client()
    client.post("/ingest", **_file())
    assert client.post("/ingest?force=true", **_file()).json()["status"] == "accepted"
    assert meter.totals()["default"].documents == 2


def test_a_retry_is_not_metered_again() -> None:
    client, components, queue, meter = _metered_client()
    _fail_once(client, components, queue, **_file())
    assert client.post("/ingest", **_file()).json()["status"] == "accepted"
    assert meter.totals()["default"].documents == 1


def test_another_tenants_copy_of_a_failed_document_is_independent() -> None:
    components = build_offline_components()
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    client = TestClient(
        create_app(
            components.object_store,
            queue,
            Settings(require_auth=True),
            retriever=components.retriever,
            jobs=components.job_store,
            keys=keys,
        )
    )
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    ha, hb = {"Authorization": f"Bearer {a}"}, {"Authorization": f"Bearer {b}"}
    _fail_once(client, components, queue, headers=ha, **_file())
    # tenant b never held these bytes, so it is a normal accept, and b's repeat stays unchanged
    assert client.post("/ingest", headers=hb, **_file()).json()["status"] == "accepted"
    queue.triggers.clear()
    assert client.post("/ingest", headers=hb, **_file()).json()["status"] == "unchanged"
    assert client.post("/ingest", headers=ha, **_file()).json()["status"] == "accepted"
    assert len(queue.triggers) == 1 and queue.triggers[0].tenant == "tenant-a"


@pytest.mark.parametrize("content", [b"", b"  \n\t\n"])
def test_an_empty_upload_is_refused_and_queues_nothing(stack, content: bytes) -> None:
    client, _components, queue = stack
    response = client.post("/ingest", files={"file": ("empty.txt", content, "text/plain")})
    assert response.status_code == 422
    assert response.json()["detail"] == "The document is empty."
    assert queue.triggers == []
