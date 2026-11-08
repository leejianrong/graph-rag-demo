"""Ingest job status: queued, running, done, failed (GRA-32)."""

from __future__ import annotations

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
