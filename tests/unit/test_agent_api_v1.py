"""The /v1 remember, recall and forget endpoints (ADR-0014, GRA-132), over the offline pipeline."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.models import IngestTrigger
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_contract import SPEC_PATH, fits

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


class Queue:
    """A publisher that keeps the triggers so a test can run the pipeline itself."""

    def __init__(self) -> None:
        self.triggers: list[IngestTrigger] = []

    def publish(self, trigger: IngestTrigger) -> None:
        self.triggers.append(trigger)


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(max_remember_bytes=200),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
    )
    return TestClient(app), components, queue


def drain(components, queue: Queue) -> None:
    for trigger in queue.triggers:
        assert components.orchestrator.process_document(trigger)
    queue.triggers.clear()


def test_remember_then_recall_then_forget_round_trips(stack) -> None:
    client, components, queue = stack
    spec = yaml.safe_load(SPEC_PATH.read_text())

    stored = client.post("/v1/remember", json={"text": TEXT, "source_id": "note-1"})
    assert stored.status_code == 202
    assert stored.json()["status"] == "accepted" and stored.json()["source_id"] == "note-1"
    assert fits(spec, "RememberResponse", stored.json()) == []
    drain(components, queue)

    found = client.post("/v1/recall", json={"question": "Where is Acme Corp based?"})
    assert found.status_code == 200 and found.json()["supporting_sentences"]
    assert fits(spec, "RecallResponse", found.json()) == []

    gone = client.post("/v1/forget", json={"source_id": "note-1"})
    assert gone.status_code == 200 and gone.json()["source_id"] == "note-1"
    assert fits(spec, "ForgetResponse", gone.json()) == []
    after = client.post("/v1/recall", json={"question": "Where is Acme Corp based?"}).json()
    assert after["supporting_sentences"] == []


def test_remember_is_safe_to_repeat_and_names_unnamed_text_by_its_content(stack) -> None:
    client, _, queue = stack
    first = client.post("/v1/remember", json={"text": "same words"}).json()
    again = client.post("/v1/remember", json={"text": "same words"}).json()
    other = client.post("/v1/remember", json={"text": "other words"}).json()

    assert (first["status"], again["status"]) == ("accepted", "unchanged")
    assert first["source_id"] == again["source_id"] != other["source_id"]
    assert len(queue.triggers) == 2
    forced = client.post("/v1/remember", json={"text": "same words", "force": True}).json()
    assert forced["status"] == "accepted"


def test_forget_takes_either_id_and_a_gone_source_is_not_found(stack) -> None:
    client, _, _ = stack
    stored = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}).json()
    # not processed yet, so the pipeline holds no record: nothing to forget
    missing = client.post("/v1/forget", json={"document_id": stored["document_id"]})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    ("path", "body", "code", "status"),
    [
        ("/v1/remember", {"text": ""}, "invalid_request", 400),
        ("/v1/remember", {"source_id": "x"}, "invalid_request", 400),
        ("/v1/remember", {"text": "x" * 201}, "payload_too_large", 413),
        ("/v1/recall", {"question": ""}, "invalid_request", 400),
        ("/v1/forget", {}, "invalid_request", 400),
        ("/v1/forget", {"source_id": "a", "document_id": "b"}, "invalid_request", 400),
    ],
)
def test_every_failure_is_the_one_error_shape(stack, path, body, code, status) -> None:
    client, _, _ = stack
    response = client.post(path, json=body)
    spec = yaml.safe_load(SPEC_PATH.read_text())

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert fits(spec, "Error", response.json()) == []


def test_a_source_id_in_the_reserved_namespace_is_refused(stack) -> None:
    client, _, queue = stack
    response = client.post("/v1/remember", json={"text": "x", "source_id": "_scopes/t/root/k"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"
    ingest = client.post(
        "/ingest", files={"file": ("a.txt", b"x", "text/plain")}, data={"source_url": "_scopes/a"}
    )
    assert ingest.status_code == 400 and queue.triggers == []


def test_malformed_json_is_invalid_request(stack) -> None:
    client, _, _ = stack
    response = client.post(
        "/v1/recall", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"


def test_recall_and_forget_say_unavailable_when_not_wired() -> None:
    components = build_offline_components()
    client = TestClient(create_app(components.object_store, Queue(), Settings()))
    for path, body in (("/v1/recall", {"question": "q"}), ("/v1/forget", {"source_id": "s"})):
        response = client.post(path, json=body)
        assert response.status_code == 503 and response.json()["error"]["code"] == "unavailable"


def test_an_unexpected_failure_is_internal_and_does_not_leak() -> None:
    class Broken:
        def retrieve(self, request):
            raise RuntimeError("secret detail")

    components = build_offline_components()
    app = create_app(components.object_store, Queue(), Settings(), retriever=Broken())
    response = TestClient(app, raise_server_exceptions=False).post(
        "/v1/recall", json={"question": "q"}
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal", "message": "Something went wrong on the server."}
    }


def test_the_old_endpoints_keep_their_own_error_shape(stack) -> None:
    client, _, _ = stack
    assert "detail" in client.post("/query", json={}).json()
    assert "detail" in client.delete("/documents/nothing").json()


def test_planned_fields_are_ignored_not_rejected(stack) -> None:
    client, _, _ = stack
    response = client.post(
        "/v1/recall", json={"question": "q", "namespace": "a", "as_of": "2020-01-01"}
    )
    assert response.status_code == 200
