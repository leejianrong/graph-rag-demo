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
        ("/v1/remember", {"text": ""}, "invalid_request", 422),
        ("/v1/remember", {"text": " \n\t"}, "invalid_request", 422),
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


def test_namespaces_keep_memory_apart_and_the_default_is_unchanged(stack) -> None:
    client, components, queue = stack
    for ns in ("alpha", "beta", None):
        body = {"text": TEXT, "source_id": "same-note"} | ({"namespace": ns} if ns else {})
        assert client.post("/v1/remember", json=body).status_code == 202
    drain(components, queue)  # the same source id in three namespaces is three documents

    ask = {"question": "Where is Acme Corp based?"}

    def recall(ns: str | None) -> dict:
        return client.post("/v1/recall", json=ask | ({"namespace": ns} if ns else {})).json()

    ids = {
        ns: {s["document_id"] for s in recall(ns)["supporting_sentences"]}
        for ns in ("alpha", "beta", None)
    }
    assert all(ids.values()) and len(set().union(*ids.values())) == 3  # no overlap
    assert recall("gamma")["supporting_sentences"] == []

    gone = client.post("/v1/forget", json={"source_id": "same-note", "namespace": "alpha"})
    assert gone.status_code == 200
    assert recall("alpha")["supporting_sentences"] == []
    assert recall("beta")["supporting_sentences"] and recall(None)["supporting_sentences"]
    again = client.post("/v1/forget", json={"source_id": "same-note", "namespace": "alpha"})
    assert again.status_code == 404  # gone from alpha only


def test_a_namespace_that_cannot_be_a_scope_is_invalid_request(stack) -> None:
    client, _, _ = stack
    for path, body in (
        ("/v1/remember", {"text": "x", "namespace": "a/b"}),
        ("/v1/recall", {"question": "q", "namespace": "a|b"}),
        ("/v1/forget", {"source_id": "s", "namespace": ""}),
    ):
        response = client.post(path, json=body)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


def test_planned_fields_are_ignored_not_rejected(stack) -> None:
    client, _, _ = stack
    response = client.post("/v1/recall", json={"question": "q", "as_of": "2020-01-01"})
    assert response.status_code == 200


# --- lookup (GRA-37) ---------------------------------------------------------------------


@pytest.fixture
def known(stack):
    from graph_rag.lookup import EntityLookup

    client, components, queue = stack
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        lookup=EntityLookup(components.entity_store, components.graph_store),
    )
    client = TestClient(app)
    client.post("/v1/remember", json={"text": TEXT, "source_id": "n"})
    drain(components, queue)
    return client


def test_lookup_by_name_returns_aliases_and_neighbours_with_provenance(known) -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text())
    found = known.post("/v1/lookup", json={"name": "acme corp"})
    assert found.status_code == 200
    assert fits(spec, "LookupResponse", found.json()) == []
    [acme] = found.json()["entities"]
    assert acme["name"] == "Acme Corp" and acme["type"] == "ORG"
    assert acme["neighbours"], "Acme Corp takes part in at least one fact"
    for n in acme["neighbours"]:
        assert n["direction"] in {"outgoing", "incoming"}
        assert n["provenance"]["source_sentence"] and n["entity"]["canonical_id"]
    assert "vector" not in acme


def test_lookup_by_id_and_the_limit(known) -> None:
    acme = known.post("/v1/lookup", json={"name": "Acme Corp"}).json()["entities"][0]
    again = known.post("/v1/lookup", json={"entity_id": acme["canonical_id"]}).json()
    assert again["entities"][0]["canonical_id"] == acme["canonical_id"]
    one = known.post("/v1/lookup", json={"name": "Acme Corp", "limit": 1}).json()
    assert len(one["entities"][0]["neighbours"]) == 1


def test_lookup_errors(known, stack) -> None:
    assert (
        known.post("/v1/lookup", json={"name": "Nobody Inc"}).json()["error"]["code"] == "not_found"
    )
    assert known.post("/v1/lookup", json={"entity_id": "nope"}).status_code == 404
    both = known.post("/v1/lookup", json={"name": "a", "entity_id": "b"})
    assert both.status_code == 400 and both.json()["error"]["code"] == "invalid_request"
    assert known.post("/v1/lookup", json={}).status_code == 400
    assert known.post("/v1/lookup", json={"name": "x", "limit": 0}).status_code == 400
    assert stack[0].post("/v1/lookup", json={"name": "x"}).json()["error"]["code"] == "unavailable"


def test_lookup_stays_inside_its_namespace(known) -> None:
    assert (
        known.post("/v1/lookup", json={"name": "Acme Corp", "namespace": "other"}).status_code
        == 404
    )


def test_a_forgotten_source_leaves_no_entity_to_look_up(known) -> None:
    known.post("/v1/forget", json={"source_id": "n"})
    assert known.post("/v1/lookup", json={"name": "Acme Corp"}).status_code == 404


@pytest.mark.parametrize("text", ["", "   ", " \n\t "])
def test_blank_remember_queues_nothing_and_says_why(stack, text) -> None:
    client, _, queue = stack
    response = client.post("/v1/remember", json={"text": text, "source_id": "blank"})
    assert response.status_code == 422
    assert "empty" in response.json()["error"]["message"]
    assert queue.triggers == []
