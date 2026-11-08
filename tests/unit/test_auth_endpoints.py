"""Auth on the endpoints: a key is needed except on /health and reaches only its tenant (GRA-23)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import TEXT, Queue, drain

QUESTION = "Where is Acme Corp based?"


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        keys=keys,
    )
    return TestClient(app), components, queue, keys


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


REQUESTS = [
    ("post", "/ingest", {"files": {"file": ("a.txt", b"hello")}}),
    ("post", "/query", {"json": {"question": QUESTION}}),
    ("delete", "/documents/abc", {}),
    ("post", "/v1/remember", {"json": {"text": "x"}}),
    ("post", "/v1/recall", {"json": {"question": QUESTION}}),
    ("post", "/v1/forget", {"json": {"source_id": "x"}}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), REQUESTS)
def test_no_key_is_401_on_every_endpoint(stack, method, path, kwargs) -> None:
    client, *_ = stack
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer grk_deadbeef_nope"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic abc"},
        {"Authorization": "grk_whatever"},
    ],
)
def test_a_bad_key_is_401(stack, headers) -> None:
    client, *_ = stack
    assert (
        client.post("/v1/recall", json={"question": QUESTION}, headers=headers).status_code == 401
    )
    assert client.post("/query", json={"question": QUESTION}, headers=headers).status_code == 401


def test_v1_answers_in_the_contract_error_shape(stack) -> None:
    client, *_ = stack
    body = client.post("/v1/recall", json={"question": QUESTION}).json()
    assert body["error"]["code"] == "unauthorized" and body["error"]["message"]
    legacy = client.post("/query", json={"question": QUESTION}).json()
    assert "detail" in legacy and "error" not in legacy


def test_an_unauthenticated_request_with_a_bad_body_is_401_not_400(stack) -> None:
    client, *_ = stack
    assert client.post("/v1/remember", json={}).status_code == 401


def test_health_stays_open(stack) -> None:
    client, *_ = stack
    assert client.get("/health").json() == {"status": "ok"}


def test_a_revoked_key_stops_working(stack) -> None:
    client, _, _, keys = stack
    secret, record = keys.create("acme")
    ask = lambda: client.post("/v1/recall", json={"question": QUESTION}, headers=bearer(secret))  # noqa: E731
    assert ask().status_code == 200
    keys.revoke(record.key_id)
    assert ask().status_code == 401


def test_a_key_reads_only_its_own_tenant_and_cannot_forget_anothers(stack) -> None:
    client, components, queue, keys = stack
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")

    stored = client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}, headers=bearer(a))
    assert stored.status_code == 202
    drain(components, queue)

    def seen(secret: str) -> list:
        body = {"question": QUESTION}
        found = client.post("/v1/recall", json=body, headers=bearer(secret))
        return found.json()["supporting_sentences"]

    assert seen(a) and seen(b) == []
    assert (
        client.post("/query", json={"question": QUESTION}, headers=bearer(b)).json()[
            "supporting_sentences"
        ]
        == []
    )
    # B names A's source id and A's document id; neither reaches A's data.
    assert client.post("/v1/forget", json={"source_id": "n"}, headers=bearer(b)).status_code == 404
    doc_id = stored.json()["document_id"]
    assert (
        client.post("/v1/forget", json={"document_id": doc_id}, headers=bearer(b)).status_code
        == 404
    )
    assert client.delete(f"/documents/{doc_id}", headers=bearer(b)).status_code == 404
    assert seen(a)
    assert client.delete(f"/documents/{doc_id}", headers=bearer(a)).status_code == 200


def test_the_same_text_in_two_tenants_gets_different_document_ids(stack) -> None:
    client, _, _, keys = stack
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    body = {"text": TEXT, "source_id": "n"}
    first = client.post("/v1/remember", json=body, headers=bearer(a)).json()
    second = client.post("/v1/remember", json=body, headers=bearer(b)).json()
    assert first["document_id"] != second["document_id"]
    assert second["status"] == "accepted"


def test_ingest_with_a_key_queues_the_keys_tenant(stack) -> None:
    client, _, queue, keys = stack
    secret, _ = keys.create("tenant-a")
    response = client.post("/ingest", files={"file": ("a.txt", b"hello")}, headers=bearer(secret))
    assert response.status_code == 200
    assert [t.tenant for t in queue.triggers] == ["tenant-a"]


def test_with_auth_off_no_key_is_needed_and_the_default_tenant_is_used() -> None:
    components = build_offline_components()
    queue = Queue()
    client = TestClient(create_app(components.object_store, queue, Settings()))
    assert client.post("/v1/remember", json={"text": "x"}).status_code == 202
    assert queue.triggers[0].tenant == "default"
