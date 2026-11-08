"""The /ingest upload cap (GRA-135): over the limit is refused and nothing is stored or queued."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.fakes import InMemoryObjectStore
from graph_rag.scope import DEFAULT_SCOPE
from tests.unit.test_agent_api_v1 import Queue

LIMIT = 1000


@pytest.fixture
def stack() -> tuple[TestClient, InMemoryObjectStore, Queue]:
    store, queue = InMemoryObjectStore(), Queue()
    app = create_app(store, queue, Settings(max_upload_bytes=LIMIT))
    return TestClient(app), store, queue


def _upload(client: TestClient, size: int, **kw: object):
    return client.post("/ingest", files={"file": ("doc.txt", b"x" * size)}, **kw)


def test_an_upload_at_the_limit_is_accepted(stack) -> None:
    client, store, queue = stack
    assert _upload(client, LIMIT).status_code == 200
    assert store.get(DEFAULT_SCOPE, Settings().minio_bucket, "doc.txt") == b"x" * LIMIT
    assert len(queue.triggers) == 1


def test_an_upload_over_the_limit_is_refused_and_leaves_nothing(stack) -> None:
    client, store, queue = stack
    response = _upload(client, LIMIT + 1)
    assert response.status_code == 413
    assert str(LIMIT) in response.json()["detail"]
    assert store.keys(DEFAULT_SCOPE) == [] and queue.triggers == []


def test_a_declared_huge_body_is_refused_before_it_is_read(stack) -> None:
    client, store, queue = stack
    response = client.post(
        "/ingest", content=b"x", headers={"content-length": str(1_000_000), "content-type": "x"}
    )
    assert response.status_code == 413
    assert store.keys(DEFAULT_SCOPE) == [] and queue.triggers == []


def test_the_cap_applies_to_a_tenant_with_a_key_too() -> None:
    keys = ApiKeys(InMemoryKeyStore())
    secret, _ = keys.create("acme")
    store, queue = InMemoryObjectStore(), Queue()
    app = create_app(store, queue, Settings(max_upload_bytes=LIMIT, require_auth=True), keys=keys)
    client = TestClient(app)
    auth = {"authorization": f"Bearer {secret}"}
    assert _upload(client, LIMIT + 1, headers=auth).status_code == 413
    assert queue.triggers == []
    assert _upload(client, 10, headers=auth).status_code == 200
