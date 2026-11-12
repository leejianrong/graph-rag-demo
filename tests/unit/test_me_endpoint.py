"""``GET /v1/me``: the tenant a key opens, never the key (GRA-47)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue


def _client(**settings: object) -> tuple[TestClient, ApiKeys]:
    components = build_offline_components()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(components.object_store, Queue(), Settings(**settings), keys=keys)  # type: ignore[arg-type]
    return TestClient(app), keys


def test_a_good_key_gets_its_tenant_and_the_last_four_characters() -> None:
    client, keys = _client(require_auth=True)
    secret, _ = keys.create("acme")
    response = client.get("/v1/me", headers={"Authorization": f"Bearer {secret}"})
    assert response.status_code == 200
    assert response.json() == {"tenant": "acme", "namespace": None, "key_suffix": secret[-4:]}
    assert secret not in response.text


def test_the_namespace_is_echoed_when_the_request_names_one() -> None:
    client, keys = _client(require_auth=True)
    secret, _ = keys.create("acme")
    response = client.get(
        "/v1/me", params={"namespace": "support"}, headers={"Authorization": f"Bearer {secret}"}
    )
    assert response.json()["namespace"] == "support"


def test_a_bad_namespace_is_invalid_request() -> None:
    client, keys = _client(require_auth=True)
    secret, _ = keys.create("acme")
    response = client.get(
        "/v1/me", params={"namespace": "a/b"}, headers={"Authorization": f"Bearer {secret}"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_no_key_or_a_bad_key_is_401_in_the_contract_shape() -> None:
    client, keys = _client(require_auth=True)
    secret, _ = keys.create("acme")
    for headers in ({}, {"Authorization": "Bearer grk_deadbeef_nope"}):
        response = client.get("/v1/me", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
    assert secret not in client.get("/v1/me").text


def test_a_revoked_key_is_401() -> None:
    client, keys = _client(require_auth=True)
    secret, record = keys.create("acme")
    keys.revoke(record.key_id)
    assert client.get("/v1/me", headers={"Authorization": f"Bearer {secret}"}).status_code == 401


def test_with_auth_off_it_is_the_default_tenant() -> None:
    client, _ = _client()
    body = client.get("/v1/me").json()
    assert body["tenant"] == "default" and body["key_suffix"] is None


def test_it_spends_from_the_keys_rate_limit() -> None:
    client, keys = _client(require_auth=True, rate_limit_per_minute=2)
    secret, _ = keys.create("acme")
    headers = {"Authorization": f"Bearer {secret}"}
    assert client.get("/v1/me", headers=headers).status_code == 200
    assert client.get("/v1/me", headers=headers).status_code == 200
    limited = client.get("/v1/me", headers=headers)
    assert limited.status_code == 429 and limited.headers["retry-after"]
