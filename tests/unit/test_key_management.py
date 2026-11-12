"""``/v1/keys``: a tenant lists, creates and revokes its own API keys (GRA-143)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore, JsonFileKeyStore
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def build(keys: ApiKeys, **settings: object) -> TestClient:
    components = build_offline_components()
    app = create_app(
        components.object_store,
        Queue(),
        Settings(require_auth=True, **settings),  # type: ignore[arg-type]
        retriever=components.retriever,
        keys=keys,
    )
    return TestClient(app)


@pytest.fixture
def keys() -> ApiKeys:
    return ApiKeys(InMemoryKeyStore())


def test_create_returns_the_secret_once_and_list_shows_only_the_suffix(keys: ApiKeys) -> None:
    first, _ = keys.create("acme", "bootstrap")
    client = build(keys)

    made = client.post("/v1/keys", json={"label": "  ci \n runner "}, headers=bearer(first))
    assert made.status_code == 201
    assert made.headers["cache-control"] == "no-store"
    body = made.json()
    secret = body["secret"]
    assert body["label"] == "ci runner" and body["status"] == "active"
    assert body["key_suffix"] == secret[-4:] and body["current"] is False

    listed = client.get("/v1/keys", headers=bearer(first))
    assert listed.status_code == 200
    assert secret not in listed.text and first not in listed.text
    rows = {k["key_id"]: k for k in listed.json()["keys"]}
    assert set(rows) == {body["key_id"], first.split("_")[1]}
    assert rows[body["key_id"]]["key_suffix"] == secret[-4:]
    assert rows[first.split("_")[1]]["current"] is True
    assert all("secret" not in k for k in rows.values())

    # the new key works at once
    assert client.get("/v1/keys", headers=bearer(secret)).status_code == 200


def test_a_revoked_key_gets_401_on_its_very_next_request(keys: ApiKeys) -> None:
    admin, _ = keys.create("acme")
    victim, record = keys.create("acme", "old")
    client = build(keys)
    assert client.get("/v1/keys", headers=bearer(victim)).status_code == 200

    gone = client.delete(f"/v1/keys/{record.key_id}", headers=bearer(admin))
    assert gone.status_code == 200
    assert gone.json()["status"] == "revoked" and gone.json()["revoked_at"]

    after = client.get("/v1/keys", headers=bearer(victim))
    assert after.status_code == 401 and after.json()["error"]["code"] == "unauthorized"
    again = client.delete(f"/v1/keys/{record.key_id}", headers=bearer(admin))
    assert again.status_code == 200  # revoking twice is harmless


def test_a_tenant_cannot_list_or_revoke_another_tenants_keys(keys: ApiKeys) -> None:
    acme, _ = keys.create("acme")
    globex, globex_record = keys.create("globex", "theirs")
    client = build(keys)

    seen = client.get("/v1/keys", headers=bearer(acme)).json()["keys"]
    assert globex_record.key_id not in {k["key_id"] for k in seen}

    attempt = client.delete(f"/v1/keys/{globex_record.key_id}", headers=bearer(acme))
    missing = client.delete("/v1/keys/00000000", headers=bearer(acme))
    assert attempt.status_code == 404 and attempt.json()["error"]["code"] == "not_found"
    assert attempt.json()["error"]["message"].replace(globex_record.key_id, "X") == missing.json()[
        "error"
    ]["message"].replace("00000000", "X")
    assert client.get("/v1/keys", headers=bearer(globex)).status_code == 200  # still alive


def test_the_key_in_use_cannot_be_revoked(keys: ApiKeys) -> None:
    secret, record = keys.create("acme")
    client = build(keys)
    refused = client.delete(f"/v1/keys/{record.key_id}", headers=bearer(secret))
    assert refused.status_code == 400
    assert refused.json()["error"]["details"] == {"reason": "own_key"}
    assert client.get("/v1/keys", headers=bearer(secret)).status_code == 200


def test_active_keys_are_capped_and_create_is_rate_limited(keys: ApiKeys) -> None:
    secret, _ = keys.create("acme")
    client = build(keys, max_keys_per_tenant=3, key_create_limit_per_minute=100)
    assert client.post("/v1/keys", json={}, headers=bearer(secret)).status_code == 201
    assert client.post("/v1/keys", json={}, headers=bearer(secret)).status_code == 201
    full = client.post("/v1/keys", json={}, headers=bearer(secret))
    assert full.status_code == 400 and full.json()["error"]["details"]["reason"] == "key_limit"

    other, _ = keys.create("globex")
    slow = build(keys, key_create_limit_per_minute=1)
    assert slow.post("/v1/keys", json={}, headers=bearer(other)).status_code == 201
    limited = slow.post("/v1/keys", json={}, headers=bearer(other))
    assert limited.status_code == 429 and limited.headers["retry-after"]


@pytest.mark.parametrize("label", ["x" * 65, "bad\x00label", 5])
def test_a_bad_label_is_invalid_request(keys: ApiKeys, label: object) -> None:
    secret, _ = keys.create("acme")
    response = build(keys).post("/v1/keys", json={"label": label}, headers=bearer(secret))
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"


def test_without_a_key_it_is_401_and_with_auth_off_it_is_a_clear_400() -> None:
    client = build(ApiKeys(InMemoryKeyStore()))
    assert client.get("/v1/keys").status_code == 401
    assert client.post("/v1/keys", json={}).status_code == 401

    components = build_offline_components()
    open_app = TestClient(
        create_app(components.object_store, Queue(), Settings(require_auth=False))
    )
    for response in (
        open_app.get("/v1/keys"),
        open_app.post("/v1/keys", json={}),
        open_app.delete("/v1/keys/abcd1234"),
    ):
        assert response.status_code == 400
        assert response.json()["error"]["details"] == {"reason": "auth_disabled"}


def test_the_api_and_the_cli_see_each_others_changes_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    cli = ApiKeys(JsonFileKeyStore(path))
    secret, _ = cli.create("acme", "first")
    client = build(ApiKeys(JsonFileKeyStore(path)))

    made = client.post("/v1/keys", json={"label": "api"}, headers=bearer(secret)).json()
    assert {k.label for k in cli.list("acme")} == {"first", "api"}

    cli.revoke(made["key_id"])  # the CLI revokes; the API sees it on the next request
    assert client.get("/v1/keys", headers=bearer(made["secret"])).status_code == 401
    assert (path.parent / "keys.json.lock").exists()
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_the_secret_never_appears_in_logs(keys: ApiKeys, caplog: pytest.LogCaptureFixture) -> None:
    first, _ = keys.create("acme")
    client = build(keys)
    with caplog.at_level(logging.DEBUG):
        made = client.post("/v1/keys", json={"label": "secretive"}, headers=bearer(first)).json()
        client.get("/v1/keys", headers=bearer(first))
        client.delete(f"/v1/keys/{made['key_id']}", headers=bearer(first))
        client.get("/v1/keys", headers=bearer(made["secret"]))  # now revoked: a failed request
    everything = "\n".join(r.getMessage() + str(r.args) for r in caplog.records)
    assert caplog.records
    assert made["secret"] not in everything and first not in everything
    assert made["secret"].split("_", 2)[2] not in everything
    assert "secretive" not in everything  # a label is the tenant's text; ids only
