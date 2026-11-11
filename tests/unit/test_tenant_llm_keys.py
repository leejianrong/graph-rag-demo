"""Tenants bring their own LLM key (GRA-25): stored encrypted, used per tenant, never logged."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import BaseModel

from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.auth import main as keys_main
from graph_rag.config import Settings, get_settings
from graph_rag.scope import acting_for
from graph_rag.tenant_keys import (
    InMemoryTenantKeyStore,
    JsonFileTenantKeyStore,
    LlmKeyRequired,
    TenantLlmKeys,
)
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue

ACME_KEY = "acme-provider-key-sentinel"
GLOBEX_KEY = "globex-provider-key-sentinel"


class Out(BaseModel):
    value: int


SECRET = Fernet.generate_key().decode()


class Backend:
    """Records the key each provider call carried."""

    def __init__(self) -> None:
        self.keys: list[str | None] = []

    def __call__(self, *, model: str, prompt: str, **params: object) -> str:
        self.keys.append(str(params["api_key"]) if "api_key" in params else None)
        return f"answer {len(self.keys)}"


@pytest.fixture
def keys() -> TenantLlmKeys:
    return TenantLlmKeys(InMemoryTenantKeyStore(), SECRET)


def client(tmp_path: Path, keys: TenantLlmKeys | None) -> tuple[LiteLLMClient, Backend]:
    backend = Backend()
    llm = LiteLLMClient(
        "m", cache_dir=tmp_path, api_key="deployment", completion_fn=backend, tenant_keys=keys
    )
    return llm, backend


def test_a_tenants_calls_use_its_own_key_and_no_one_elses(
    tmp_path: Path, keys: TenantLlmKeys
) -> None:
    keys.set("acme", ACME_KEY)
    keys.set("globex", GLOBEX_KEY)
    llm, backend = client(tmp_path, keys)
    with acting_for("acme"):
        llm.complete("one")
    with acting_for("globex"):
        llm.complete("two")
    assert backend.keys == [ACME_KEY, GLOBEX_KEY]


def test_a_tenant_without_a_key_is_refused_and_never_borrows_the_deployments(
    tmp_path: Path, keys: TenantLlmKeys
) -> None:
    keys.set("acme", ACME_KEY)
    llm, backend = client(tmp_path, keys)
    with acting_for("globex"), pytest.raises(LlmKeyRequired, match="globex"):
        llm.complete("one")
    assert backend.keys == []


def test_the_default_tenant_is_unchanged(tmp_path: Path, keys: TenantLlmKeys) -> None:
    llm, backend = client(tmp_path, keys)
    llm.complete("one")  # no key, no error: the deployment's own key is used downstream
    assert backend.keys == [None]


def test_without_the_flag_nothing_changes(tmp_path: Path) -> None:
    assert TenantLlmKeys.from_settings(Settings()) is None
    llm, backend = client(tmp_path, None)
    with acting_for("globex"):
        llm.complete("one")
    assert backend.keys == [None]


def test_a_key_is_stored_encrypted_and_only_that_master_key_reads_it(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    TenantLlmKeys(JsonFileTenantKeyStore(path), SECRET).set("acme", ACME_KEY)
    assert ACME_KEY not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    assert TenantLlmKeys(JsonFileTenantKeyStore(path), SECRET).get("acme") == ACME_KEY
    other = TenantLlmKeys(JsonFileTenantKeyStore(path), Fernet.generate_key().decode())
    assert other.get("acme") is None


def test_a_removed_key_stops_working(keys: TenantLlmKeys) -> None:
    keys.set("acme", ACME_KEY)
    assert keys.remove("acme") and not keys.remove("acme")
    with pytest.raises(LlmKeyRequired):
        keys.require("acme")


def test_the_default_tenant_gets_no_key_and_a_bad_master_key_fails_at_start(
    keys: TenantLlmKeys,
) -> None:
    with pytest.raises(ValueError, match="deployment"):
        keys.set("default", ACME_KEY)
    with pytest.raises(ValueError, match="Fernet"):
        TenantLlmKeys.from_settings(Settings(require_tenant_llm_key=True))


def stack(tmp_path: Path, keys: TenantLlmKeys) -> tuple[TestClient, ApiKeys, Queue]:
    components = build_offline_components()
    queue = Queue()
    api_keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True, require_tenant_llm_key=True),
        keys=api_keys,
        llm_keys=keys,
    )
    return TestClient(app), api_keys, queue


def test_ingest_is_refused_for_a_tenant_with_no_key_and_stores_nothing(
    tmp_path: Path, keys: TenantLlmKeys
) -> None:
    http, api_keys, queue = stack(tmp_path, keys)
    acme, _ = api_keys.create("acme")
    globex, _ = api_keys.create("globex")
    keys.set("acme", ACME_KEY)

    ok = http.post("/v1/remember", json={"text": "Ada met Bob."}, headers=_bearer(acme))
    assert ok.status_code == 202

    refused = http.post("/v1/remember", json={"text": "Ada met Bob."}, headers=_bearer(globex))
    assert refused.status_code == 400
    assert refused.json()["error"]["details"] == {"field": "llm_key"}
    assert "globex" in refused.json()["error"]["message"]
    upload = http.post("/ingest", files={"file": ("a.txt", b"hi")}, headers=_bearer(globex))
    assert upload.status_code == 400 and "no LLM provider key" in upload.json()["detail"]
    assert len(queue.triggers) == 1  # only acme's reached the pipeline


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_the_key_never_appears_in_logs_responses_or_command_output(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "llm-keys.json"
    monkeypatch.setenv("TENANT_LLM_KEYS_PATH", str(path))
    monkeypatch.setenv("TENANT_LLM_KEY_SECRET", SECRET)
    get_settings.cache_clear()
    try:
        with caplog.at_level(logging.DEBUG):
            monkeypatch.setattr("sys.stdin", io.StringIO(ACME_KEY + "\n"))
            assert keys_main(["llm-key", "set", "acme"]) == 0
            assert keys_main(["llm-key", "list"]) == 0

            stored = TenantLlmKeys.from_settings(get_settings(), always=True)
            assert stored is not None and stored.get("acme") == ACME_KEY
            llm, backend = client(tmp_path, stored)
            with acting_for("acme"):
                llm.complete("one")
                with pytest.raises(ValueError):  # fails validation, so it logs warnings
                    llm.structured("two", Out)
            http, api_keys, _ = stack(tmp_path, stored)
            secret, _ = api_keys.create("acme")
            response = http.post("/v1/remember", json={"text": "x"}, headers=_bearer(secret))
            assert keys_main(["llm-key", "remove", "acme"]) == 0
        assert (
            backend.keys[0] == ACME_KEY
        )  # the key was in play, so its absence below means something
    finally:
        get_settings.cache_clear()

    everything = "\n".join(r.getMessage() + str(r.args) for r in caplog.records)
    everything += response.text + capsys.readouterr().out + path.read_text()
    assert caplog.records
    assert ACME_KEY not in everything
