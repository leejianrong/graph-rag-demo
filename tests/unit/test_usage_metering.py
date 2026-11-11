"""Per-tenant usage metering (GRA-58): totals match traffic, survive restarts, stay private."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings, get_settings
from graph_rag.scope import acting_for
from graph_rag.tenant_admin import main as admin_main
from graph_rag.usage import InMemoryUsageStore, JsonFileUsageStore, Usage, UsageMeter, UsageStore
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue

QUESTION = "Where is Acme Corp based?"


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> UsageStore:
    """Both stores must behave alike (the file one is the real adapter)."""
    return InMemoryUsageStore() if request.param == "memory" else JsonFileUsageStore(tmp_path / "u")


def test_stores_add_per_tenant_and_start_at_zero(store: UsageStore) -> None:
    assert store.get("acme") == Usage()
    store.add("acme", Usage(documents=1))
    store.add("acme", Usage(queries=2, llm_tokens=30))
    store.add("globex", Usage(documents=5))
    assert store.get("acme") == Usage(1, 2, 30)
    assert store.all() == {"acme": Usage(1, 2, 30), "globex": Usage(5, 0, 0)}


def test_the_file_store_survives_a_restart(tmp_path: Path) -> None:
    JsonFileUsageStore(tmp_path / "u.json").add("acme", Usage(documents=3))
    assert JsonFileUsageStore(tmp_path / "u.json").get("acme") == Usage(documents=3)


def stack(path: Path):
    components = build_offline_components()
    api_keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        Queue(),
        Settings(require_auth=True, meter_usage=True, usage_path=str(path)),
        retriever=components.retriever,
        keys=api_keys,
    )
    return TestClient(app), api_keys


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_totals_match_the_traffic_of_each_tenant(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    http, api_keys = stack(path)
    acme, _ = api_keys.create("acme")
    globex, _ = api_keys.create("globex")

    for n in range(3):
        http.post("/v1/remember", json={"text": f"Ada met Bob {n}."}, headers=bearer(acme))
    # the same text again is unchanged: nothing runs, so nothing is counted
    http.post("/v1/remember", json={"text": "Ada met Bob 0."}, headers=bearer(acme))
    http.post("/ingest", files={"file": ("a.txt", b"hi")}, headers=bearer(globex))
    for _ in range(2):
        http.post("/query", json={"question": QUESTION}, headers=bearer(acme))
    http.post("/v1/recall", json={"question": QUESTION}, headers=bearer(globex))
    # refused requests are not usage
    assert http.post("/query", json={"question": QUESTION}).status_code == 401
    assert http.post("/v1/recall", json={}, headers=bearer(globex)).status_code == 400

    assert JsonFileUsageStore(path).all() == {
        "acme": Usage(documents=3, queries=2),
        "globex": Usage(documents=1, queries=1),
    }


def test_force_counts_as_a_new_ingest(tmp_path: Path) -> None:
    http, api_keys = stack(tmp_path / "usage.json")
    key, _ = api_keys.create("acme")
    body = {"text": "Ada met Bob."}
    http.post("/v1/remember", json=body, headers=bearer(key))
    http.post("/v1/remember", json={**body, "force": True}, headers=bearer(key))
    assert JsonFileUsageStore(tmp_path / "usage.json").get("acme").documents == 2


def test_metering_is_off_by_default_and_writes_nothing(tmp_path: Path) -> None:
    assert UsageMeter.from_settings(Settings()) is None
    components = build_offline_components()
    app = create_app(components.object_store, Queue(), Settings(usage_path=str(tmp_path / "u")))
    TestClient(app).post("/ingest", files={"file": ("a.txt", b"hi")})
    assert not (tmp_path / "u").exists()


class Response:
    """What litellm returns: a completion and the tokens it cost."""

    def __init__(self, tokens: int) -> None:
        message = types.SimpleNamespace(content="done")
        self.choices = [types.SimpleNamespace(message=message)]
        self.usage = types.SimpleNamespace(total_tokens=tokens)


def test_llm_tokens_are_counted_for_the_acting_tenant_and_cache_hits_are_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("litellm")
    fake.completion = lambda **_: Response(40)  # type: ignore[attr-defined]
    fake.completion_cost = lambda **_: 0.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake)
    store = InMemoryUsageStore()
    llm = LiteLLMClient("m", cache_dir=tmp_path, meter=UsageMeter(store))
    with acting_for("acme"):
        llm.complete("one")
        llm.complete("one")  # a cache hit: no call, no tokens
        llm.complete("two")
    with acting_for("globex"):
        llm.complete("one")  # its own cache directory, so a call
    llm.complete("three")
    assert store.all() == {
        "acme": Usage(llm_tokens=80),
        "globex": Usage(llm_tokens=40),
        "default": Usage(llm_tokens=40),
    }


def test_operators_read_the_totals_with_the_admin_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "usage.json"
    JsonFileUsageStore(path).add("acme", Usage(2, 3, 400))
    JsonFileUsageStore(path).add("globex", Usage(documents=1))
    monkeypatch.setenv("USAGE_PATH", str(path))
    get_settings.cache_clear()
    try:
        assert admin_main(["usage"]) == 0
        both = capsys.readouterr().out
        assert admin_main(["usage", "acme"]) == 0
        one = capsys.readouterr().out
    finally:
        get_settings.cache_clear()
    assert "acme: 2 documents, 3 queries, 400 llm tokens" in both and "globex: 1 documents" in both
    assert "acme: 2 documents" in one and "globex" not in one
