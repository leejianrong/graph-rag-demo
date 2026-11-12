"""The built UI is served at / by the API process without shadowing the API (GRA-53)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.offline_server import build_app
from graph_rag.ui_static import UI_HEADERS, mount_ui
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue

PAGE = "<!doctype html><title>fake ui</title><div id=app></div>"


@pytest.fixture
def ui_dir(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text(PAGE)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)")
    return tmp_path


def make_client(settings: Settings) -> TestClient:
    components = build_offline_components()
    app = create_app(
        components.object_store,
        Queue(),
        settings,
        retriever=components.retriever,
        keys=ApiKeys(InMemoryKeyStore()),
    )
    mount_ui(app, settings)
    return TestClient(app)


def test_index_is_served_with_security_headers(ui_dir: Path) -> None:
    client = make_client(Settings(ui_dir=str(ui_dir)))
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == PAGE
    assert response.headers["cache-control"] == "no-cache"
    for name, value in UI_HEADERS.items():
        assert response.headers[name] == value
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_hashed_asset_is_immutable_and_missing_asset_is_404(ui_dir: Path) -> None:
    client = make_client(Settings(ui_dir=str(ui_dir)))
    asset = client.get("/assets/index-abc123.js")
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset.headers["x-content-type-options"] == "nosniff"
    assert client.get("/assets/nope.js").status_code == 404


def test_api_routes_are_not_shadowed(ui_dir: Path) -> None:
    client = make_client(Settings(ui_dir=str(ui_dir)))
    health = client.get("/health")
    assert health.status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert "/v1/recall" in spec.json()["paths"]
    assert "content-security-policy" not in spec.headers
    assert "content-security-policy" not in health.headers
    assert client.get("/docs").status_code == 200
    answer = client.post("/query", json={"question": "Where is Acme Corp based?"})
    assert answer.status_code == 200
    assert "content-security-policy" not in answer.headers


def test_auth_still_guards_the_api_while_the_ui_is_public(ui_dir: Path) -> None:
    client = make_client(Settings(ui_dir=str(ui_dir), require_auth=True))
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.post("/query", json={"question": "x"}).status_code == 401
    assert client.post("/v1/recall", json={"question": "x"}).status_code == 401


@pytest.mark.parametrize("ui_dir_setting", ["", "/no/such/dir"])
def test_without_a_build_the_app_is_unchanged(ui_dir_setting: str) -> None:
    client = make_client(Settings(ui_dir=ui_dir_setting))
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200


def test_offline_server_serves_the_ui(ui_dir: Path) -> None:
    client = TestClient(build_app(Settings(ui_dir=str(ui_dir))))
    assert client.get("/").text == PAGE
    assert client.get("/health").status_code == 200
