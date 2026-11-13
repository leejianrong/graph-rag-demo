"""docs/openapi.json must match the app, so the UI's generated client cannot go stale (GRA-46)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_openapi.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("gen_openapi", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_openapi_matches_the_app() -> None:
    gen = _load()
    assert gen.SPEC_PATH.read_text(encoding="utf-8") == gen.render(), "run `make openapi`"


def test_spec_is_deterministic_and_has_health() -> None:
    gen = _load()
    assert gen.render() == gen.render()
    assert "/health" in json.loads(gen.render())["paths"]


def test_ingest_and_job_routes_declare_response_schemas() -> None:
    gen = _load()
    paths = json.loads(gen.render())["paths"]
    ok = paths["/ingest"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok == {"$ref": "#/components/schemas/IngestAccepted"}
    job = paths["/jobs/{doc_id}"]["get"]["responses"]["200"]["content"]["application/json"]
    assert job["schema"] == {"$ref": "#/components/schemas/Job"}
    events = paths["/jobs/{doc_id}/events"]["get"]["responses"]["200"]["content"]
    assert events["text/event-stream"]["schema"]["$ref"] == "#/components/schemas/JobStreamEvent"


def test_spec_ignores_the_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gen = _load()
    baseline = gen.render()
    for name, value in {
        "REQUIRE_AUTH": "true",
        "UI_ENABLED": "false",
        "MAX_REMEMBER_CHARS": "5",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
    }.items():
        monkeypatch.setenv(name, value)
    assert gen.render() == baseline
