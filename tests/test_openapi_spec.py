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
