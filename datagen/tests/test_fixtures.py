"""The committed fixtures are exactly what the generator makes from their seeds."""

from __future__ import annotations

import json

import pytest
from kggen.fixtures import DEFAULT_DIR, FIXTURES, build, index


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_regenerates_identically_from_its_seed(name: str) -> None:
    files = build(name)
    committed = {p.name for p in (DEFAULT_DIR / name).iterdir()}
    assert committed == set(files)
    for filename, data in files.items():
        assert (DEFAULT_DIR / name / filename).read_bytes() == data, (
            f"{name}/{filename} drifted: run `uv run kggen fixtures write` and review the diff"
        )


def test_readme_documents_sizes_and_seeds() -> None:
    built = {name: build(name) for name in FIXTURES}
    assert (DEFAULT_DIR / "README.md").read_bytes() == index(built)
    text = (DEFAULT_DIR / "README.md").read_text()
    for name, files in built.items():
        assert f"| `{name}` | {json.loads(files['spec.json'])['n_entities']} | " in text


def test_fixtures_carry_questions_and_a_fixed_seed() -> None:
    for name in FIXTURES:
        files = build(name)
        assert json.loads(files["questions.json"])
        assert "seed" in json.loads(files["spec.json"])
