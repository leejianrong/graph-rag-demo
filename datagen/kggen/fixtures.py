"""Committed seeded fixture sets: fixed inputs for reproducible runs.

Each fixture is a preset world with its questions and template documents, written as the files
``kggen export`` produces. ``datagen/tests/test_fixtures.py`` regenerates them and compares byte
for byte, so a change to the generator that alters a fixture shows up as a diff to review.
"""

from __future__ import annotations

import json
from pathlib import Path

from kggen.export import export_files
from kggen.questions import QuestionSpec
from kggen.render import RenderSpec
from kggen.spec import load_preset
from kggen.world import generate

#: name -> preset. The seed is the preset's (0) and is written into each fixture's ``spec.json``.
FIXTURES: dict[str, str] = {"small": "small", "medium": "medium"}
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def build(name: str) -> dict[str, bytes]:
    world = generate(load_preset(FIXTURES[name]))
    return export_files(world, QuestionSpec(), RenderSpec())


def index(built: dict[str, dict[str, bytes]]) -> bytes:
    """``README.md`` for the fixtures directory: what each set is, its seed and its size."""
    rows = []
    for name, files in built.items():
        spec = json.loads(files["spec.json"])
        facts = json.loads(files["world.stats.json"])["stats"]["facts"]
        docs = json.loads(files["documents.json"])
        questions = len(json.loads(files["questions.json"]))
        kb = round(sum(map(len, files.values())) / 1024)
        rows.append(
            f"| `{name}` | {spec['n_entities']} | {spec['seed']} | {facts} | "
            f"{len(docs)} | {questions} | {kb} KB |"
        )
    return (
        "# Fixtures\n\nSeeded datasets, committed so runs start from fixed inputs. Regenerate with "
        "`uv run kggen fixtures write`; a test fails if they drift from the generator.\n\n"
        "| Fixture | Entities | Seed | Facts | Documents | Questions | Size |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Each folder is a `kggen export` of that preset (see its own `README.md`). Documents are "
        "template text. Update/delete/reorder scenarios regenerate from the world with "
        "`kggen scenario generate`. LLM-written documents need a cache, see `examples/`.\n"
    ).encode()


def write(directory: Path = DEFAULT_DIR) -> None:
    built = {name: build(name) for name in FIXTURES}
    for name, files in built.items():
        for filename, data in files.items():
            path = directory / name / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (directory / "README.md").write_bytes(index(built))
