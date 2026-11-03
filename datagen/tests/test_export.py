"""The dataset export: complete, deterministic, and reproducible from the specs it carries."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from kggen.cli import main
from kggen.export import COMMAND, Specs, export_archive, export_files, reproduce, zip_bytes
from kggen.questions import QuestionSpec
from kggen.render import RenderSpec
from kggen.spec import WorldSpec
from kggen.world import generate

WORLD = WorldSpec.model_validate(
    {
        "n_entities": 250,
        "avg_degree": 4,
        "seed": 6,
        "alias_rate": 0.5,
        "distractor_rate": 0.1,
        "churn_rate": 0.15,
    }
)
QUESTIONS = QuestionSpec(n_questions=15, hop_mix={1: 1, 3: 2})
RENDER = RenderSpec(alias_use_rate=0.7, max_facts_per_doc=4)


@pytest.fixture(scope="module")
def files():
    return export_files(generate(WORLD), QUESTIONS, RENDER)


def test_the_archive_holds_the_data_the_stats_the_specs_and_a_readme(files) -> None:
    assert set(files) == {
        "README.md",
        "documents.json",
        "questions.json",
        "questions_spec.json",
        "render_spec.json",
        "spec.json",
        "world.json",
        "world.stats.json",
    }
    for name, data in files.items():
        if name.endswith(".json"):
            json.loads(data)  # every data file is valid JSON


def test_the_files_are_the_same_ones_the_cli_commands_write(files) -> None:
    world = generate(WORLD)
    assert files["world.json"] == world.to_json().encode()
    questions = json.loads(files["questions.json"])
    assert len(questions) == 15 and {q["hops"] for q in questions} == {1, 3}
    documents = json.loads(files["documents.json"])
    assert documents and all(d["sentences"] for d in documents)
    assert json.loads(files["world.stats.json"])["stats"]["entities"] == 250


def test_the_specs_in_the_archive_describe_how_it_was_made(files) -> None:
    assert WorldSpec.model_validate_json(files["spec.json"]) == WORLD
    assert QuestionSpec.model_validate_json(files["questions_spec.json"]) == QUESTIONS
    assert RenderSpec.model_validate_json(files["render_spec.json"]) == RENDER


def test_the_zip_is_deterministic_and_lists_its_files_in_order(files) -> None:
    first, second = zip_bytes(files), zip_bytes(dict(reversed(files.items())))
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == sorted(files)
        assert {i.date_time for i in archive.infolist()} == {(2020, 1, 1, 0, 0, 0)}
        assert archive.testzip() is None
        assert archive.read("world.json") == files["world.json"]


def test_remaking_the_dataset_from_its_own_specs_gives_the_same_bytes(files) -> None:
    again = reproduce(
        Specs(
            world=WorldSpec.model_validate_json(files["spec.json"]),
            questions=QuestionSpec.model_validate_json(files["questions_spec.json"]),
            render=RenderSpec.model_validate_json(files["render_spec.json"]),
        )
    )
    assert again == files


def test_the_command_in_the_readme_remakes_the_archive_byte_for_byte(tmp_path, files) -> None:
    original = tmp_path / "original.zip"
    original.write_bytes(zip_bytes(files))
    for name in ("spec.json", "questions_spec.json", "render_spec.json"):
        (tmp_path / name).write_bytes(files[name])
    out = tmp_path / "dataset.zip"
    argv = COMMAND.split()[1:]  # drop the program name
    argv = [str(tmp_path / a) if a.endswith(".json") else a for a in argv]
    argv[argv.index("dataset.zip")] = str(out)
    assert main(argv) == 0
    assert out.read_bytes() == original.read_bytes()
    assert COMMAND in files["README.md"].decode()


def test_a_world_that_cannot_supply_the_questions_still_exports_and_says_why() -> None:
    tiny = WorldSpec.model_validate({"n_entities": 6, "avg_degree": 1, "seed": 1})
    impossible = QuestionSpec(n_questions=500, hop_mix={4: 1})
    files = export_files(generate(tiny), impossible, RENDER)
    assert "questions.json" not in files
    assert "questions.json is not included" in files["README.md"].decode()
    assert "world.json" in files and "documents.json" in files


def test_a_different_seed_gives_a_different_archive(files) -> None:
    other = export_archive(generate(WORLD.with_overrides(seed=7)), QUESTIONS, RENDER)
    assert other != zip_bytes(files)


def test_export_without_optional_specs_uses_the_defaults(tmp_path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(WORLD.model_dump_json())
    out = tmp_path / "out.zip"
    assert main(["export", "--spec", str(spec), "--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        assert (
            QuestionSpec.model_validate_json(archive.read("questions_spec.json")) == QuestionSpec()
        )
