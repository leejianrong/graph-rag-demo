"""The stats report measures the world, and every knob lands within tolerance of its spec."""

from __future__ import annotations

import json

import pytest
from kggen.cli import main
from kggen.spec import WorldSpec, load_preset
from kggen.stats import check_against_spec, report, world_stats
from kggen.world import generate

FULL = {
    "n_entities": 600,
    "avg_degree": 4,
    "seed": 11,
    "degree_skew": 0.8,
    "n_communities": 5,
    "community_mixing": 0.2,
    "cycle_rate": 0.05,
    "alias_rate": 0.4,
    "homonym_rate": 0.08,
    "distractor_rate": 0.1,
    "noise_rate": 0.1,
    "churn_rate": 0.15,
    "retraction_rate": 0.05,
}


def make(**overrides: object) -> WorldSpec:
    return WorldSpec.model_validate({**FULL, **overrides})


def test_every_knob_is_within_tolerance_of_its_spec() -> None:
    world = generate(make())
    checks = check_against_spec(world)
    assert {c.knob for c in checks} >= {
        "n_entities",
        "avg_degree",
        "type_mix",
        "community_mixing",
        "alias_rate",
        "homonym_rate",
        "distractor_rate",
        "noise_rate",
        "churn_rate",
        "retraction_rate",
    }
    assert [c for c in checks if not c.ok] == []


@pytest.mark.parametrize("name", ["tiny", "small", "medium"])
def test_presets_are_within_tolerance(name: str) -> None:
    world = generate(load_preset(name))
    assert [c for c in check_against_spec(world) if not c.ok] == []


def test_a_check_fails_when_the_world_does_not_match_its_spec() -> None:
    world = generate(make(alias_rate=0.0))
    lying = type(world)(spec=make(alias_rate=0.5), entities=world.entities, edges=world.edges)
    failed = [c.knob for c in check_against_spec(lying) if not c.ok]
    assert "alias_rate" in failed


def test_stats_describe_the_graph() -> None:
    world = generate(
        make(
            distractor_rate=0.0,
            homonym_rate=0.0,
            churn_rate=0.0,
            retraction_rate=0.0,
            noise_rate=0.0,
        )
    )
    stats = world_stats(world)
    assert stats["entities"] == 600
    assert sum(stats["degree"]["histogram"].values()) == 600
    assert sum(stats["communities"].values()) == 600
    assert sum(stats["components"]["sizes"]) <= 600
    assert stats["components"]["largest"] == stats["components"]["sizes"][0]
    assert stats["degree"]["max"] == max(stats["degree"]["histogram"])
    assert stats["path_lengths"] and min(stats["path_lengths"]) == 1
    assert stats["ambiguity"]["shared_names"] == 0
    assert stats["churn"] == {"replaced": 0, "retracted": 0}


def test_skew_shows_up_in_the_degree_stats() -> None:
    flat = world_stats(generate(make(degree_skew=0.0)))["degree"]["max"]
    skewed = world_stats(generate(make(degree_skew=1.5)))["degree"]["max"]
    assert skewed > 2 * flat


def test_report_is_json_serialisable_and_deterministic() -> None:
    world = generate(make())
    assert json.dumps(report(world), sort_keys=True) == json.dumps(report(world), sort_keys=True)


def test_cli_writes_a_stats_report_beside_the_world(tmp_path) -> None:
    out = tmp_path / "world.json"
    assert main(["world", "generate", "--preset", "tiny", "--out", str(out)]) == 0
    data = json.loads((tmp_path / "world.stats.json").read_text())
    assert data["stats"]["entities"] == load_preset("tiny").n_entities
    assert all(c["ok"] for c in data["checks"])


def test_cli_stats_command_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["world", "stats", "--preset", "tiny"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["stats"]["entities"] == load_preset("tiny").n_entities
    )


def test_community_mixing_is_not_checked_when_communities_are_tiny() -> None:
    world = generate(load_preset("tiny"))
    assert "community_mixing" not in {c.knob for c in check_against_spec(world)}
