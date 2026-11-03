"""The UI's logic, tested with no indah and no browser."""

from __future__ import annotations

import pytest
from kggen.spec import PRESET_NAMES, load_preset
from kggen_ui import model


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_fills_every_knob(name: str) -> None:
    values = model.preset_values(name)
    assert set(values) == set(model.KNOB_NAMES)
    assert values["n_entities"] == load_preset(name).n_entities


def test_values_round_trip_to_the_same_spec() -> None:
    spec = load_preset("small")
    assert model.spec_from_values(model.values_from_spec(spec), base=spec) == spec


def test_same_values_and_seed_give_the_same_world() -> None:
    values = model.preset_values("tiny")
    a = model.summarize(model.generate_world(values))
    b = model.summarize(model.generate_world(dict(values)))
    assert a == b
    other = model.summarize(model.generate_world({**values, "seed": values["seed"] + 1}))
    assert (other.entities, other.facts) == (a.entities, a.facts)  # same size, different world


def test_summary_counts_match_the_world() -> None:
    values = model.preset_values("tiny")
    world = model.generate_world(values)
    summary = model.summarize(world)
    assert summary.entities == len(world.entities) == values["n_entities"]
    assert summary.facts == len(world.edges)


def test_form_floats_for_whole_number_knobs_are_accepted() -> None:
    spec = model.spec_from_values({**model.preset_values("tiny"), "n_entities": 40.0, "seed": "7"})
    assert (spec.n_entities, spec.seed) == (40, 7)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_entities": 1}, "Entities must be between"),
        ({"n_entities": model.MAX_ENTITIES + 1}, "Entities must be between"),
        ({"avg_degree": "lots"}, "Average degree must be a number"),
        ({"community_mixing": 1.5}, "Community mixing must be between"),
        ({"seed": ""}, "Seed needs a value"),
        ({"n_entities": 10, "n_communities": 50}, "n_communities cannot exceed"),
    ],
)
def test_bad_values_give_a_readable_error(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model.spec_from_values({**model.preset_values("tiny"), **change})


def test_a_world_that_cannot_be_placed_is_a_readable_error() -> None:
    values = {**model.preset_values("tiny"), "n_entities": 4, "avg_degree": 20}
    with pytest.raises(ValueError, match="edges"):
        model.generate_world(values)


def test_every_knob_belongs_to_a_known_group_and_the_form_covers_the_spec() -> None:
    assert {k.group for k in model.KNOBS} <= set(model.GROUPS)
    assert {"alias_rate", "homonym_rate", "distractor_rate", "noise_rate"} <= set(model.KNOB_NAMES)
    assert {"churn_rate", "retraction_rate"} <= set(model.KNOB_NAMES)


def test_check_rows_compare_what_was_asked_with_what_was_realized() -> None:
    values = {**model.preset_values("small"), "alias_rate": 0.4, "churn_rate": 0.1}
    rows = model.check_rows(model.generate_world(values))
    by_knob = {r[0]: r for r in rows}
    assert by_knob["alias_rate"][1] == "0.4"
    assert by_knob["n_entities"][1:3] == ["150", "150"]
    assert all(r[3] == "ok" for r in rows)
