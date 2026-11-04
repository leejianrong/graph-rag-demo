"""WorldSpec validation and the shipped presets."""

from __future__ import annotations

import pytest
from kggen.spec import PRESET_NAMES, WorldSpec, load_preset, load_spec
from pydantic import ValidationError


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": 100, "avg_degree": 4}
    return WorldSpec.model_validate({**base, **overrides})


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_loads_and_validates(name: str) -> None:
    assert load_preset(name).n_entities > 0


def test_presets_grow_in_size() -> None:
    sizes = [load_preset(name).n_entities for name in PRESET_NAMES]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        load_preset("gigantic")


def test_spec_loads_from_yaml(tmp_path) -> None:
    path = tmp_path / "w.yaml"
    path.write_text("n_entities: 50\navg_degree: 3\nseed: 9\n")
    spec = load_spec(path)
    assert (spec.n_entities, spec.avg_degree, spec.seed) == (50, 3, 9)


@pytest.mark.parametrize(
    "bad",
    [
        {"n_entities": 1},
        {"avg_degree": 0},
        {"degree_skew": -1},
        {"community_mixing": 1.5},
        {"cycle_rate": -0.1},
        {"n_communities": 500},
        {"mystery_knob": 1},
        {"type_mix": {"PERSON": 1.0, "SPACESHIP": 1.0}},
        {"type_mix": {"PERSON": -1.0}},
        {"predicate_mix": {"WORKS_FOR": 0.0}},
    ],
)
def test_invalid_specs_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make(**bad)


def test_type_that_no_relation_can_reach_is_rejected() -> None:
    # With only WORKS_FOR selected, nothing can reach a LAW, so it would be an isolated node.
    with pytest.raises(ValidationError, match="isolated"):
        make(type_mix={"PERSON": 1.0, "ORG": 1.0, "LAW": 1.0}, predicate_mix={"WORKS_FOR": 1.0})
    make(type_mix={"PERSON": 1.0, "ORG": 1.0, "LAW": 1.0})  # GOVERNED_BY reaches it (GRA-94)


def test_with_overrides_revalidates() -> None:
    assert make().with_overrides(seed=5).seed == 5
    with pytest.raises(ValidationError):
        make().with_overrides(avg_degree=-1)
