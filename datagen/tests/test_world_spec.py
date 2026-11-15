"""WorldSpec validation and the shipped presets."""

from __future__ import annotations

from pathlib import Path

import pytest
from kggen.spec import (
    DIFFICULTY,
    DIFFICULTY_NAMES,
    PRESET_NAMES,
    WorldSpec,
    load_preset,
    load_spec,
    with_difficulty,
)
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


# --- difficulty (GRA-183) ----------------------------------------------------------------------


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_clean_is_the_default_and_changes_nothing(name: str) -> None:
    """Every existing preset keeps generating exactly what it did."""
    assert load_preset(name) == load_preset(name, "clean")
    assert load_preset(name).model_dump() == load_preset(name, "clean").model_dump()
    assert load_preset(name).alias_rate == load_preset(name).homonym_rate == 0


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_a_difficulty_changes_only_ambiguity_never_the_shape(name: str) -> None:
    clean, hard = load_preset(name), load_preset(name, "hard")
    for field in ("n_entities", "avg_degree", "degree_skew", "n_communities", "seed"):
        assert getattr(hard, field) == getattr(clean, field)
    assert hard.alias_rate > 0 and hard.homonym_rate > 0 and hard.distractor_rate > 0


def test_hard_is_what_the_hard_small_example_was_made_with() -> None:
    example = load_spec(Path(__file__).parents[1] / "examples" / "hard-small" / "spec.yaml")
    for knob in ("alias_rate", "aliases_per_entity", "homonym_rate", "distractor_rate"):
        assert DIFFICULTY["hard"][knob] == getattr(example, knob)


def test_an_unknown_difficulty_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown difficulty"):
        with_difficulty(make(), "impossible")
    assert set(DIFFICULTY_NAMES) == {"clean", "aliases", "hard"}
