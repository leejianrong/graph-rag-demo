"""Determinism and structural validity of generated worlds."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from kgbench.spec import PREDICATE_SIGNATURES, PRESET_NAMES, WorldSpec, load_preset
from kgbench.world import WorldGenerationError, apportion, generate

from graph_rag.predicates import CLOSED_PREDICATES

FAST_PRESETS = ("tiny", "small", "medium")


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": 300, "avg_degree": 4, "seed": 1}
    return WorldSpec.model_validate({**base, **overrides})


def test_same_spec_and_seed_give_byte_identical_output() -> None:
    assert generate(make()).to_json() == generate(make()).to_json()


def test_a_different_seed_gives_a_different_world() -> None:
    assert generate(make(seed=1)).to_json() != generate(make(seed=2)).to_json()


def test_output_is_canonical_json_with_the_spec_embedded() -> None:
    text = generate(make()).to_json()
    data = json.loads(text)
    assert data["spec"]["seed"] == 1
    assert text.endswith("\n")
    assert list(data) == sorted(data)


@pytest.mark.parametrize("name", FAST_PRESETS)
def test_presets_generate_the_requested_size(name: str) -> None:
    spec = load_preset(name)
    world = generate(spec)
    assert len(world.entities) == spec.n_entities
    assert len(world.edges) == round(spec.n_entities * spec.avg_degree / 2)


def test_all_presets_are_defined_for_generation() -> None:
    assert set(PRESET_NAMES) == {"tiny", "small", "medium", "large", "stress"}


def test_entity_count_and_unique_names_and_ids() -> None:
    world = generate(make(n_entities=500))
    assert len(world.entities) == 500
    assert len({e.name for e in world.entities}) == 500
    assert len({e.id for e in world.entities}) == 500


def test_edge_count_follows_avg_degree() -> None:
    for degree in (2, 4, 8):
        world = generate(make(avg_degree=degree))
        assert len(world.edges) == round(300 * degree / 2)


def test_edges_are_valid_typed_unique_and_not_self_loops() -> None:
    world = generate(make(n_entities=400, cycle_rate=0.2))
    kind = {e.id: e.type for e in world.entities}
    triples = [(e.subject, e.predicate, e.object) for e in world.edges]
    assert len(triples) == len(set(triples))
    for subject, predicate, obj in triples:
        assert subject != obj
        assert predicate in CLOSED_PREDICATES
        assert (kind[subject], kind[obj]) in dict.fromkeys(PREDICATE_SIGNATURES[predicate])


def test_type_mix_is_followed_exactly_by_largest_remainder() -> None:
    assert apportion(100, {"A": 1, "B": 1, "C": 1}) == {"A": 34, "B": 33, "C": 33}
    world = generate(make(n_entities=200, type_mix={"PERSON": 3.0, "ORG": 1.0}))
    assert Counter(e.type for e in world.entities) == {"PERSON": 150, "ORG": 50}


def test_zero_weight_predicates_never_appear() -> None:
    world = generate(
        make(
            type_mix={"PERSON": 1.0, "ORG": 1.0},
            predicate_mix={"WORKS_FOR": 1.0, "FOUNDED": 1.0, "OWNS": 0.0},
        )
    )
    assert {e.predicate for e in world.edges} <= {"WORKS_FOR", "FOUNDED"}


def test_predicate_mix_shifts_the_realised_mix() -> None:
    world = generate(
        make(
            n_entities=600,
            type_mix={"PERSON": 1.0, "ORG": 1.0},
            predicate_mix={"WORKS_FOR": 9.0, "FOUNDED": 1.0},
        )
    )
    counts = Counter(e.predicate for e in world.edges)
    assert counts["WORKS_FOR"] > 5 * counts["FOUNDED"]


def test_edges_are_sorted_for_stable_output() -> None:
    edges = [(e.subject, e.predicate, e.object) for e in generate(make()).edges]
    assert edges == sorted(edges)


def test_overdense_spec_fails_clearly() -> None:
    spec = make(
        n_entities=4,
        avg_degree=200,
        type_mix={"PERSON": 1.0, "ORG": 1.0},
        predicate_mix={"WORKS_FOR": 1.0},
    )
    with pytest.raises(WorldGenerationError, match="could only place"):
        generate(spec)
