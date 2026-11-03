"""Ambiguity and churn knobs: each must change the world by the amount the spec asks for."""

from __future__ import annotations

from collections import Counter

import pytest
from kgbench.spec import WorldSpec
from kgbench.world import WorldGenerationError, generate

from graph_rag.predicates import Predicate

N = 400


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": N, "avg_degree": 4, "seed": 5}
    return WorldSpec.model_validate({**base, **overrides})


def test_knobs_off_leave_the_graph_unchanged() -> None:
    world = generate(make())
    assert all(not e.aliases and not e.distractor for e in world.entities)
    assert all(e.valid_from == 2000 and e.valid_to is None for e in world.edges)
    assert len({e.name for e in world.entities}) == N


def test_enabling_a_knob_keeps_the_base_entities_and_types() -> None:
    plain = generate(make())
    noisy = generate(make(alias_rate=0.5, distractor_rate=0.1))
    assert [(e.id, e.type) for e in plain.entities] == [(e.id, e.type) for e in noisy.entities[:N]]


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.6])
def test_alias_rate_sets_the_share_of_aliased_entities(rate: float) -> None:
    world = generate(make(alias_rate=rate))
    aliased = [e for e in world.entities if e.aliases]
    assert len(aliased) == round(rate * N)


def test_aliases_per_entity_is_an_upper_bound_and_aliases_are_not_names() -> None:
    world = generate(make(alias_rate=1.0, aliases_per_entity=2))
    names = {e.name for e in world.entities}
    assert max(len(e.aliases) for e in world.entities) == 2
    assert all(a not in names for e in world.entities for a in e.aliases)
    assert all(len(set(e.aliases)) == len(e.aliases) for e in world.entities)


@pytest.mark.parametrize("rate", [0.05, 0.2])
def test_homonym_rate_sets_the_number_of_shared_names(rate: float) -> None:
    world = generate(make(homonym_rate=rate))
    assert N - len({e.name for e in world.entities}) == round(rate * N)


def test_homonym_same_type_controls_whether_twins_share_a_type() -> None:
    def same_type_share(share: float) -> float:
        world = generate(make(homonym_rate=0.2, homonym_same_type=share))
        by_name: dict[str, set[str]] = {}
        for e in world.entities:
            by_name.setdefault(e.name, set()).add(e.type)
        shared = [
            types
            for name, types in by_name.items()
            if sum(e.name == name for e in world.entities) > 1
        ]
        return sum(len(t) == 1 for t in shared) / len(shared)

    assert same_type_share(1.0) > 0.95
    assert same_type_share(0.0) < 0.05


def test_distractors_are_extra_look_alike_entities_linked_only_to_each_other() -> None:
    world = generate(make(distractor_rate=0.1))
    distractors = [e for e in world.entities if e.distractor]
    assert len(distractors) == round(0.1 * N)
    assert len(world.entities) == N + len(distractors)
    real_names = {e.name for e in world.entities if not e.distractor}
    real_first = {n.split()[0] for n in real_names}
    assert all(d.name not in real_names for d in distractors)
    assert all(d.name.split()[0] in real_first or len(d.name) > 3 for d in distractors)
    flagged = {e.id for e in distractors}
    for edge in world.edges:
        assert (edge.subject in flagged) == (edge.object in flagged)


@pytest.mark.parametrize("rate", [0.1, 0.4])
def test_noise_rate_sets_the_share_of_vague_edges(rate: float) -> None:
    world = generate(make(noise_rate=rate))
    vague = sum(e.predicate == Predicate.RELATED_TO for e in world.edges)
    assert vague == round(rate * len(world.edges))
    assert len(world.edges) == round(N * 4 / 2)


def test_churn_replaces_facts_with_dated_successors() -> None:
    spec = make(churn_rate=0.2)
    world = generate(spec)
    ended = [e for e in world.edges if e.valid_to is not None]
    assert len(ended) == round(0.2 * round(N * 4 / 2))
    assert all(spec.start_year < e.valid_to <= spec.end_year for e in ended)
    successors = Counter(
        (e.subject, e.predicate, e.valid_from) for e in world.edges if e.valid_from > 2000
    )
    assert successors == Counter((e.subject, e.predicate, e.valid_to) for e in ended)


def test_churn_final_state_and_point_in_time_view() -> None:
    world = generate(make(churn_rate=0.2, retraction_rate=0.1))
    total = round(N * 4 / 2)
    n_replace, n_retract = round(0.2 * total), round(0.1 * total)
    assert len(world.edges) == total + n_replace
    assert len(world.current_edges()) == total - n_retract
    assert world.edges_as_of(world.spec.start_year) == tuple(
        e for e in world.edges if e.valid_from == world.spec.start_year
    )
    assert world.edges_as_of(world.spec.end_year) == world.current_edges()
    # A replaced fact is true before its change year and gone after it.
    old = next(e for e in world.edges if e.valid_to is not None)
    assert old in world.edges_as_of(old.valid_to - 1)
    assert old not in world.edges_as_of(old.valid_to)


def test_a_retraction_leaves_no_successor() -> None:
    world = generate(make(retraction_rate=0.1))
    assert len(world.edges) == round(N * 4 / 2)
    ended = Counter(e.valid_to is not None for e in world.edges)
    assert ended[True] == round(0.1 * len(world.edges))


def test_all_knobs_together_are_deterministic() -> None:
    spec = make(
        alias_rate=0.4,
        homonym_rate=0.1,
        distractor_rate=0.1,
        noise_rate=0.1,
        churn_rate=0.1,
        retraction_rate=0.05,
    )
    assert generate(spec).to_json() == generate(spec).to_json()


def test_churn_that_cannot_be_placed_is_an_error() -> None:
    spec = make(n_entities=2, avg_degree=1, type_mix={"PERSON": 1, "ORG": 1}, churn_rate=1.0)
    with pytest.raises(WorldGenerationError, match="churn"):
        generate(spec)


@pytest.mark.parametrize(
    "bad",
    [{"alias_rate": 1.5}, {"end_year": 2000}, {"churn_rate": 0.7, "retraction_rate": 0.5}],
)
def test_invalid_ambiguity_and_churn_values_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make(**bad)
