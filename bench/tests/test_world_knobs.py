"""Each shape knob must measurably change the realised graph."""

from __future__ import annotations

from collections import Counter

from kgbench.spec import WorldSpec
from kgbench.world import World, generate


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {"n_entities": 800, "avg_degree": 4, "seed": 3}
    return WorldSpec.model_validate({**base, **overrides})


def max_degree(world: World) -> int:
    degree: Counter[str] = Counter()
    for edge in world.edges:
        degree[edge.subject] += 1
        degree[edge.object] += 1
    return max(degree.values())


def crossing_fraction(world: World) -> float:
    community = {e.id: e.community for e in world.entities}
    crossing = sum(community[e.subject] != community[e.object] for e in world.edges)
    return crossing / len(world.edges)


def triangle_count(world: World) -> int:
    present = {(e.subject, e.object) for e in world.edges}
    onward: dict[str, list[str]] = {}
    for subject, obj in present:
        onward.setdefault(subject, []).append(obj)
    return sum((c, a) in present for a, b in present for c in onward.get(b, []) if c != a)


def test_degree_skew_creates_hubs() -> None:
    flat = max_degree(generate(make(degree_skew=0.0)))
    skewed = max_degree(generate(make(degree_skew=1.5)))
    assert skewed > 2 * flat


def test_community_mixing_controls_cross_community_edges() -> None:
    sealed = crossing_fraction(generate(make(n_communities=5, community_mixing=0.0)))
    open_ = crossing_fraction(generate(make(n_communities=5, community_mixing=1.0)))
    assert sealed < 0.05
    assert open_ > 0.7


def test_more_communities_with_no_mixing_still_seals_them() -> None:
    assert crossing_fraction(generate(make(n_communities=20, community_mixing=0.0))) < 0.1


def test_cycle_rate_adds_cycles() -> None:
    none = triangle_count(generate(make(cycle_rate=0.0)))
    many = triangle_count(generate(make(cycle_rate=0.4)))
    assert many > none + 50


def test_a_single_community_has_no_crossings() -> None:
    assert crossing_fraction(generate(make(n_communities=1))) == 0.0
