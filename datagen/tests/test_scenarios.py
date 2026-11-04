"""Scenarios give the expected post-change graph, and every arrival order reaches it."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from kggen.cli import main
from kggen.scenarios import ScenarioSpec, apply_events, generate_scenario
from kggen.spec import PREDICATE_SIGNATURES, WorldSpec, load_preset
from kggen.world import generate


@pytest.fixture(scope="module")
def world():
    return generate(WorldSpec(n_entities=300, avg_degree=4, seed=5, distractor_rate=0.1))


def test_expected_graph_is_the_current_facts_after_updates_and_deletes(world) -> None:
    sc = generate_scenario(world, ScenarioSpec(n_updates=6, n_deletes=4))
    base = {e.id: e for e in world.current_edges()}
    updates = [e for e in sc.events if e.kind == "update"]
    deletes = [e for e in sc.events if e.kind == "delete"]
    assert (len(updates), len(deletes)) == (6, 4)
    want = dict(base)
    for e in deletes:
        del want[e.fact.id]
    for e in updates:
        del want[e.replaces]
        want[e.fact.id] = e.fact
    assert {e.id: e for e in sc.expected} == want
    assert len(sc.expected) == len(base) - 4  # an update is one out, one in


def test_updates_change_the_object_to_a_type_valid_new_one(world) -> None:
    entities = {e.id: e for e in world.entities}
    sc = generate_scenario(world, ScenarioSpec(n_updates=10, n_deletes=0))
    old = {e.fact.id: e.fact for e in sc.events if e.kind == "add"}
    for e in (x for x in sc.events if x.kind == "update"):
        before = old[e.replaces]
        assert (e.fact.subject, e.fact.predicate) == (before.subject, before.predicate)
        assert e.fact.object != before.object
        pair = (entities[e.fact.subject].type, entities[e.fact.object].type)
        assert pair in PREDICATE_SIGNATURES[e.fact.predicate]
        assert not entities[e.fact.object].distractor


def test_every_arrival_order_is_causal_and_reaches_the_same_graph(world) -> None:
    sc = generate_scenario(world, ScenarioSpec(n_updates=8, n_deletes=8, n_orders=5))
    assert len(set(sc.orders)) > 1
    for order in sc.orders:
        assert sorted(order) == list(range(len(sc.events)))
        assert apply_events([sc.events[i] for i in order]) == sc.expected


def test_orders_and_scenarios_reproduce_by_seed(world) -> None:
    a = generate_scenario(world, ScenarioSpec(seed=1))
    assert generate_scenario(world, ScenarioSpec(seed=1)) == a
    other = generate_scenario(world, ScenarioSpec(seed=2))
    assert other.orders != a.orders and other.events != a.events


def test_fewer_facts_than_requested_changes_is_handled() -> None:
    tiny = generate(load_preset("tiny"))
    sc = generate_scenario(tiny, ScenarioSpec(n_updates=500, n_deletes=500))
    assert len(sc.expected) <= len(tiny.current_edges())


def test_scales_linearly_enough_for_a_large_world() -> None:
    big = generate(WorldSpec(n_entities=20000, avg_degree=3, seed=1))
    start = time.perf_counter()
    generate_scenario(big, ScenarioSpec(n_updates=2000, n_deletes=2000, n_orders=2))
    assert time.perf_counter() - start < 5


def test_cli_writes_the_scenario(tmp_path: Path) -> None:
    out = tmp_path / "s.json"
    argv = ["scenario", "generate", "--preset", "small", "--updates", "3", "--deletes", "2"]
    assert main([*argv, "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert set(data) == {"events", "expected", "orders"} and len(data["orders"]) == 3
