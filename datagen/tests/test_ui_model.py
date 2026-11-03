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


# -- the graph view and the detail panel -------------------------------------------------------


def rich_world():
    values = {
        **model.preset_values("small"),
        "alias_rate": 0.5,
        "distractor_rate": 0.1,
        "churn_rate": 0.2,
        "retraction_rate": 0.05,
    }
    return model.generate_world(values)


def test_graph_data_has_one_node_per_entity_and_one_edge_per_current_fact() -> None:
    world = rich_world()
    nodes, edges = model.graph_data(world)
    assert [n["id"] for n in nodes] == [e.id for e in world.entities]
    assert {n["label"] for n in nodes} == {e.name for e in world.entities}
    current = world.current_edges()
    assert [e["id"] for e in edges] == [f.id for f in current]
    assert [(e["source"], e["target"]) for e in edges] == [(f.subject, f.object) for f in current]


def test_nodes_are_coloured_by_type_with_distractors_apart() -> None:
    world = rich_world()
    nodes, _ = model.graph_data(world)
    by_id = {e.id: e for e in world.entities}
    for node in nodes:
        entity = by_id[node["id"]]
        assert node["group"] == (model.DISTRACTOR_GROUP if entity.distractor else entity.type)
    assert model.DISTRACTOR_GROUP in {n["group"] for n in nodes}


def test_a_small_world_fits_and_an_oversize_one_says_why() -> None:
    assert model.graph_limits_message(rich_world()) is None
    # 8,000 entities at average degree 20 is 80,000 facts, past the 60,000 the graph view holds
    huge = model.generate_world(
        {**model.preset_values("tiny"), "n_entities": 8000, "avg_degree": 20.0}
    )
    message = model.graph_limits_message(huge)
    assert message and "60,000" in message and "average degree" in message


def test_the_detail_lists_every_fact_that_touches_an_entity_with_its_dates() -> None:
    world = rich_world()
    index = model.index_world(world)
    changed = next(f for f in world.edges if f.valid_to is not None)
    rows = model.fact_rows(index, changed.subject)
    ended = [r for r in rows if r[3] == str(changed.valid_from) and r[4] == str(changed.valid_to)]
    assert ended, "a fact that ended is listed with its start and end year"
    assert any(r[4] == "now" for r in model.fact_rows(index, world.current_edges()[0].subject))
    touching = [f for f in world.edges if entity_touches(f, changed.subject)]
    assert len(rows) == len(touching)
    assert [r[3] for r in rows] == sorted(r[3] for r in rows)  # oldest first


def entity_touches(fact, entity_id: str) -> bool:
    return entity_id in (fact.subject, fact.object)


def test_the_detail_shows_direction_and_the_other_entity() -> None:
    world = rich_world()
    index = model.index_world(world)
    fact = world.edges[0]
    out = [r for r in model.fact_rows(index, fact.subject) if r[0] == fact.predicate]
    assert any(r[1] == "to" and f"({fact.object})" in r[2] for r in out)
    back = [r for r in model.fact_rows(index, fact.object) if r[0] == fact.predicate]
    assert any(r[1] == "from" and f"({fact.subject})" in r[2] for r in back)


def test_the_summary_names_the_entity_and_prompts_when_nothing_is_selected() -> None:
    world = rich_world()
    index = model.index_world(world)
    aliased = next(e for e in world.entities if e.aliases)
    text = model.entity_summary(index, aliased.id)
    assert aliased.name in text and aliased.id in text and aliased.aliases[0] in text
    distractor = next(e for e in world.entities if e.distractor)
    assert "distractor" in model.entity_summary(index, distractor.id)
    for nothing in (None, "no-such-id"):
        assert model.entity_summary(index, nothing) == "Click a node to see its facts."
        assert model.fact_rows(index, nothing) == []
    assert model.entity_summary(None, aliased.id) == "Click a node to see its facts."
