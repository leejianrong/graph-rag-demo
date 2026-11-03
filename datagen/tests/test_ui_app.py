"""The indah layer: the form fills from a preset and Generate reports the world's size."""

from __future__ import annotations

import pytest

pytest.importorskip("indah")  # the UI is an optional extra; the fast gate runs without it

from kggen_ui import model  # noqa: E402
from kggen_ui.app import build_session, build_ui  # noqa: E402


def test_the_form_starts_filled_from_the_default_preset() -> None:
    ui = build_ui()
    assert ui.current_values() == model.preset_values(model.DEFAULT_PRESET)
    assert ui.entities.value == "-"


def test_choosing_a_preset_refills_the_knobs() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    assert ui.current_values() == model.preset_values("tiny")


def test_generate_shows_the_counts_and_is_repeatable() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.generate()
    first = (ui.entities.value, ui.facts.value)
    assert first[0] == str(model.preset_values("tiny")["n_entities"])
    assert "seed" in ui.status.value
    ui.generate()
    assert (ui.entities.value, ui.facts.value) == first


def test_a_bad_value_shows_an_error_and_keeps_the_last_world() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.generate()
    before = (ui.entities.value, ui.world.value)
    ui.values["avg_degree"].set(-3)
    ui.generate()
    assert "Average degree" in ui.status.value
    assert (ui.entities.value, ui.world.value) == before


def test_two_viewers_do_not_share_state() -> None:
    a, b = build_ui(), build_ui()
    a.preset.set("tiny")
    assert b.preset.value == model.DEFAULT_PRESET


def test_a_session_builds() -> None:
    assert build_session() is not None


def test_generate_fills_the_stats_panel_and_says_everything_is_on_target() -> None:
    ui = build_ui()
    ui.preset.set("small")
    assert ui.checks.value == []
    ui.values["alias_rate"].set(0.3)
    ui.generate()
    assert {r[0] for r in ui.checks.value} >= {"n_entities", "alias_rate", "avg_degree"}
    assert all(r[3] == "ok" for r in ui.checks.value)


def test_the_form_has_one_input_per_knob() -> None:
    ui = build_ui()
    assert set(ui.values) == set(model.KNOB_NAMES)


# -- the graph view (needs an indah with Graph, 0.12 or newer) ---------------------------------

import indah  # noqa: E402

needs_graph = pytest.mark.skipif(not hasattr(indah, "Graph"), reason="this indah has no Graph")


@needs_graph
def test_generate_fills_the_graph_with_the_worlds_entities_and_current_facts() -> None:
    ui = build_ui()
    assert ui.graph.value == ([], [])
    ui.preset.set("tiny")
    ui.generate()
    nodes, edges = ui.graph.value
    assert len(nodes) == len(ui.world.value.entities)
    assert len(edges) == len(ui.world.value.current_edges())
    assert ui.graph_note.value == ""


@needs_graph
def test_selecting_a_node_shows_its_facts_and_regenerating_clears_the_selection() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.generate()
    entity = ui.graph.value[1][0]["source"]
    ui.selected.set(entity)
    assert entity in model.entity_summary(ui.index.value, ui.selected.value)
    assert model.fact_rows(ui.index.value, ui.selected.value)
    ui.values["seed"].set(ui.values["seed"].value + 1)
    ui.generate()  # a new world: the old id must not survive into it
    assert ui.selected.value is None


@needs_graph
def test_an_oversize_world_is_generated_but_not_drawn_and_says_why() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.values["n_entities"].set(8000)
    ui.values["avg_degree"].set(20)
    ui.generate()
    assert ui.graph.value == ([], [])
    assert "average degree" in ui.graph_note.value
    assert ui.entities.value == "8,000"  # the world still exists and its stats are shown
    assert ui.checks.value


@needs_graph
def test_the_focus_size_is_a_signal_the_graph_reads() -> None:
    ui = build_ui()
    assert ui.window.value == 300
    ui.window.set(50)
    assert ui.window.value == 50
