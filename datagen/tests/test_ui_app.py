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


# -- questions and gold paths ------------------------------------------------------------------


def graph_props(ui):
    """The props the Graph component sends right now, read from a live session snapshot."""
    snap = indah.Session(ui.root).snapshot()

    def find(node):
        if node.get("type") == "graph":
            return node["props"]
        for child in node.get("children", []) or []:
            found = find(child)
            if found is not None:
                return found

    return find(snap)


def test_generate_derives_questions_and_fills_the_table() -> None:
    ui = build_ui()
    ui.preset.set("small")
    assert ui.questions.value == ()
    ui.generate()
    assert len(ui.questions.value) == 20
    assert [r["id"] for r in ui.question_rows.value] == [q.id for q in ui.questions.value]
    assert ui.question_note.value == "20 questions."


def test_the_hop_mix_control_changes_the_derived_questions() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    for hop in (1, 2, 4):
        ui.question_values[f"hops_{hop}"].set(0)
    ui.derive()  # re-derive from the same world, with the new mix
    assert {q.hops for q in ui.questions.value} == {3}
    assert ui.world.value is not None


def test_a_bad_question_form_says_why_and_keeps_the_world() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    ui.question_values["n_questions"].set(0)
    ui.derive()
    assert "Questions must be between" in ui.question_note.value
    assert ui.questions.value == () and ui.entities.value == "150"


def test_deriving_before_a_world_exists_asks_for_one() -> None:
    ui = build_ui()
    ui.derive()
    assert ui.question_note.value == "Generate a world first."


def test_regenerating_clears_the_picked_question() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    ui.picked.set(ui.questions.value[0].id)
    ui.values["seed"].set(ui.values["seed"].value + 1)
    ui.generate()
    assert ui.picked.value is None and ui.picked_question() is None


@needs_graph
def test_picking_a_question_highlights_its_gold_path_and_centres_on_its_start() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    q = next(q for q in ui.questions.value if q.hops == 3)
    assert graph_props(ui)["highlight"] == {"nodes": [], "edges": []}
    ui.picked.set(q.id)
    nodes, edges = model.gold_path(ui.index.value, q)
    props = graph_props(ui)
    assert props["highlight"] == {"nodes": nodes, "edges": edges}
    assert props["focus"] == q.start
    ui.picked.set(None)
    assert graph_props(ui)["highlight"] == {"nodes": [], "edges": []}


@needs_graph
def test_an_oversize_world_lists_questions_but_does_not_highlight() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    # Sparse, so unique-answer questions exist, but past the graph view's 20,000-node limit.
    for knob, value in {"n_entities": 20_000, "avg_degree": 2, "distractor_rate": 0.2}.items():
        ui.values[knob].set(value)
    ui.values["n_communities"].set(8)
    ui.generate()
    assert ui.graph_note.value and ui.questions.value
    ui.picked.set(ui.questions.value[0].id)
    assert ui.highlight() is None and ui.focus() is None
    assert graph_props(ui)["highlight"] == {"nodes": [], "edges": []}
