"""The year view (GRA-192): a world drawn as of a year, what each year changed, dated documents."""

from __future__ import annotations

import pytest

pytest.importorskip("indah")  # the UI is an optional extra; the fast gate runs without it

from kggen_ui import model  # noqa: E402
from kggen_ui.app import build_ui  # noqa: E402


def world(evolution: str = "heavy"):  # type: ignore[no-untyped-def]
    return model.generate_world(
        model.preset_values("small", "clean", evolution), name_style="invented"
    )


def test_a_world_that_never_changes_has_no_year_view() -> None:
    assert model.time_graph(world("none")) is None
    assert not model.evolves(world("none"))


def test_every_year_draws_the_same_entities_and_facts() -> None:
    """The browser lays the graph out once, so the set of nodes and edges must not move."""
    tg = model.time_graph(world())
    assert tg is not None and tg.first < tg.last
    ids = [
        ([n["id"] for n in tg.nodes_edges(y)[0]], [e["id"] for e in tg.nodes_edges(y)[1]])
        for y in tg.years
    ]
    assert all(i == ids[0] for i in ids)


def test_the_group_is_the_type_in_every_year_so_the_layout_does_not_move() -> None:
    tg = model.time_graph(world())
    assert tg is not None
    groups = [[n["group"] for n in tg.nodes_edges(y)[0]] for y in tg.years]
    assert all(g == groups[0] for g in groups)
    assert not {model.BORN, model.GONE, model.ENDED, model.RENAMED} & set(groups[0])


def test_the_first_year_is_the_start_with_nothing_marked_as_new() -> None:
    tg = model.time_graph(world())
    assert tg is not None
    nodes, edges = tg.nodes_edges(tg.first)
    marked = {model.STATE_COLOURS[k] for k in (model.BORN, model.RENAMED, model.ENDED)}
    assert not marked & {n.get("color") for n in nodes}
    assert model.STATE_COLOURS[model.BORN] not in {e["color"] for e in edges}
    rows = tg.events(tg.first)
    assert rows[0][0] == "" and "The start" in rows[0][1]


def test_a_fact_is_new_in_its_first_year_and_ended_in_the_year_it_stops() -> None:
    w = world()
    tg = model.time_graph(w)
    assert tg is not None
    began = next(f for f in w.edges if f.valid_from > tg.first)
    colour = {
        y: next(e["color"] for e in tg.nodes_edges(y)[1] if e["id"] == began.id)
        for y in (began.valid_from - 1, began.valid_from, began.valid_from + 1)
    }
    assert colour[began.valid_from - 1] == model.STATE_COLOURS[model.GONE]
    assert colour[began.valid_from] == model.STATE_COLOURS[model.BORN]
    assert colour[began.valid_from + 1] in (model.EDGE_ACTIVE, model.STATE_COLOURS[model.ENDED])
    ended = next(f for f in w.edges if f.valid_to is not None)
    at = {
        y: next(e["color"] for e in tg.nodes_edges(y)[1] if e["id"] == ended.id)
        for y in (ended.valid_to - 1, ended.valid_to, ended.valid_to + 1)
    }
    assert at[ended.valid_to] == model.STATE_COLOURS[model.ENDED]
    assert at[ended.valid_to + 1] == model.STATE_COLOURS[model.GONE]
    assert at[ended.valid_to - 1] != model.STATE_COLOURS[model.GONE] or ended.valid_to - 1 < (
        ended.valid_from
    )


def test_a_renamed_entity_wears_its_old_name_until_the_year_it_changes() -> None:
    w = world()
    tg = model.time_graph(w)
    assert tg is not None
    renamed = next(e for e in w.entities if e.renamed is not None and e.renamed > tg.first)
    label = {
        y: next(n["label"] for n in tg.nodes_edges(y)[0] if n["id"] == renamed.id)
        for y in (renamed.renamed - 1, renamed.renamed, renamed.renamed + 1)
    }
    assert label[renamed.renamed - 1] == renamed.former_name
    assert label[renamed.renamed] == f"{renamed.name} (was {renamed.former_name})"
    assert label[renamed.renamed + 1] == renamed.name
    rows = tg.events(renamed.renamed)
    assert ["~", f"{renamed.former_name} is now {renamed.name}"] in rows


def test_births_and_closures_are_listed_and_closed_entities_turn_pale() -> None:
    w = world()
    tg = model.time_graph(w)
    assert tg is not None
    born = next(e for e in w.entities if e.born is not None and e.born > tg.first)
    assert ["+", f"{born.name} appears ({born.type.lower()})"] in tg.events(born.born)
    before = next(n for n in tg.nodes_edges(born.born - 1)[0] if n["id"] == born.id)
    assert before["color"] == model.STATE_COLOURS[model.GONE]
    closed = next(e for e in w.entities if e.closed is not None)
    node = next(n for n in tg.nodes_edges(closed.closed)[0] if n["id"] == closed.id)
    assert node["color"] == model.STATE_COLOURS[model.ENDED]
    after = next(n for n in tg.nodes_edges(closed.closed + 1)[0] if n["id"] == closed.id)
    assert after["color"] == model.STATE_COLOURS[model.GONE]


def test_stepping_moves_a_year_or_to_the_next_change_and_stops_at_the_ends() -> None:
    tg = model.time_graph(world())
    assert tg is not None
    assert tg.step(tg.first, 1, False) == tg.first + 1
    assert tg.step(tg.first, -1, False) == tg.first
    assert tg.step(tg.last, 1, True) == tg.last
    year = tg.first
    seen = [year]
    while (nxt := tg.step(year, 1, True)) != year:
        year = nxt
        seen.append(year)
    assert seen == list(tg.years)


def test_dated_documents_are_listed_up_to_the_chosen_year() -> None:
    w = world()
    docs = model.render_world(w, model.RENDER_DEFAULTS, "plain", True)
    assert all(d.date for d in docs)
    tg = model.time_graph(w)
    assert tg is not None
    mid = tg.years[len(tg.years) // 2]
    rows = model.document_rows(docs, mid)
    assert 0 < len(rows) < len(docs)
    assert all(int(r["date"][:4]) <= mid for r in rows)
    assert len(model.document_rows(docs)) == len(docs)  # no year: all of them, as before


def test_the_ui_shows_the_final_state_until_the_year_view_is_switched_on() -> None:
    ui = build_ui()
    ui.difficulty.set("clean")
    ui.preset.set("small")
    ui.evolution.set("heavy")
    ui.timeline.set("on")
    ui.generate()
    assert ui.tgraph.value is not None and not ui.viewing_time()
    assert ui.drawn() == ui.graph.value
    ui.time_on.set(True)
    assert ui.viewing_time() and ui.current_year() == ui.tgraph.value.first
    start = ui.drawn()
    ui.step_year(1, True)
    assert ui.current_year() == ui.tgraph.value.years[1]
    assert ui.drawn() != start
    assert len(ui.document_rows_now()) < len(ui.documents.value)
    assert ui.year_events()[0][0] in "+~-"
    ui.generate()  # a new world opens on its final state again
    assert not ui.viewing_time() and ui.offset.value == 0


def test_a_static_world_says_why_there_is_nothing_to_step() -> None:
    ui = build_ui()
    ui.difficulty.set("clean")
    ui.preset.set("small")
    ui.generate()
    ui.time_on.set(True)
    assert not ui.viewing_time()
    assert "does not change over time" in ui.year_note()
    assert ui.drawn() == ui.graph.value


def test_the_year_view_singles_out_what_the_year_changed() -> None:
    w = world()
    tg = model.time_graph(w)
    assert tg is not None
    assert tg.changed(tg.first) is None
    year = tg.years[1]
    shown = tg.changed(year)
    assert shown is not None and shown["nodes"] and shown["edges"]
    began = [f for f in w.edges if f.valid_from == year]
    assert all(f.id in shown["edges"] and f.subject in shown["nodes"] for f in began)
    nodes, edges = tg.nodes_edges(year)
    assert set(shown["nodes"]) <= {n["id"] for n in nodes}
    assert set(shown["edges"]) <= {e["id"] for e in edges}


def test_a_picked_question_still_wins_over_the_years_changes() -> None:
    ui = build_ui()
    ui.difficulty.set("clean")
    ui.preset.set("small")
    ui.evolution.set("heavy")
    ui.generate()
    ui.time_on.set(True)
    ui.step_year(1, True)
    assert ui.highlight() == ui.tgraph.value.changed(ui.current_year())
    ui.picked.set(ui.questions.value[0].id)
    assert ui.highlight() != ui.tgraph.value.changed(ui.current_year())
