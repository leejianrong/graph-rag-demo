"""The indah layer: the form fills from a preset and Generate reports the world's size."""

from __future__ import annotations

import json

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
def test_an_oversize_world_is_drawn_as_a_sample_and_says_so() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.values["n_entities"].set(8000)
    ui.values["avg_degree"].set(20)
    ui.generate()
    nodes, edges = ui.graph.value
    assert nodes and edges and len(edges) <= 60_000
    assert "most connected" in ui.graph_note.value
    assert ui.entities.value == "8,000"  # the world is whole and its stats are shown
    assert ui.checks.value


@needs_graph
def test_a_sampled_graph_is_asked_to_emphasise_only_what_it_draws() -> None:
    ui = build_ui()
    ui.preset.set("tiny")
    ui.values["n_entities"].set(8000)
    ui.values["avg_degree"].set(20)
    ui.generate()
    nodes, edges = ui.graph.value
    drawn_nodes, drawn_edges = {n["id"] for n in nodes}, {e["id"] for e in edges}
    world = ui.world.value
    fact = next(f for f in world.current_edges() if f.subject not in drawn_nodes)
    ui.picked_fact.set(fact.id)
    wanted = ui.highlight()
    assert wanted["edges"] == [] and set(wanted["nodes"]) <= drawn_nodes
    assert ui.focus() is None  # the clicked fact's subject is not on the graph
    inside = next(f for f in world.current_edges() if f.id in drawn_edges)
    ui.picked_fact.set(inside.id)
    assert ui.highlight()["edges"] == [inside.id] and ui.focus() == inside.subject


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


# -- documents -----------------------------------------------------------------------------------


def test_generate_renders_documents_and_the_list_matches() -> None:
    ui = build_ui()
    ui.preset.set("small")
    assert ui.documents.value == ()
    ui.generate()
    assert ui.documents.value
    assert [r["id"] for r in ui.doc_rows.value] == [d.id for d in ui.documents.value]
    assert ui.doc_note.value.endswith("documents.") and set(ui.sentences.value) == {
        f.id for f in ui.world.value.edges
    }


def test_the_render_form_changes_the_documents_without_regenerating_the_world() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    world = ui.world.value
    before = len(ui.documents.value)
    ui.doc_values["max_facts_per_doc"].set(1)
    ui.render()
    assert len(ui.documents.value) > before and ui.world.value is world


def test_a_bad_render_form_says_why_and_keeps_the_world() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    ui.doc_values["alias_use_rate"].set(5)
    ui.render()
    assert "between" in ui.doc_note.value and ui.documents.value == ()
    assert ui.entities.value == "150"


def test_rendering_before_a_world_exists_asks_for_one() -> None:
    ui = build_ui()
    ui.render()
    assert ui.doc_note.value == "Generate a world first."


def test_opening_another_document_clears_the_clicked_sentence() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    first, second = ui.documents.value[0], ui.documents.value[1]
    ui.picked_doc.set(first.id)
    ui.picked_fact.set(first.sentences[0].fact_ids[0])
    ui.picked_doc.set(second.id)
    assert ui.picked_fact.value is None


def test_a_question_and_a_sentence_never_highlight_at_once() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    ui.picked_doc.set(ui.documents.value[0].id)
    fact = ui.documents.value[0].sentences[0].fact_ids[0]
    ui.picked.set(ui.questions.value[0].id)
    ui.picked_fact.set(fact)
    assert ui.picked.value is None and ui.picked_fact.value == fact
    ui.picked.set(ui.questions.value[0].id)
    assert ui.picked_fact.value is None and ui.picked.value == ui.questions.value[0].id


def test_regenerating_clears_the_open_document_and_sentence() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    ui.picked_doc.set(ui.documents.value[0].id)
    ui.picked_fact.set(ui.documents.value[0].sentences[0].fact_ids[0])
    ui.values["seed"].set(ui.values["seed"].value + 1)
    ui.generate()
    assert ui.picked_doc.value is None and ui.picked_fact.value is None


@needs_graph
def test_clicking_a_sentence_highlights_its_fact_and_centres_on_its_subject() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    fact = ui.world.value.current_edges()[0]
    ui.picked_fact.set(fact.id)
    props = graph_props(ui)
    assert props["highlight"] == {"nodes": [fact.subject, fact.object], "edges": [fact.id]}
    assert props["focus"] == fact.subject


@needs_graph
def test_clicking_a_sentence_about_a_fact_that_ended_does_not_break_the_graph() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.values["churn_rate"].set(0.3)
    ui.generate()
    ended = next(f for f in ui.world.value.edges if f.valid_to is not None)
    ui.picked_fact.set(ended.id)  # an edge the graph does not draw: must not raise
    props = graph_props(ui)
    assert props["highlight"] == {"nodes": [ended.subject, ended.object], "edges": []}


# -- export --------------------------------------------------------------------------------------

import io  # noqa: E402
import zipfile  # noqa: E402

from kggen.export import Specs, reproduce, zip_bytes  # noqa: E402
from kggen.questions import QuestionSpec  # noqa: E402
from kggen.render import RenderSpec  # noqa: E402
from kggen.spec import WorldSpec  # noqa: E402


def exported_files(ui) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(ui.export_file.value.data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_preparing_an_export_offers_a_zip_of_the_world_on_screen() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    assert ui.export_file.value is None
    ui.prepare_export()
    file = ui.export_file.value
    assert file.filename == "kggen-150-entities-seed0.zip" and file.media_type == "application/zip"
    files = exported_files(ui)
    assert {"world.json", "questions.json", "documents.json", "spec.json", "README.md"} <= set(
        files
    )
    assert file.filename in ui.export_note.value


def test_exporting_before_a_world_exists_asks_for_one() -> None:
    ui = build_ui()
    ui.prepare_export()
    assert ui.export_note.value == "Generate a world first." and ui.export_file.value is None


def test_the_download_remakes_the_same_bytes_from_the_specs_inside_it() -> None:
    """The card's promise, for the file the UI actually offers."""
    ui = build_ui()
    ui.preset.set("small")
    ui.values["alias_rate"].set(0.4)
    ui.values["distractor_rate"].set(0.1)
    ui.values["churn_rate"].set(0.1)
    ui.question_values["n_questions"].set(12)
    ui.doc_values["max_facts_per_doc"].set(4)
    ui.generate()
    ui.prepare_export()
    files = exported_files(ui)
    again = reproduce(
        Specs(
            world=WorldSpec.model_validate_json(files["spec.json"]),
            questions=QuestionSpec.model_validate_json(files["questions_spec.json"]),
            render=RenderSpec.model_validate_json(files["render_spec.json"]),
        )
    )
    assert again == files
    assert ui.export_file.value.data == zip_bytes(files)
    assert len(json.loads(files["questions.json"])) == 12  # the form's settings, not the defaults


def test_the_export_records_the_settings_the_questions_on_screen_were_made_with() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    for hop in (1, 2, 4):
        ui.question_values[f"hops_{hop}"].set(0)
    ui.derive()
    ui.prepare_export()
    spec = QuestionSpec.model_validate_json(exported_files(ui)["questions_spec.json"])
    assert set(spec.hop_mix) == {3}
    questions = json.loads(exported_files(ui)["questions.json"])
    assert {q["hops"] for q in questions} == {3}


def test_changing_the_world_the_questions_or_the_documents_drops_a_stale_export() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.generate()
    for change in (ui.generate, ui.derive, ui.render):
        ui.prepare_export()
        assert ui.export_file.value is not None
        change()
        assert ui.export_file.value is None and ui.export_note.value == ""


def test_the_export_card_is_in_the_page_with_a_download_link() -> None:
    ui = build_ui()
    kinds = []

    def walk(node):
        kinds.append(node.get("type"))
        for child in node.get("children", []) or []:
            walk(child)

    walk(indah.Session(ui.root).snapshot())
    assert "download" in kinds


# --- the opt-in LLM renderer -------------------------------------------------------------------


@pytest.fixture
def llm_ui(endpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "LLM_CACHE_DIR", tmp_path)
    ui = build_ui()
    ui.preset.set("tiny")
    ui.llm_endpoint.set(endpoint().base_url)
    ui.llm_model.set("test-model")
    return ui


def test_the_llm_renderer_is_off_by_default_and_a_default_render_calls_nothing(
    llm_ui, endpoint
) -> None:
    assert llm_ui.llm_on.value is False
    llm_ui.generate()
    llm_ui.render()
    assert endpoint.served == []
    assert llm_ui.doc_source.value == "templates" and llm_ui.documents.value


def test_generating_a_world_never_calls_the_llm_even_with_the_switch_on(llm_ui, endpoint) -> None:
    llm_ui.llm_on.set(True)
    llm_ui.generate()
    assert endpoint.served == []
    assert llm_ui.doc_source.value == "templates"


def test_enabling_it_shows_calls_made_versus_cached(llm_ui, endpoint) -> None:
    llm_ui.generate()
    llm_ui.llm_on.set(True)
    llm_ui.render()
    n = len(llm_ui.documents.value)
    assert llm_ui.doc_source.value == "llm"
    assert f"{n} endpoint calls, 0 answered from the cache" in llm_ui.doc_note.value
    assert len(endpoint.served) == n
    llm_ui.render()
    assert f"0 endpoint calls, {n} answered from the cache" in llm_ui.doc_note.value
    assert len(endpoint.served) == n


def test_a_failed_llm_render_says_why_and_keeps_the_documents(llm_ui) -> None:
    llm_ui.generate()
    templates = llm_ui.documents.value
    llm_ui.llm_on.set(True)
    llm_ui.llm_endpoint.set("")  # cache only, and the cache is empty
    llm_ui.render()
    assert "cache" in llm_ui.doc_note.value
    assert llm_ui.documents.value == templates and llm_ui.doc_source.value == "templates"


def test_exporting_llm_documents_says_the_zip_holds_template_text(llm_ui) -> None:
    llm_ui.generate()
    llm_ui.llm_on.set(True)
    llm_ui.render()
    llm_ui.prepare_export()
    assert "template text" in llm_ui.export_note.value
