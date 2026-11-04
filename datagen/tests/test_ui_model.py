"""The UI's logic, tested with no indah and no browser."""

from __future__ import annotations

import pytest
from kggen.questions import QUESTION_TYPES
from kggen.spec import PRESET_NAMES, load_preset
from kggen_ui import model

# Plain questions only, for tests that read a gold path as one chain with a fixed hop count.
ENTITY_ONLY = {
    **model.QUESTION_DEFAULTS,
    **{f"type_{t}": 0 for t in QUESTION_TYPES if t != "entity"},
}


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


# -- questions ---------------------------------------------------------------------------------


def question_world():
    values = {**model.preset_values("small"), "alias_rate": 0.5, "churn_rate": 0.1}
    return model.generate_world(values)


def test_the_question_form_defaults_to_twenty_questions_across_all_hop_counts() -> None:
    spec = model.question_spec_from_values(model.QUESTION_DEFAULTS)
    assert spec.n_questions == 20 and set(spec.hop_mix) == {1, 2, 3, 4}


def test_a_hop_weight_of_zero_drops_that_hop_count() -> None:
    values = {**ENTITY_ONLY, "hops_1": 0, "hops_2": 0, "hops_4": 0}
    assert set(model.question_spec_from_values(values).hop_mix) == {3}
    questions = model.derive_questions(question_world(), values)
    assert {q.hops for q in questions} == {3} and len(questions) == 20


def test_the_hop_mix_sets_the_distribution_of_derived_questions() -> None:
    values = {
        **ENTITY_ONLY,
        "n_questions": 40,
        "hops_1": 1,
        "hops_2": 3,
        "hops_3": 0,
        "hops_4": 0,
    }
    counts = {}
    for q in model.derive_questions(question_world(), values):
        counts[q.hops] = counts.get(q.hops, 0) + 1
    assert counts == {1: 10, 2: 30}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_questions": 0}, "Questions must be between"),
        ({"n_questions": ""}, "Questions needs a value"),
        ({"hops_2": 11}, "2-hop weight must be between"),
        ({"hops_1": 0, "hops_2": 0, "hops_3": 0, "hops_4": 0}, "at least one hop count"),
    ],
)
def test_a_bad_question_form_gives_a_readable_error(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model.question_spec_from_values({**model.QUESTION_DEFAULTS, **change})


def test_questions_a_world_cannot_supply_are_a_readable_error() -> None:
    tiny = model.generate_world({**model.preset_values("tiny"), "n_entities": 6, "avg_degree": 1})
    with pytest.raises(ValueError, match="could only derive"):
        model.derive_questions(
            tiny,
            {**model.QUESTION_DEFAULTS, "n_questions": 500, "hops_1": 0, "hops_2": 0, "hops_3": 0},
        )


def test_question_rows_carry_the_columns_the_table_shows() -> None:
    questions = model.derive_questions(question_world(), ENTITY_ONLY)
    rows = model.question_rows(questions)
    assert [set(r) for r in rows] == [set(model.QUESTION_COLUMNS)] * len(questions)
    assert [r["id"] for r in rows] == [q.id for q in questions]
    assert all(r["question"].startswith("Name the ") for r in rows)


def test_the_gold_path_is_the_chain_from_the_start_to_the_answer() -> None:
    world = question_world()
    index = model.index_world(world)
    for q in model.derive_questions(world, ENTITY_ONLY):
        nodes, edges = model.gold_path(index, q)
        assert nodes[0] == q.start and nodes[-1] == q.answer
        assert len(nodes) == q.hops + 1 and edges == list(q.supporting_facts)
        for hop, fid in enumerate(edges):
            fact = index.facts[fid]
            assert {fact.subject, fact.object} == {nodes[hop], nodes[hop + 1]}


def test_the_highlight_names_only_what_the_graph_draws() -> None:
    world = question_world()
    index = model.index_world(world)
    _, edges = model.graph_data(world)
    drawn_edges = {e["id"] for e in edges}
    drawn_nodes = {n["id"] for n in model.graph_data(world)[0]}
    for q in model.derive_questions(world, ENTITY_ONLY):
        highlight = model.highlight_for(index, q)
        assert set(highlight["edges"]) <= drawn_edges and set(highlight["nodes"]) <= drawn_nodes
    assert model.highlight_for(index, None) is None and model.highlight_for(None, q) is None


def test_the_gold_rows_read_each_fact_and_when_it_held() -> None:
    world = question_world()
    index = model.index_world(world)
    q = next(q for q in model.derive_questions(world, ENTITY_ONLY) if q.hops == 3)
    rows = model.gold_rows(index, q)
    assert [r[0] for r in rows] == ["1", "2", "3"] and [r[1] for r in rows] == list(
        q.supporting_facts
    )
    for row, fid in zip(rows, q.supporting_facts, strict=True):
        fact = index.facts[fid]
        subject, obj = index.entities[fact.subject].name, index.entities[fact.object].name
        assert row[2] == f"{subject} {fact.predicate} {obj}"
        assert row[3].startswith(str(fact.valid_from))
    assert model.gold_rows(index, None) == [] and model.gold_rows(None, q) == []


def test_the_answer_text_lists_accepted_aliases_and_prompts_when_nothing_is_picked() -> None:
    world = question_world()
    questions = model.derive_questions(world, {**ENTITY_ONLY, "n_questions": 100})
    aliased = next(q for q in questions if len(q.accepted_answers) > 1)
    text = model.answer_text(aliased)
    assert aliased.answer_name in text and aliased.accepted_answers[1] in text
    assert model.answer_text(None) == "Pick a question to see its answer and its gold path."
    assert (
        model.find_question(questions, aliased.id) is aliased
        and model.find_question(questions, None) is None
    )


# -- documents ---------------------------------------------------------------------------------


def doc_world():
    values = {
        **model.preset_values("small"),
        "alias_rate": 0.5,
        "churn_rate": 0.2,
        "distractor_rate": 0.1,
    }
    return model.generate_world(values)


def test_render_form_defaults_and_validation() -> None:
    spec = model.render_spec_from_values(model.RENDER_DEFAULTS)
    assert (spec.alias_use_rate, spec.max_facts_per_doc) == (0.5, 6)
    for bad, message in (
        ({"alias_use_rate": 2}, "Mentions that use an alias must be between"),
        ({"max_facts_per_doc": 0}, "Most facts in a document must be between"),
        ({"max_facts_per_doc": ""}, "needs a value"),
    ):
        with pytest.raises(ValueError, match=message):
            model.render_spec_from_values({**model.RENDER_DEFAULTS, **bad})


SHORT = {**model.RENDER_DEFAULTS, "target_words": 0}


def test_every_fact_has_exactly_one_sentence_in_the_lookup() -> None:
    world = doc_world()
    docs = model.render_world(world, SHORT)
    lookup = model.sentence_lookup(docs)
    assert set(lookup) == {f.id for f in world.edges}
    assert sum(len(d.sentences) for d in docs) == len(lookup)


def test_document_rows_flag_distractors_and_count_sentences() -> None:
    docs = model.render_world(doc_world(), model.RENDER_DEFAULTS)
    rows = model.document_rows(docs)
    assert [r["id"] for r in rows] == [d.id for d in docs]
    assert {r["kind"] for r in rows} == {"", "distractor"}
    assert [r["sentences"] for r in rows] == [len(d.sentences) for d in docs]


def test_the_document_list_is_capped_but_every_document_is_still_rendered() -> None:
    docs = model.render_world(doc_world(), {**model.RENDER_DEFAULTS, "max_facts_per_doc": 1})
    assert len(docs) > 100
    assert len(model.document_rows(docs)) == min(len(docs), model.MAX_DOCUMENT_ROWS)


def test_sentence_rows_are_keyed_by_the_fact_they_express() -> None:
    docs = model.render_world(doc_world(), SHORT)
    doc = docs[0]
    rows = model.sentence_rows(doc)
    assert [r["fact"] for r in rows] == [s.fact_ids[0] for s in doc.sentences]
    assert [r["sentence"] for r in rows] == [s.text for s in doc.sentences]
    assert model.sentence_rows(None) == [] and model.find_document(docs, doc.id) is doc


def test_a_fact_that_still_holds_highlights_its_entities_and_its_edge() -> None:
    world = doc_world()
    index = model.index_world(world)
    fact = world.current_edges()[0]
    assert model.fact_highlight(index, fact.id) == {
        "nodes": [fact.subject, fact.object],
        "edges": [fact.id],
    }


def test_a_fact_that_ended_highlights_its_entities_but_not_an_edge_the_graph_lacks() -> None:
    world = doc_world()
    index = model.index_world(world)
    _, drawn = model.graph_data(world)
    drawn_ids = {e["id"] for e in drawn}
    ended = [f for f in world.edges if f.valid_to is not None]
    assert ended
    for fact in ended:
        highlight = model.fact_highlight(index, fact.id)
        assert highlight["edges"] == [] and fact.id not in drawn_ids
        assert highlight["nodes"] == [fact.subject, fact.object]
    assert "ended" in model.fact_note(index, ended[0].id)


def test_every_fact_highlight_names_only_what_the_graph_draws() -> None:
    world = doc_world()
    index = model.index_world(world)
    nodes, edges = model.graph_data(world)
    node_ids, edge_ids = {n["id"] for n in nodes}, {e["id"] for e in edges}
    for fact in world.edges:
        highlight = model.fact_highlight(index, fact.id)
        assert set(highlight["nodes"]) <= node_ids and set(highlight["edges"]) <= edge_ids


def test_the_mentions_of_a_sentence_say_how_each_was_written_and_who_it_is() -> None:
    world = doc_world()
    index = model.index_world(world)
    lookup = model.sentence_lookup(
        model.render_world(world, {**model.RENDER_DEFAULTS, "alias_use_rate": 1})
    )
    aliased = [
        f
        for f in world.edges
        if index.entities[f.subject].aliases or index.entities[f.object].aliases
    ]
    rows = model.mention_rows(index, lookup, aliased[0].id)
    assert [r[2] for r in rows] == [aliased[0].subject, aliased[0].object]
    assert any(r[0] != r[1] for r in rows)  # at least one written as an alias
    assert (
        model.mention_rows(index, lookup, None) == []
        and model.mention_rows(None, lookup, "F0") == []
    )


def test_the_fact_note_prompts_until_a_sentence_is_picked() -> None:
    index = model.index_world(doc_world())
    for nothing in (None, "no-such-fact"):
        assert model.fact_note(index, nothing).startswith("Click a sentence")
        assert model.fact_highlight(index, nothing) is None
    assert model.fact_highlight(None, "F000") is None


def test_rendering_needs_no_network() -> None:
    import socket

    real = socket.socket

    def refuse(*args, **kwargs):
        raise AssertionError("rendering opened a socket")

    socket.socket = refuse
    try:
        assert model.render_world(doc_world(), model.RENDER_DEFAULTS)
    finally:
        socket.socket = real


# -- export ------------------------------------------------------------------------------------


def test_the_export_name_says_what_and_which() -> None:
    world = model.generate_world({**model.preset_values("tiny"), "seed": 9})
    assert model.export_name(world) == "kggen-30-entities-seed9.zip"


def test_export_falls_back_to_default_specs_when_a_form_was_not_valid() -> None:
    world = model.generate_world(model.preset_values("small"))
    data, name = model.export_dataset(world, None, None)
    assert name.endswith(".zip") and data[:2] == b"PK"
    assert "MB" in model.export_summary(data, name) and name in model.export_summary(data, name)


# --- the opt-in LLM renderer -------------------------------------------------------------------


def test_llm_render_counts_calls_then_serves_the_cache(endpoint, tmp_path) -> None:
    world = model.generate_world(model.preset_values("tiny"))
    client = endpoint()
    cold = model.render_world_llm(
        world, model.RENDER_DEFAULTS, client.base_url, "test-model", tmp_path
    )
    assert cold.calls == len(cold.documents) > 0 and cold.hits == 0
    warm = model.render_world_llm(world, model.RENDER_DEFAULTS, "", "test-model", tmp_path)
    assert (warm.calls, warm.hits) == (0, len(cold.documents))
    assert warm.documents == cold.documents
    assert f"{cold.calls} endpoint calls, 0 answered" in model.llm_note(cold, "test-model")


def test_llm_render_says_what_is_wrong_instead_of_raising(endpoint, tmp_path) -> None:
    world = model.generate_world(model.preset_values("tiny"))
    values = model.RENDER_DEFAULTS
    with pytest.raises(ValueError, match="model name"):
        model.render_world_llm(world, values, "", " ", tmp_path)
    with pytest.raises(ValueError, match="cache"):
        model.render_world_llm(world, values, "", "m", tmp_path)
    with pytest.raises(ValueError, match="stopped"):
        model.render_world_llm(world, values, "http://127.0.0.1:9/v1", "m", tmp_path)
    nonsense = endpoint(lambda prompt: "not json")
    with pytest.raises(ValueError, match="stopped"):
        model.render_world_llm(world, values, nonsense.base_url, "m", tmp_path)


def test_every_setting_has_an_info_tip() -> None:
    knobs = (*model.KNOBS, *model.QUESTION_KNOBS, *model.RENDER_KNOBS)
    assert [k.name for k in knobs if not model.knob_help(k)] == []


def test_the_form_accepts_the_stress_preset_size() -> None:
    assert model.MAX_ENTITIES >= 50_000
    model.read_knob(model.KNOBS[0], 50_000)


def test_the_question_form_reads_a_type_mix_and_rejects_all_zero() -> None:
    values = {**model.QUESTION_DEFAULTS, "type_entity": 0, "type_set": 3}
    spec = model.question_spec_from_values(values)
    assert spec.type_mix == {t: (3.0 if t == "set" else 1.0) for t in QUESTION_TYPES[1:]}
    with pytest.raises(ValueError, match="question type"):
        model.question_spec_from_values({**values, **{f"type_{t}": 0 for t in spec.type_mix}})


def test_a_set_question_highlights_every_entity_and_fact_it_involves() -> None:
    world = model.generate_world(model.preset_values("small"))
    index = model.index_world(world)
    qs = model.derive_questions(world, {**model.QUESTION_DEFAULTS, "n_questions": 30})
    q = next(q for q in qs if q.type == "set")
    nodes, edges = model.gold_path(index, q)
    assert set(q.answers) <= set(nodes) and set(edges) == {f for p in q.gold_paths for f in p}
    assert model.answer_text(q).startswith(f"Answers ({len(q.answers)})")


def test_the_question_summary_names_the_types_so_a_fallback_is_visible() -> None:
    world = model.generate_world(model.preset_values("small"))
    qs = model.derive_questions(world, {**model.QUESTION_DEFAULTS, "n_questions": 30})
    text = model.question_summary(qs)
    assert text.startswith(f"{len(qs)} questions: ") and "entity" in text
    assert "as_of" not in text  # the small world never changes, so those fell back
    assert model.question_summary([]) == ""


def test_a_long_document_lists_only_fact_sentences_and_shows_its_text() -> None:
    world = doc_world()
    docs = model.render_world(world, {**model.RENDER_DEFAULTS, "target_words": 500})
    doc = docs[0]
    rows = model.sentence_rows(doc)
    assert rows and len(rows) < len(doc.sentences)
    assert [r["fact"] for r in rows] == [s.fact_ids[0] for s in doc.sentences if s.fact_ids]
    assert model.document_text(doc) == doc.text and model.document_text(None) == ""
    assert model.document_rows(docs)[0]["words"] == len(doc.text.split()) >= 500


def test_the_lookup_prefers_the_sentence_in_the_facts_own_document() -> None:
    world = doc_world()
    docs = model.render_world(world, {**model.RENDER_DEFAULTS, "target_words": 300})
    lookup = model.sentence_lookup(docs)
    assert set(lookup) == {f.id for f in world.edges}
    subject_of = {f.id: f.subject for f in world.edges}
    for d in docs:
        for s in d.sentences:
            for f in s.fact_ids:
                if subject_of[f] == d.subject:
                    assert lookup[f] is s


def test_the_ui_refuses_to_render_more_words_than_it_can_hold() -> None:
    world = doc_world()
    huge = {**model.RENDER_DEFAULTS, "target_words": 10_000}
    with pytest.raises(ValueError, match="words the UI renders"):
        model.render_world(
            type("W", (), {"entities": range(model.MAX_RENDER_WORDS // 10_000 + 1)})(), huge
        )
    assert model.render_world(world, {**model.RENDER_DEFAULTS, "target_words": 100})
