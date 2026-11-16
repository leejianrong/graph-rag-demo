"""Worlds that change over time, and dated documents about them (GRA-190)."""

from __future__ import annotations

import json

import pytest
from kggen.cli import main
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import RenderSpec, render_documents
from kggen.spec import EVOLUTION, load_preset, with_evolution
from kggen.world import generate

SEEDS = (0, 1, 2)


def evolving(seed: int = 0, level: str = "heavy"):
    return generate(load_preset("small", "hard", level).with_overrides(seed=seed))


@pytest.fixture(scope="module")
def world():
    return evolving()


def test_the_default_world_has_no_evolution_and_serialises_as_before() -> None:
    plain = generate(load_preset("small"))
    assert all(e.born is None and e.closed is None and e.renamed is None for e in plain.entities)
    assert "born" not in json.dumps(plain.to_dict())
    spec = json.loads(plain.spec.model_dump_json())
    assert not {"birth_rate", "closure_rate", "rename_rate"} & set(spec)


def test_a_level_sets_the_knobs_and_none_changes_nothing() -> None:
    base = load_preset("small")
    assert with_evolution(base, "none") == base
    heavy = with_evolution(base, "heavy")
    assert heavy.birth_rate > 0 and heavy.closure_rate > 0 and heavy.rename_rate > 0
    with pytest.raises(ValueError, match="unknown evolution"):
        with_evolution(base, "wild")
    assert list(EVOLUTION) == ["none", "light", "heavy"]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_timeline_is_consistent(seed: int) -> None:
    world = evolving(seed)
    entity = {e.id: e for e in world.entities}
    for f in world.edges:
        assert f.valid_to is None or f.valid_from < f.valid_to
        for end in (f.subject, f.object):
            e = entity[end]
            assert e.born is None or e.born <= f.valid_from, "a fact starts before its entity"
            assert e.closed is None or (f.valid_to is not None and f.valid_to <= e.closed), (
                "a fact outlives its entity"
            )
    for e in world.entities:
        if e.renamed is not None:
            assert e.former_name and e.former_name != e.name and e.former_name in e.aliases
            assert (e.born or world.spec.start_year) < e.renamed
            assert e.closed is None or e.renamed < e.closed


def test_a_heavy_world_has_every_kind_of_change(world) -> None:
    assert any(e.born for e in world.entities)
    assert any(e.closed for e in world.entities)
    assert any(e.renamed for e in world.entities)
    assert any(f.valid_to is not None for f in world.edges)


def test_state_at_a_date(world) -> None:
    last = world.spec.end_year
    start = world.spec.start_year
    born = next(e for e in world.entities if e.born)
    assert not born.exists_in(born.born - 1) and born.exists_in(born.born)
    assert born not in world.entities_as_of(born.born - 1)
    closed = next(e for e in world.entities if e.closed)
    assert closed.exists_in(closed.closed - 1) and not closed.exists_in(closed.closed)
    renamed = next(e for e in world.entities if e.renamed)
    assert renamed.name_in(renamed.renamed - 1) == renamed.former_name
    assert renamed.name_in(renamed.renamed) == renamed.name
    # Nothing true in a year involves an entity that does not exist that year.
    entity = {e.id: e for e in world.entities}
    for year in (start, (start + last) // 2, last):
        for f in world.edges_as_of(year):
            assert entity[f.subject].exists_in(year) and entity[f.object].exists_in(year)


def test_the_same_spec_makes_the_same_world() -> None:
    assert evolving().to_json() == evolving().to_json()
    assert evolving(0).to_json() != evolving(1).to_json()


# --- dated documents --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timeline(world):
    return render_documents(world, RenderSpec(timeline=True))


def test_timeline_documents_are_dated_and_in_date_order(timeline) -> None:
    dates = [d.date for d in timeline]
    assert all(dates) and dates == sorted(dates)
    assert dates[0].startswith(("2000",))
    plain = render_documents(evolving(), RenderSpec())
    assert all(d.date is None for d in plain) and "date" not in plain[0].to_dict()
    assert timeline[0].to_dict()["date"] == dates[0]


def test_every_mention_points_at_its_words(timeline) -> None:
    for d in timeline:
        for s in d.sentences:
            for m in s.mentions:
                assert d.text[m.start : m.end] == m.surface


def test_each_fact_is_stated_when_it_begins_and_again_when_it_ends(world, timeline) -> None:
    start = world.spec.start_year
    when: dict[str, list[tuple[int, str]]] = {}
    for d in timeline:
        for s in d.sentences:
            for fid in s.fact_ids:
                when.setdefault(fid, []).append((int(d.date[:4]), s.text))
    for f in world.edges:
        years = {y for y, _ in when[f.id]}
        assert f.valid_from in years
        if f.valid_to is not None:
            assert f.valid_to in years
            assert any(f"until {f.valid_to}" in t for _, t in when[f.id])
        if f.valid_from > start:
            assert any(f"since {f.valid_from}" in t for y, t in when[f.id] if y == f.valid_from)


def test_an_entity_goes_by_the_name_it_had_that_year(world, timeline) -> None:
    renamed = {e.id: e for e in world.entities if e.renamed}
    before = after = 0
    for d in timeline:
        year = int(d.date[:4])
        for s in d.sentences:
            if "was renamed" in s.text:
                continue
            for m in s.mentions:
                e = renamed.get(m.entity_id)
                if e is None or m.surface in e.aliases and m.surface != e.former_name:
                    continue
                if year < e.renamed:
                    assert m.surface in (e.former_name, e.name_in(year))
                    before += 1
                else:
                    assert m.surface == e.name
                    after += 1
    assert before and after


def test_lifecycle_sentences_state_no_fact(world, timeline) -> None:
    text = " ".join(s.text for d in timeline for s in d.sentences if not s.fact_ids)
    assert "was established in" in text
    assert "ceased operations in" in text
    assert "was renamed" in text


def test_timeline_documents_are_repeatable_and_short(world) -> None:
    a = render_documents(world, RenderSpec(timeline=True))
    assert a == render_documents(world, RenderSpec(timeline=True))
    with pytest.raises(ValueError, match="timeline documents are short"):
        RenderSpec(timeline=True, target_words=300)


# --- the questions ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def asked(world):
    mix = {"when": 1, "before": 1, "duration": 1}
    return generate_questions(world, QuestionSpec(n_questions=90, type_mix=mix))


def test_every_temporal_kind_is_asked(asked) -> None:
    kinds = {q.type for q in asked}
    assert {"when", "before", "duration"} <= kinds


def test_when_and_duration_answers_match_the_facts_and_are_in_the_text(
    world, asked, timeline
) -> None:
    facts = {f.id: f for f in world.edges}
    stated = {
        fid: " ".join(s.text for d in timeline for s in d.sentences if fid in s.fact_ids)
        for fid in facts
    }
    got = {"when": 0, "duration": 0}
    for q in asked:
        if q.type == "when":
            f = facts[q.supporting_facts[0]]
            assert q.number in (f.valid_from, f.valid_to)
            assert str(q.number) in stated[f.id], "the year must be readable from the documents"
            got["when"] += 1
        elif q.type == "duration":
            f = facts[q.supporting_facts[0]]
            assert q.number == f.valid_to - f.valid_from > 0
            assert str(f.valid_from) in stated[f.id] and str(f.valid_to) in stated[f.id]
            got["duration"] += 1
        assert q.grading == "numeric" or q.type == "before"
    assert all(got.values())


def test_before_names_what_held_just_before(world, asked) -> None:
    facts = {f.id: f for f in world.edges}
    found = [q for q in asked if q.type == "before"]
    assert found
    for q in found:
        earlier, later = (facts[i] for i in q.supporting_facts)
        assert earlier.subject == later.subject and earlier.predicate == later.predicate
        assert earlier.valid_to == later.valid_from
        assert q.answer == earlier.object != later.object


def test_a_world_with_no_change_asks_no_temporal_question() -> None:
    still = generate(load_preset("small"))
    qs = generate_questions(still, QuestionSpec(n_questions=10, type_mix={"when": 1, "entity": 1}))
    assert all(q.type == "entity" for q in qs)


# --- the command line ---------------------------------------------------------------------------


def test_cli_evolution_and_timeline(tmp_path, capsys) -> None:
    world = tmp_path / "w.json"
    assert (
        main(
            ["world", "generate", "--preset", "small", "--evolution", "heavy", "--out", str(world)]
        )
        == 0
    )
    assert '"closed"' in world.read_text()
    docs = tmp_path / "d.json"
    assert (
        main(
            [
                "docs",
                "generate",
                "--preset",
                "small",
                "--evolution",
                "heavy",
                "--timeline",
                "--out",
                str(docs),
            ]
        )
        == 0
    )
    assert '"date": "2000-07-01"' in docs.read_text()
