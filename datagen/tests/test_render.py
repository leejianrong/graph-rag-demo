"""Template rendering is deterministic and offline, and every sentence maps back to its facts."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from kggen.cli import main
from kggen.render import TEMPLATES, RenderSpec, render_documents
from kggen.spec import PREDICATE_SIGNATURES, VAGUE_PREDICATE, WorldSpec
from kggen.world import generate


def make(**overrides: object) -> WorldSpec:
    base: dict[str, object] = {
        "n_entities": 300,
        "avg_degree": 4,
        "seed": 4,
        "alias_rate": 0.6,
        "distractor_rate": 0.1,
        "noise_rate": 0.1,
        "churn_rate": 0.1,
        "retraction_rate": 0.05,
    }
    return WorldSpec.model_validate({**base, **overrides})


@pytest.fixture(scope="module")
def world():
    return generate(make())


def test_every_relation_has_templates() -> None:
    assert set(TEMPLATES) == set(PREDICATE_SIGNATURES) | {VAGUE_PREDICATE}


def test_rendering_is_deterministic_and_seeded(world) -> None:
    assert render_documents(world) == render_documents(world)
    assert render_documents(world, RenderSpec(seed=1)) != render_documents(
        world, RenderSpec(seed=2)
    )


def test_every_fact_is_expressed_by_exactly_one_sentence(world) -> None:
    docs = render_documents(world)
    cited = Counter(f for d in docs for s in d.sentences for f in s.fact_ids)
    assert set(cited) == {e.id for e in world.edges}
    assert set(cited.values()) == {1}
    assert all(s.fact_ids for d in docs for s in d.sentences)


def test_sentence_and_mention_offsets_index_the_document_text(world) -> None:
    for d in render_documents(world):
        assert d.text == " ".join(s.text for s in d.sentences)
        for s in d.sentences:
            assert d.text[s.start : s.end] == s.text
            for m in s.mentions:
                assert d.text[m.start : m.end] == m.surface
                assert s.start <= m.start < m.end <= s.end


def test_mentions_are_the_names_or_aliases_of_the_facts_entities(world) -> None:
    by_id = {e.id: e for e in world.entities}
    facts = {e.id: e for e in world.edges}
    for d in render_documents(world):
        for s in d.sentences:
            fact = facts[s.fact_ids[0]]
            assert [m.entity_id for m in s.mentions] == [fact.subject, fact.object]
            for m in s.mentions:
                entity = by_id[m.entity_id]
                assert m.surface in (entity.name, *entity.aliases)


def test_aliases_appear_in_text_when_enabled_and_never_when_off(world) -> None:
    by_id = {e.id: e for e in world.entities}

    def alias_mentions(spec: RenderSpec) -> int:
        return sum(
            m.surface != by_id[m.entity_id].name
            for d in render_documents(world, spec)
            for s in d.sentences
            for m in s.mentions
        )

    assert alias_mentions(RenderSpec(alias_use_rate=0.0)) == 0
    assert alias_mentions(RenderSpec(alias_use_rate=1.0)) > 50


def test_distractor_documents_are_flagged_and_stay_apart(world) -> None:
    by_id = {e.id: e for e in world.entities}
    docs = render_documents(world)
    flagged = [d for d in docs if d.distractor]
    assert flagged and len(flagged) < len(docs)
    for d in docs:
        involved = {m.entity_id for s in d.sentences for m in s.mentions}
        assert {by_id[i].distractor for i in involved} == {d.distractor}


def test_changed_facts_carry_their_dates(world) -> None:
    facts = {e.id: e for e in world.edges}
    texts = {s.fact_ids[0]: s.text for d in render_documents(world) for s in d.sentences}
    ended = [f for f in facts.values() if f.valid_to is not None]
    replaced = [f for f in facts.values() if f.valid_from > world.spec.start_year]
    assert ended and replaced
    assert all(texts[f.id].endswith(f"until {f.valid_to}.") for f in ended)
    assert all(texts[f.id].endswith(f"since {f.valid_from}.") for f in replaced)


def test_documents_respect_max_facts_per_doc(world) -> None:
    docs = render_documents(world, RenderSpec(max_facts_per_doc=2))
    assert max(len(d.sentences) for d in docs) == 2
    assert len({d.id for d in docs}) == len(docs)


def test_cli_writes_documents(tmp_path) -> None:
    out = tmp_path / "docs.json"
    assert main(["docs", "generate", "--preset", "tiny", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data and all(d["sentences"] and d["text"] for d in data)


# -- long documents ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_world():
    return generate(make(n_entities=60, avg_degree=3))


@pytest.mark.parametrize("target", [500, 2000])
def test_a_long_document_reaches_its_word_target(small_world, target) -> None:
    docs = render_documents(small_world, RenderSpec(target_words=target))
    words = [len(d.text.split()) for d in docs]
    assert min(words) >= target and max(words) <= target + 60


def test_long_documents_keep_exact_offsets_and_mentions(small_world) -> None:
    ents = {e.id: e for e in small_world.entities}
    for d in render_documents(small_world, RenderSpec(target_words=500)):
        for s in d.sentences:
            assert d.text[s.start : s.end] == s.text
            for m in s.mentions:
                assert d.text[m.start : m.end] == m.surface
                assert m.surface in (ents[m.entity_id].name, *ents[m.entity_id].aliases)
        starts = [s.start for s in d.sentences]
        assert starts == sorted(starts)


def test_every_fact_is_still_expressed_and_filler_states_none(small_world) -> None:
    docs = render_documents(small_world, RenderSpec(target_words=500))
    expressed = {f for d in docs for s in d.sentences for f in s.fact_ids}
    assert expressed == {e.id for e in small_world.edges}
    fact_words = sum(len(s.text.split()) for d in docs for s in d.sentences if s.fact_ids)
    assert fact_words < 0.3 * sum(len(d.text.split()) for d in docs)  # mostly background


def test_filler_names_no_entity_but_the_documents_subject(small_world) -> None:
    by_name = {e.name: e.id for e in small_world.entities}
    for d in render_documents(small_world, RenderSpec(target_words=500)):
        subject = by_name[d.title]
        for s in d.sentences:
            if not s.fact_ids:
                assert {m.entity_id for m in s.mentions} <= {subject}


def test_a_long_document_opens_and_has_paragraphs(small_world) -> None:
    docs = render_documents(small_world, RenderSpec(target_words=2000))
    assert all(not d.sentences[0].fact_ids and d.sentences[0].mentions for d in docs)
    assert all("\n\n" in d.text and not d.text.endswith("\n") for d in docs)


def test_long_documents_add_context_facts_about_connected_entities(small_world) -> None:
    on = render_documents(small_world, RenderSpec(target_words=500, neighbour_facts=3))
    off = render_documents(small_world, RenderSpec(target_words=500, neighbour_facts=0))
    ids = lambda ds: sum(len(s.fact_ids) for d in ds for s in d.sentences)  # noqa: E731
    assert ids(on) > ids(off) == len(small_world.edges)


def test_distractor_documents_never_mention_a_real_entity() -> None:
    world = generate(make(n_entities=80, distractor_rate=0.3))
    ents = {e.id: e for e in world.entities}
    by_name = {}
    for e in world.entities:
        by_name.setdefault(e.name, []).append(e)
    seen = 0
    for d in render_documents(world, RenderSpec(target_words=500)):
        if d.distractor:
            seen += 1
            assert all(ents[m.entity_id].distractor for s in d.sentences for m in s.mentions)
    assert seen


def test_long_rendering_is_deterministic_and_the_default_stays_short(small_world) -> None:
    spec = RenderSpec(target_words=500)
    assert render_documents(small_world, spec) == render_documents(small_world, spec)
    assert RenderSpec().target_words == 0
    assert max(len(d.text.split()) for d in render_documents(small_world)) < 100


def test_a_long_document_does_not_repeat_a_sentence_word_for_word(small_world) -> None:
    for d in render_documents(small_world, RenderSpec(target_words=2000)):
        texts = [s.text for s in d.sentences if not s.fact_ids]
        assert len(set(texts)) == len(texts)
