"""Extraction scoring: perfect scores 1, and each way of being bad costs the right metric."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from kgbench.cli import main
from kgbench.extraction import (
    PRF,
    Extraction,
    PredictedMention,
    gold_from,
    oracle,
    run_pipeline,
    score,
)
from kggen.render import RenderSpec, render_documents
from kggen.spec import WorldSpec
from kggen.world import generate

from graph_rag.wiring import build_offline_components


def build(**knobs: object):
    world = generate(WorldSpec(n_entities=200, avg_degree=3, seed=2, **knobs))  # type: ignore[arg-type]
    documents = render_documents(world, RenderSpec(alias_use_rate=0.6))
    return world, documents, gold_from(world, documents)


@pytest.fixture(scope="module")
def ambiguous():
    return build(alias_rate=0.7, aliases_per_entity=2, homonym_rate=0.15, distractor_rate=0.2)


@pytest.fixture(scope="module")
def clean():
    return build()


def test_a_perfect_extractor_scores_one_everywhere(ambiguous) -> None:
    _, _, gold = ambiguous
    report = score(gold, oracle(gold))
    for prf in (report.mentions, report.entities, report.triples, report.edges):
        assert (prf.precision, prf.recall, prf.f1) == (1.0, 1.0, 1.0)
    assert report.by_predicate
    assert all(s.strict.f1 == 1.0 and s.edge_recall == 1.0 for s in report.by_predicate.values())


def test_the_report_has_prf_per_predicate_for_every_gold_predicate(ambiguous) -> None:
    _, _, gold = ambiguous
    report = score(gold, oracle(gold))
    assert set(report.by_predicate) == {p for _, p, _ in gold.triples}
    assert sum(s.gold for s in report.by_predicate.values()) == len(gold.triples)
    assert "AFFILIATED_WITH" in report.format_table() or "WORKS_FOR" in report.format_table()


def test_dropping_mentions_costs_recall_not_precision(ambiguous) -> None:
    _, _, gold = ambiguous
    perfect = oracle(gold)
    half = replace(perfect, mentions=perfect.mentions[::2])
    report = score(gold, half)
    assert report.mentions.precision == 1.0
    assert report.mentions.recall == pytest.approx(0.5, abs=0.01)


def test_shifted_spans_score_zero_mentions(ambiguous) -> None:
    _, _, gold = ambiguous
    shifted = replace(
        oracle(gold),
        mentions=tuple(replace(m, start=m.start + 1, end=m.end + 1) for m in oracle(gold).mentions),
    )
    assert score(gold, shifted).mentions.f1 == 0.0


def test_no_linking_makes_duplicates_and_costs_entity_precision(ambiguous) -> None:
    _, _, gold = ambiguous
    # every mention becomes its own canonical entity: spans are perfect, linking is absent
    unlinked = replace(
        oracle(gold),
        mentions=tuple(
            replace(m, canonical_id=f"c{i}") for i, m in enumerate(oracle(gold).mentions)
        ),
        triples=(),
    )
    report = score(gold, unlinked)
    assert report.mentions.f1 == 1.0
    # one canonical per mention: only one canonical per gold entity is right
    assert report.entities.precision == pytest.approx(len(gold.entities) / len(gold.mentions))
    assert (
        report.entities.recall == 1.0
    )  # every gold entity is still the plurality of some canonical


def test_merging_homonyms_costs_entity_recall(ambiguous) -> None:
    world, _, gold = ambiguous
    names = {e.id: e.name for e in world.entities}
    assert len({n for n in names.values()}) < len(names)  # the world really has homonyms
    merged = replace(
        oracle(gold),
        mentions=tuple(
            replace(m, canonical_id=names[m.canonical_id]) for m in oracle(gold).mentions
        ),
        triples=(),
    )
    report = score(gold, merged)
    assert report.entities.recall < 1.0
    assert report.mentions.f1 == 1.0


def test_wrong_predicates_cost_strict_triples_but_not_edges(ambiguous) -> None:
    _, _, gold = ambiguous
    vague = replace(oracle(gold), triples=tuple((s, "RELATED_TO", o) for s, _, o in gold.triples))
    report = score(gold, vague)
    assert report.triples.f1 == 0.0
    assert report.edges.f1 == 1.0
    assert all(s.edge_recall == 1.0 for s in report.by_predicate.values() if s.gold)
    assert report.by_predicate["RELATED_TO"].gold == 0


def test_duplicate_predictions_cost_precision_and_never_push_recall_over_one(ambiguous) -> None:
    _, _, gold = ambiguous
    perfect = oracle(gold)
    # the same facts again under a second set of canonical ids that align to the same entities
    twin = {m.canonical_id: f"{m.canonical_id}~twin" for m in perfect.mentions}
    mentions = perfect.mentions + tuple(
        replace(m, canonical_id=twin[m.canonical_id]) for m in perfect.mentions
    )
    triples = perfect.triples + tuple((twin[s], p, twin[o]) for s, p, o in perfect.triples)
    report = score(gold, Extraction(mentions, triples))
    assert report.triples.recall == 1.0
    assert report.triples.precision == 1.0  # both copies are right
    assert (
        report.entities.recall == 1.0
        and report.entities.precision == 1.0
        or report.entities.precision < 1.0
    )
    assert report.entities.predicted == 2 * len({m.canonical_id for m in perfect.mentions})


def test_unlinked_and_unaligned_triples_count_as_wrong() -> None:
    gold = _tiny_gold()
    wrong = Extraction(
        (PredictedMention("D0", 0, 3, "x"),), (("x", "WORKS_FOR", "y"), ("x", "OWNS", "x"))
    )
    report = score(gold, wrong)
    assert report.triples.predicted == 2 and report.triples.correct == 0


def test_prf_handles_empty_sets() -> None:
    assert PRF.of_sets(0, 0, 0).f1 == 0.0


def _tiny_gold():
    from kgbench.extraction import Gold

    return Gold(
        {("D0", 0, 3): "A", ("D0", 10, 13): "B"},
        frozenset({"A", "B"}),
        frozenset({("A", "WORKS_FOR", "B")}),
    )


def test_offline_pipeline_is_deterministic_and_scored(clean) -> None:
    _, documents, gold = clean
    first = score(gold, run_pipeline(build_offline_components(), documents))
    second = score(gold, run_pipeline(build_offline_components(), documents))
    assert first == second
    assert first.mentions.f1 > 0.9  # template text with no ambiguity is easy for the heuristic
    assert first.triples.f1 == 0.0  # it only emits RELATED_TO co-occurrence edges
    assert first.edges.f1 > 0.9
    json.dumps(first.to_dict())


def test_ambiguity_makes_the_offline_pipeline_score_lower(clean, ambiguous) -> None:
    easy = score(clean[2], run_pipeline(build_offline_components(), clean[1]))
    hard = score(ambiguous[2], run_pipeline(build_offline_components(), ambiguous[1]))
    assert hard.mentions.f1 < easy.mentions.f1
    assert hard.entities.f1 < easy.entities.f1
    assert hard.edges.precision < easy.edges.precision


def test_extract_command_prints_the_table_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "--preset", "tiny"]) == 0
    assert "Extraction vs gold" in capsys.readouterr().out
    assert main(["extract", "--preset", "tiny", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {
        "mentions",
        "entities",
        "triples",
        "edges",
        "by_predicate",
    }
