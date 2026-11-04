"""B-cubed linking: merging homonyms costs precision, splitting aliases costs recall."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from kgbench.cli import main
from kgbench.extraction import Extraction, PredictedMention, gold_from, oracle, run_pipeline
from kgbench.linking import CATEGORIES, DIFFICULTY, linking_gold, score_linking
from kggen.render import render_documents
from kggen.spec import WorldSpec
from kggen.world import generate

from graph_rag.wiring import build_offline_components


def build(difficulty: str):
    spec = WorldSpec.model_validate(
        {"n_entities": 200, "avg_degree": 3, "seed": 2, **DIFFICULTY[difficulty]}
    )
    world = generate(spec)
    documents = render_documents(world)
    return world, documents, linking_gold(world, documents), oracle(gold_from(world, documents))


@pytest.fixture(scope="module")
def hard():
    return build("ambiguous")


def with_clusters(extraction: Extraction, cluster) -> Extraction:
    return replace(
        extraction,
        mentions=tuple(replace(m, canonical_id=cluster(m)) for m in extraction.mentions),
    )


def test_perfect_linking_scores_one_in_every_category(hard) -> None:
    _, _, gold, perfect = hard
    report = score_linking(gold, perfect)
    assert (report.overall.precision, report.overall.recall) == (1.0, 1.0)
    for category in CATEGORIES:
        assert report.by_category[category].mentions > 0  # the hard world has all three kinds
        assert report.by_category[category].f1 == 1.0
    assert report.clusters_predicted == report.clusters_gold


def test_merging_homonyms_lowers_precision_and_leaves_recall(hard) -> None:
    world, _, gold, perfect = hard
    names = {e.id: e.name for e in world.entities}
    merged = with_clusters(perfect, lambda m: names[m.canonical_id])  # one canonical per name
    report = score_linking(gold, merged)
    assert report.overall.precision < 1.0
    assert report.by_category["homonym"].precision < 1.0
    assert report.by_category["plain"].precision == 1.0  # nothing to confuse them with
    assert report.overall.recall == 1.0  # merging never separates a true cluster


def test_splitting_aliases_lowers_recall_and_leaves_precision(hard) -> None:
    _, documents, gold, perfect = hard
    surface = {
        (d.id, m.start, m.end): m.surface
        for d in documents
        for s in d.sentences
        for m in s.mentions
    }
    split = with_clusters(
        perfect, lambda m: f"{m.canonical_id}|{surface[(m.doc, m.start, m.end)]}"
    )  # one canonical per spelling
    report = score_linking(gold, split)
    assert report.overall.recall < 1.0
    assert report.by_category["alias"].recall < report.by_category["plain"].recall
    assert report.overall.precision == 1.0  # no cluster holds a foreign mention


def test_an_unlinked_mention_is_a_cluster_of_one(hard) -> None:
    _, _, gold, perfect = hard
    unlinked = with_clusters(perfect, lambda m: None)
    report = score_linking(gold, unlinked)
    assert report.overall.precision == 1.0
    assert report.overall.recall < 0.5
    assert report.clusters_predicted == report.overall.mentions


def test_missed_mentions_are_not_a_linking_error(hard) -> None:
    _, _, gold, perfect = hard
    half = replace(perfect, mentions=perfect.mentions[::2])
    report = score_linking(gold, half)
    assert report.overall.mentions == len(perfect.mentions[::2])
    assert (report.overall.precision, report.overall.recall) == (1.0, 1.0)
    # mentions at spans that are not gold mentions are ignored too
    stray = Extraction(perfect.mentions + (PredictedMention("D0", 9998, 9999, "x"),), ())
    assert score_linking(gold, stray).overall.mentions == len(perfect.mentions)


def test_offline_pipeline_scores_lower_as_difficulty_rises() -> None:
    f1 = {}
    for name in DIFFICULTY:
        world, documents, gold, _ = build(name)
        extraction = run_pipeline(build_offline_components(), documents)
        f1[name] = score_linking(gold, extraction)
    assert f1["clean"].overall.f1 == 1.0
    assert f1["aliases"].overall.recall < f1["clean"].overall.recall  # aliases are not linked
    assert f1["ambiguous"].by_category["homonym"].precision < 1.0  # merge-by-name merges homonyms
    assert f1["ambiguous"].overall.f1 < f1["clean"].overall.f1


def test_report_is_serialisable_and_marks_empty_categories(hard) -> None:
    world, documents, gold, perfect = build("clean")
    report = score_linking(gold, perfect)
    json.dumps(report.to_dict())
    assert "n/a" in report.format_table()  # a clean world has no homonym or alias mentions


def test_linking_command_reports_every_difficulty(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["linking", "--preset", "tiny", "--difficulty", "clean", "ambiguous"]) == 0
    out = capsys.readouterr().out
    assert "[clean]" in out and "[ambiguous]" in out and "B-cubed" in out
    assert main(["linking", "--preset", "tiny", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(DIFFICULTY)
