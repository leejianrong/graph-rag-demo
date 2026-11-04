"""Order sensitivity: the same corpus ingested in several orders, and what changes (GRA-29)."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest
from kgbench.cli import main
from kgbench.extraction import Extraction, PredictedMention, gold_from, oracle, score
from kgbench.order import OrderReport, run_orders, shuffled
from kggen.render import render_documents
from kggen.spec import WorldSpec
from kggen.world import generate

from graph_rag.wiring import build_offline_components


@pytest.fixture(scope="module")
def corpus():
    world = generate(
        WorldSpec(n_entities=120, avg_degree=4, seed=5, alias_rate=0.5, homonym_rate=0.1)
    )
    return world, render_documents(world)


def invariant_pipeline(world, documents):
    """A perfect pipeline: the same clusters whatever order documents arrive in."""
    gold = gold_from(world, documents)

    def run(components, ordered):
        return oracle(gold)

    return run


def first_come_pipeline(world, documents):
    """A pipeline whose linking depends on arrival order: the first document's mentions all merge
    into one entity, so which document comes first decides which entities are wrongly merged."""
    gold = gold_from(world, documents)

    def run(components, ordered):
        first = ordered[0].id
        mentions = tuple(
            replace(m, canonical_id="MERGED") if m.doc == first else m
            for m in oracle(gold).mentions
        )
        return Extraction(mentions, ())

    return run


def test_shuffling_keeps_every_document_and_is_fixed_by_the_seed(corpus) -> None:
    _, documents = corpus
    a, again, other = shuffled(documents, 1), shuffled(documents, 1), shuffled(documents, 2)
    assert a == again and a != other and a != documents
    assert sorted(d.id for d in a) == sorted(d.id for d in documents)


def test_an_order_independent_pipeline_scores_the_same_in_every_order(corpus) -> None:
    world, documents = corpus
    report = run_orders(
        world,
        documents,
        range(1, 7),
        build_offline_components,
        invariant_pipeline(world, documents),
    )
    assert len(report.runs) == 7 and report.runs[0].order is None
    assert report.order_sensitive() == []
    assert (
        report.spread("linking F1").std == 0 and "every metric is the same" in report.format_table()
    )


def test_an_order_dependent_pipeline_shows_a_spread(corpus) -> None:
    world, documents = corpus
    report = run_orders(
        world,
        documents,
        range(1, 7),
        build_offline_components,
        first_come_pipeline(world, documents),
    )
    assert "linking F1" in report.order_sensitive()
    spread = report.spread("linking F1")
    assert spread.range > 0 and spread.low <= spread.mean <= spread.high and spread.std > 0
    assert "differs by order" in report.format_table()


def test_the_report_lists_every_orders_value_and_which_metrics_move(corpus) -> None:
    world, documents = corpus
    report = run_orders(
        world,
        documents,
        range(1, 6),
        build_offline_components,
        first_come_pipeline(world, documents),
    )
    data = report.to_dict()
    assert data["orders"] == [None, 1, 2, 3, 4, 5]
    assert len(data["metrics"]["linking F1"]["per_order"]) == 6
    assert "linking F1" in data["order_sensitive"]
    assert isinstance(report, OrderReport)


def test_entity_scores_do_not_depend_on_the_order_mentions_arrive_in(corpus) -> None:
    """A cluster that merges two gold entities ties on votes. Another cluster holds only one of
    them, so which way the tie goes changes how many entities were found; it must not depend on
    which mention came first."""
    world, documents = corpus
    gold = gold_from(world, documents)
    by_entity: dict[str, list] = {}
    for span in sorted(gold.mentions):
        by_entity.setdefault(gold.mentions[span], []).append(span)
    a = next(e for e, spans in by_entity.items() if len(spans) >= 2)
    b = next(e for e, spans in by_entity.items() if len(spans) >= 3 and e != a)
    clusters = [(s, "ONE") for s in by_entity[a][:2] + by_entity[b][:2]]
    clusters += [(s, "TWO") for s in by_entity[b][2:3]]
    scores = set()
    for seed in range(12):
        order = clusters[:]
        random.Random(seed).shuffle(order)
        mentions = tuple(PredictedMention(d, x, y, c) for (d, x, y), c in order)
        scores.add(round(score(gold, Extraction(mentions, ())).entities.f1, 9))
    assert len(scores) == 1


def test_the_command_runs_the_offline_pipeline_and_writes_json(tmp_path, capsys) -> None:
    out = tmp_path / "order.json"
    assert main(["order", "--preset", "small", "--orders", "5", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["orders"]) == 6 and "linking F1" in data["metrics"]
    assert "Order sensitivity over 6 orders" in capsys.readouterr().out


def test_fewer_than_five_orders_is_refused() -> None:
    assert main(["order", "--orders", "3"]) == 2
