"""How much does the order documents arrive in change what the pipeline builds? (GRA-29)

Entity linking can depend on what it has already seen, and an incremental store cannot choose the
order its documents arrive in. This ingests the same corpus several times, once as generated and
then in shuffled orders, each into a fresh pipeline, and scores every run against the generator's
gold: entity linking (B-cubed), entities, triples and edges. The report gives each metric's mean,
spread and range over the orders, so a pipeline whose result depends on arrival order shows a
spread and one that does not shows none.

Scores do not depend on how a pipeline names its canonical entities, only on which mentions it
groups, so two orders that build the same clusters under different ids score the same.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kggen.render import Document
from kggen.world import World

from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.linking import CATEGORIES, linking_gold, score_linking

AS_GENERATED = None  # the order of the first run: documents as the generator wrote them


def shuffled(documents: Sequence[Document], seed: int) -> tuple[Document, ...]:
    """The same documents in an order fixed by ``seed``."""
    order = list(documents)
    random.Random(f"order:{seed}").shuffle(order)
    return tuple(order)


@dataclass(frozen=True)
class OrderRun:
    order: int | None  # the shuffle seed; None for the order the documents were generated in
    metrics: dict[str, float]


@dataclass(frozen=True)
class Spread:
    mean: float
    std: float
    low: float
    high: float

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class OrderReport:
    runs: tuple[OrderRun, ...]
    documents: int
    world_seed: int

    def metric_names(self) -> list[str]:
        return list(self.runs[0].metrics)

    def spread(self, name: str) -> Spread:
        values = [r.metrics[name] for r in self.runs]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        return Spread(mean, std, min(values), max(values))

    def order_sensitive(self, tolerance: float = 1e-9) -> list[str]:
        """The metrics that differ between orders."""
        return [m for m in self.metric_names() if self.spread(m).range > tolerance]

    def to_dict(self) -> dict:
        return {
            "world_seed": self.world_seed,
            "documents": self.documents,
            "orders": [r.order for r in self.runs],
            "metrics": {
                m: {
                    "mean": round(self.spread(m).mean, 4),
                    "std": round(self.spread(m).std, 4),
                    "min": round(self.spread(m).low, 4),
                    "max": round(self.spread(m).high, 4),
                    "per_order": [round(r.metrics[m], 4) for r in self.runs],
                }
                for m in self.metric_names()
            },
            "order_sensitive": self.order_sensitive(),
        }

    def format_table(self) -> str:
        lines = [
            f"Order sensitivity over {len(self.runs)} orders of {self.documents} documents "
            f"(world seed {self.world_seed})",
            f"  {'metric':<22} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}",
        ]
        for m in self.metric_names():
            s = self.spread(m)
            lines.append(f"  {m:<22} {s.mean:>8.3f} {s.std:>8.3f} {s.low:>8.3f} {s.high:>8.3f}")
        moved = self.order_sensitive()
        lines.append(
            "  every metric is the same in every order"
            if not moved
            else f"  differs by order: {', '.join(moved)}"
        )
        return "\n".join(lines)


def to_markdown(report: OrderReport, *, preset: str, difficulty: str, command: str) -> str:
    """The report as a page: setup, the table, and what it does and does not show."""
    moved = report.order_sensitive()
    rows = ["| metric | mean | std | min | max |", "|---|---|---|---|---|"]
    for m in report.metric_names():
        s = report.spread(m)
        rows.append(f"| {m} | {s.mean:.3f} | {s.std:.3f} | {s.low:.3f} | {s.high:.3f} |")
    verdict = (
        "Every metric came out the same in every order."
        if not moved
        else f"These metrics differed by order: {', '.join(moved)}."
    )
    return "\n".join(
        [
            f"# Ingest order sensitivity: {preset} world, {difficulty} ambiguity",
            "",
            "Entity linking can depend on what the pipeline has already seen, and incremental "
            "ingest does not control arrival order. The same corpus was ingested "
            f"{len(report.runs)} times, once as generated and {len(report.runs) - 1} times in "
            "shuffled orders, each into a fresh pipeline, and scored against the generator's gold.",
            "",
            "## Setup",
            "",
            f"- Preset `{preset}` with the `{difficulty}` ambiguity setting, world seed "
            f"{report.world_seed}, {report.documents} documents.",
            f"- Orders: as generated, then shuffle seeds {[r.order for r in report.runs[1:]]}.",
            f"- Reproduce: `{command}`",
            "",
            "## Result",
            "",
            verdict,
            "",
            *rows,
            "",
            "## What this does and does not show",
            "",
            "- This is the offline pipeline: heuristic entity detection and linking that merges by "
            "name. Merging by name does not depend on what came first, so no spread is expected "
            "here, and none is the finding. It shows the harness reports no order effect when "
            "there is none (the tests show it reports one when there is).",
            "- The real stack (spaCy, an LLM-built graph, linking against a persistent entity "
            "store) was not run. Its linking may well depend on order; run `kgbench order --real` "
            "with the services up to measure that.",
            "- Scores depend only on which mentions are grouped, not on canonical ids. Entity "
            "alignment breaks vote ties by entity id, so the scorer adds no order effect of its "
            "own (it used to: ties went to whichever mention arrived first).",
            "",
        ]
    )


def measure(
    world: World, documents: tuple[Document, ...], ingest_order, components, pipeline=run_pipeline
) -> dict:
    """Score one ingest of ``documents`` (in ``ingest_order``) against the gold of the world."""
    extraction = pipeline(components, ingest_order)
    extracted = score(gold_from(world, documents), extraction)
    linked = score_linking(linking_gold(world, documents), extraction)
    metrics = {
        "canonical entities": float(linked.clusters_predicted),
        "linking precision": linked.overall.precision,
        "linking recall": linked.overall.recall,
        "linking F1": linked.overall.f1,
    }
    for category in CATEGORIES:
        if linked.by_category[category].mentions:
            metrics[f"linking F1 ({category})"] = linked.by_category[category].f1
    metrics |= {
        "entities F1": extracted.entities.f1,
        "triples F1": extracted.triples.f1,
        "edges F1": extracted.edges.f1,
    }
    return metrics


def run_orders(
    world: World,
    documents: tuple[Document, ...],
    orders: Sequence[int],
    make_components: Callable[[], object],
    pipeline=run_pipeline,
) -> OrderReport:
    """Ingest the corpus as generated and in each shuffled order, every time into a new pipeline."""
    runs = [
        OrderRun(
            order,
            measure(
                world,
                documents,
                documents if order is AS_GENERATED else shuffled(documents, order),
                make_components(),
                pipeline,
            ),
        )
        for order in (AS_GENERATED, *orders)
    ]
    return OrderReport(tuple(runs), len(documents), world.spec.seed)
