"""An offline quality gate: fail when graph-rag scores below the floors stored in the repo (GRA-88).

``measure`` runs one small world through the deterministic offline pipeline (fake embedder, no
provider) and returns a flat ``{metric: value}`` map: retrieval recall@k and path@k per hop count
for graph-rag, extraction F1, and entity-linking F1 per difficulty. ``check`` compares that map
with the floors in ``gate.json``. The floors are the measured values minus a margin, so noise
cannot trip the gate but a real drop does; ``kgbench gate --update`` rewrites them after a change
that is meant to move the numbers, which makes raising or lowering a floor a visible diff.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from kggen.render import render_documents
from kggen.spec import WorldSpec, load_preset
from kggen.world import generate

from graph_rag.wiring import build_embedder, build_offline_components
from kgbench.compare import CachingEmbedder, run_seed
from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.linking import DIFFICULTY, linking_gold, score_linking

DEFAULT_FLOORS = Path(__file__).resolve().parent / "gate.json"
GATED_SYSTEMS = ("graph-rag passages", "graph-rag graph")


@dataclass(frozen=True)
class Setup:
    preset: str = "small"
    seed: int = 0
    questions: int = 100
    k: int = 5
    margin: float = 0.02


@dataclass(frozen=True)
class Failure:
    metric: str
    floor: float
    value: float | None  # None: the metric was not measured at all


def measure(setup: Setup) -> dict[str, float]:
    spec = load_preset(setup.preset).model_copy(update={"seed": setup.seed})
    metrics: dict[str, float] = {}

    run = run_seed(
        spec,
        n_questions=setup.questions,
        embedder=CachingEmbedder(build_embedder()),
        ks=(setup.k,),
    )
    for system in GATED_SYSTEMS:
        report = run.reports[system]
        rows = [("all", report.overall)] + [(str(h), s) for h, s in sorted(report.by_hops.items())]
        for hops, s in rows:
            metrics[f"retrieval/{system}/hops={hops}/recall@{setup.k}"] = s.recall_at[setup.k]
            metrics[f"retrieval/{system}/hops={hops}/path@{setup.k}"] = s.path_at[setup.k]

    world = generate(spec)
    documents = render_documents(world)
    extraction = score(
        gold_from(world, documents), run_pipeline(build_offline_components(), documents)
    )
    for name in ("mentions", "entities", "triples", "edges"):
        metrics[f"extraction/{name}/f1"] = getattr(extraction, name).f1

    for name, overrides in DIFFICULTY.items():
        world = generate(WorldSpec.model_validate({**spec.model_dump(), **overrides}))
        documents = render_documents(world)
        linking = score_linking(
            linking_gold(world, documents),
            run_pipeline(build_offline_components(), documents),
        )
        metrics[f"linking/{name}/f1"] = linking.overall.f1
    return metrics


def floors_from(metrics: dict[str, float], margin: float) -> dict[str, float]:
    """Floors a margin under the measured values. A metric measured at 0 has nothing to protect."""
    # Rounded down, so a floor can never end up above the value it was taken from.
    return {
        m: max(0.0, math.floor(v * 10_000 - margin * 10_000) / 10_000)
        for m, v in sorted(metrics.items())
        if v > margin
    }


def check(metrics: dict[str, float], floors: dict[str, float]) -> list[Failure]:
    failures = []
    for metric, floor in sorted(floors.items()):
        value = metrics.get(metric)
        if value is None or value < floor:
            failures.append(Failure(metric, floor, value))
    return failures


def load(path: Path = DEFAULT_FLOORS) -> tuple[Setup, dict[str, float]]:
    data = json.loads(path.read_text())
    return Setup(**data["setup"]), data["floors"]


def save(path: Path, setup: Setup, floors: dict[str, float]) -> None:
    body = {"setup": setup.__dict__, "floors": floors}
    path.write_text(json.dumps(body, indent=1) + "\n")


def format_report(metrics: dict[str, float], floors: dict[str, float]) -> str:
    width = max(map(len, floors), default=0)
    lines = [f"  {'metric':<{width}}  floor  value"]
    for metric, floor in sorted(floors.items()):
        value = metrics.get(metric)
        shown = "missing" if value is None else f"{value:.3f}"
        mark = "FAIL" if value is None or value < floor else "ok"
        lines.append(f"  {metric:<{width}}  {floor:.3f}  {shown:<7} {mark}")
    return "\n".join(lines)
