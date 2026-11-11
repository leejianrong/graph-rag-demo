"""A nightly answer-quality gate: fail when EM or token-F1 drops past a tolerance (GRA-17).

``kgbench gate`` (``gate.json``) guards retrieval, extraction and linking on a generated world. It
never looks at answers. This guards the other half: the answer metrics of ``kgbench run`` (exact
match, token F1, and supporting-fact F1) on the small subset of the bundled 2Wiki mini fixture,
offline and deterministic, against the baseline in ``nightly.json``. A metric fails when it falls
more than ``tolerance`` (absolute) below its baseline. ``--update`` rewrites the baseline after a
change that is meant to move the numbers, so moving it is a visible diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent / "nightly.json"
GATED = ("exact_match", "token_f1", "supporting_fact_f1")


@dataclass(frozen=True)
class Drop:
    metric: str
    baseline: float
    value: float | None


def load(path: Path = DEFAULT_BASELINE) -> dict:
    return json.loads(path.read_text())


def save(path: Path, subset: str, tolerance: float, aggregate: dict[str, float]) -> None:
    body = {
        "subset": subset,
        "tolerance": tolerance,
        "baseline": {m: round(aggregate[m], 4) for m in GATED},
    }
    path.write_text(json.dumps(body, indent=1) + "\n")


def check(aggregate: dict[str, float], baseline: dict[str, float], tolerance: float) -> list[Drop]:
    """Metrics more than ``tolerance`` below their baseline (a missing metric always fails)."""
    drops = []
    for metric, base in sorted(baseline.items()):
        value = aggregate.get(metric)
        if value is None or value < base - tolerance - 1e-9:
            drops.append(Drop(metric, base, value))
    return drops


def summary_markdown(aggregate: dict[str, float], baseline: dict[str, float], tol: float) -> str:
    """A table for the job summary: baseline, today's value, change, verdict."""
    failed = {d.metric for d in check(aggregate, baseline, tol)}
    lines = [
        "### Nightly benchmark (small subset, offline)",
        "",
        f"Tolerance: {tol:.3f} absolute below baseline.",
        "",
        "| metric | baseline | value | change | |",
        "|---|---|---|---|---|",
    ]
    for metric, base in sorted(baseline.items()):
        value = aggregate.get(metric)
        shown = "missing" if value is None else f"{value:.4f}"
        change = "" if value is None else f"{value - base:+.4f}"
        verdict = "FAIL" if metric in failed else "ok"
        lines.append(f"| {metric} | {base:.4f} | {shown} | {change} | {verdict} |")
    return "\n".join(lines) + "\n"
