"""Sweep one world knob at a time and watch where each retriever degrades (GRA-86).

For every value of a knob the same four systems as ``kgbench compare`` answer the same kind of
generated questions, over several world seeds. The report gives a table and a chart per knob, the
seed and the realised statistics of every world behind each point, and a plain verdict on where
graph-rag stops beating the baselines, or that it never did.

"Beats" means: graph-rag's graph evidence solves more whole 2-hop paths (path@k) than the better of
BM25 and dense retrieval by at least ``MARGIN``. Two hops is where the systems differ: a 1-hop
path is solved by all of them and no system solves a 3-hop or 4-hop path in the top 5, so a score
over every hop count mostly measures how many 1-hop questions there were. Gaps under ``MARGIN`` are
inside the spread between seeds. The ``hops`` knob is read at each hop count instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kggen.spec import WorldSpec
from kggen.stats import report as world_report
from kggen.world import generate

from graph_rag.ports import Embedder
from kgbench.compare import SYSTEMS, Comparison, SeedRun, run_seed
from kgbench.retrieval import DEFAULT_KS

CHALLENGER = "graph-rag graph"
BASELINES = ("BM25", "Dense")
VERDICT_HOPS = 2  # the hop count the verdict is read at
MARGIN = 0.05

# The knobs swept and the values tried by default; "hops" is read from one run's per-hop scores.
KNOBS: dict[str, tuple[float, ...]] = {
    "n_entities": (100, 300, 1000, 2000),
    "avg_degree": (2, 4, 6, 10),
    "degree_skew": (0, 0.5, 1.0, 1.5),
    "homonym_rate": (0, 0.1, 0.3, 0.5),
    "churn_rate": (0, 0.1, 0.3),
    "hops": (1, 2, 3, 4),
}
_COLOURS = {
    "BM25": "#3457d5",
    "Dense": "#2e6d62",
    "graph-rag passages": "#e0701a",
    CHALLENGER: "#b5296b",
}


@dataclass(frozen=True)
class Point:
    value: float
    comparison: Comparison
    hops: int | None  # None scores every hop count; a number scores only that one
    realised: tuple[dict, ...]  # per seed: the seed and what its world actually came out as


@dataclass(frozen=True)
class Sweep:
    knob: str
    preset: str
    base: dict  # overrides applied to the preset before the knob
    k: int
    points: tuple[Point, ...]
    seeds: tuple[int, ...] = field(default_factory=tuple)

    def path(self, point: Point, system: str, hops: int | None = None) -> float:
        """path@k over the point's hop count (every one by default), or at ``hops``."""
        return point.comparison.mean(system, point.hops if hops is None else hops, self.k, "path")

    def recall(self, point: Point, system: str) -> float:
        return point.comparison.mean(system, point.hops, self.k, "recall")

    def verdict_hops(self, point: Point) -> int:
        """The hop count the verdict reads: 2, or the point's own on the ``hops`` knob."""
        return point.hops if point.hops is not None else VERDICT_HOPS

    def margin(self, point: Point) -> float:
        """How far graph-rag's path@k is above the better baseline, at the verdict's hop count
        (negative when below)."""
        hops = self.verdict_hops(point)
        return self.path(point, CHALLENGER, hops) - max(
            self.path(point, b, hops) for b in BASELINES
        )

    def verdict(self) -> str:
        wins = [self.margin(p) >= MARGIN for p in self.points]
        values = [_fmt(p.value) for p in self.points]
        if not any(wins):
            return (
                f"graph-rag never beat the baselines on {self._metric()} at any value of "
                f"{self.knob} tried ({', '.join(values)})."
            )
        if all(wins):
            best = min(self.points, key=self.margin)
            return (
                f"graph-rag beat the baselines on {self._metric()} at every value of {self.knob} "
                f"tried; the margin was smallest at {_fmt(best.value)} "
                f"({self.margin(best):+.2f})."
            )
        first_loss = wins.index(False)
        if first_loss > 0 and not any(wins[first_loss:]):
            return (
                f"graph-rag beat the baselines on {self._metric()} up to {values[first_loss - 1]} "
                f"and stopped beating them from {values[first_loss]} on."
            )
        held = ", ".join(v for v, w in zip(values, wins, strict=True) if w)
        return f"graph-rag beat the baselines on {self._metric()} only at {held} of {self.knob}."

    def _metric(self) -> str:
        return f"path@{self.k}" if self.knob == "hops" else f"2-hop path@{self.k}"


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _spec(base: WorldSpec, knob: str, value: float, seed: int) -> WorldSpec:
    typed = int(value) if knob == "n_entities" else value
    return WorldSpec.model_validate({**base.model_dump(), knob: typed, "seed": seed})


def _realised(spec: WorldSpec) -> dict:
    stats = world_report(generate(spec))["stats"]
    return {
        "seed": spec.seed,
        "entities": stats["entities"],
        "facts": stats["facts"],
        "current_facts": stats["current_facts"],
        "degree_mean": stats["degree"]["mean"],
        "degree_max": stats["degree"]["max"],
        "largest_component": stats["components"]["largest"],
        "shared_names": stats["ambiguity"]["shared_names"],
        "replaced_facts": stats["churn"]["replaced"],
        "retracted_facts": stats["churn"]["retracted"],
    }


def run_sweep(
    base: WorldSpec,
    knob: str,
    values: Sequence[float],
    seeds: Sequence[int],
    *,
    preset: str,
    embedder: Embedder,
    embedder_name: str,
    n_questions: int = 100,
    k: int = 5,
    ks: Sequence[int] = DEFAULT_KS,
    overrides: dict | None = None,
) -> Sweep:
    """One sweep: ``base`` is the preset (with ``overrides`` already applied) the knob varies."""
    if knob not in KNOBS:
        raise ValueError(f"unknown knob {knob!r}; choose from {sorted(KNOBS)}")
    ks = tuple(sorted({*ks, k}))

    def runs_for(spec_of) -> tuple[tuple[SeedRun, ...], tuple[dict, ...]]:
        specs = [spec_of(seed) for seed in seeds]
        runs = tuple(run_seed(s, n_questions=n_questions, embedder=embedder, ks=ks) for s in specs)
        return runs, tuple(_realised(s) for s in specs)

    points: list[Point] = []
    if knob == "hops":
        runs, realised = runs_for(lambda seed: base.with_overrides(seed=seed))
        comparison = Comparison(preset, embedder_name, ks, runs)
        have = set(comparison.hop_counts())
        points = [Point(h, comparison, int(h), realised) for h in values if int(h) in have]
    else:
        for value in values:
            runs, realised = runs_for(lambda seed, v=value: _spec(base, knob, v, seed))
            points.append(Point(value, Comparison(preset, embedder_name, ks, runs), None, realised))
    return Sweep(knob, preset, overrides or {}, k, tuple(points), tuple(seeds))


def chart_svg(sweep: Sweep) -> str:
    """A line chart of path@k per system across the knob's values, as a standalone SVG."""
    w, h, left, right, top, bottom = 520, 280, 46, 150, 24, 44
    plot_w, plot_h = w - left - right, h - top - bottom
    n = len(sweep.points)

    def x(i: int) -> float:
        return left + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{sweep._metric()} against {sweep.knob}" font-family="sans-serif" '
        'font-size="11">',
        f'<rect width="{w}" height="{h}" fill="#fffdf9"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" '
            'stroke="#e2d9d0"/>'
            f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" fill="#5b5148">'
            f"{tick:.2f}</text>"
        )
    for i, p in enumerate(sweep.points):
        out.append(
            f'<text x="{x(i):.1f}" y="{h - bottom + 18}" text-anchor="middle" fill="#5b5148">'
            f"{_fmt(p.value)}</text>"
        )
    out.append(
        f'<text x="{left + plot_w / 2}" y="{h - 8}" text-anchor="middle" fill="#2a241f">'
        f"{sweep.knob}</text>"
        f'<text x="12" y="{top + plot_h / 2}" transform="rotate(-90 12 {top + plot_h / 2})" '
        f'text-anchor="middle" fill="#2a241f">{sweep._metric()}</text>'
    )
    for row, system in enumerate(SYSTEMS):
        colour = _COLOURS[system]
        pts = [
            (x(i), y(sweep.path(p, system, sweep.verdict_hops(p))))
            for i, p in enumerate(sweep.points)
        ]
        out.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="'
            + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            + '"/>'
        )
        out += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{colour}"/>' for a, b in pts]
        ly = top + 14 + 18 * row
        out.append(
            f'<line x1="{left + plot_w + 14}" x2="{left + plot_w + 34}" y1="{ly}" y2="{ly}" '
            f'stroke="{colour}" stroke-width="2"/>'
            f'<text x="{left + plot_w + 40}" y="{ly + 4}" fill="#2a241f">{system}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def to_dict(sweep: Sweep) -> dict:
    return {
        "knob": sweep.knob,
        "preset": sweep.preset,
        "overrides": sweep.base,
        "k": sweep.k,
        "seeds": list(sweep.seeds),
        "verdict": sweep.verdict(),
        "points": [
            {
                "value": p.value,
                "hops": p.hops,
                "embedder": p.comparison.embedder,
                "verdict_hops": sweep.verdict_hops(p),
                "verdict_path": {
                    s: round(sweep.path(p, s, sweep.verdict_hops(p)), 4) for s in SYSTEMS
                },
                "verdict_questions": p.comparison.questions_at(sweep.verdict_hops(p)),
                "path": {s: round(sweep.path(p, s), 4) for s in SYSTEMS},
                "path_by_hops": {
                    s: {h: round(sweep.path(p, s, h), 4) for h in p.comparison.hop_counts()}
                    for s in SYSTEMS
                },
                "recall": {s: round(sweep.recall(p, s), 4) for s in SYSTEMS},
                "margin": round(sweep.margin(p), 4),
                "worlds": list(p.realised),
            }
            for p in sweep.points
        ],
    }


def to_markdown(sweeps: Sequence[Sweep], *, command: str, chart_dir: str = ".") -> str:
    first = sweeps[0]
    k = first.k
    lines = [
        f"# Complexity sweeps: {first.preset} worlds",
        "",
        "Where does each retriever degrade as the graph gets harder? Each section changes one knob "
        "of the world, holds the rest at the base, and scores BM25, dense retrieval and graph-rag "
        f"on generated questions. path@{k} is the share of questions whose whole path is in the "
        f"top {k} sentences.",
        "",
        "## Setup",
        "",
        f"- Preset `{first.preset}`"
        + (f" with {', '.join(f'{k_}={v}' for k_, v in first.base.items())}" if first.base else "")
        + f", world seeds {list(first.seeds)}, every other knob at its base value.",
        f"- Embedder for the dense baseline and graph-rag: {first.points[0].comparison.embedder}.",
        f"- Reproduce: `{command}`",
        f"- **The verdict reads 2-hop path@{k}**, not the score over every question. A 1-hop path "
        f"is solved by every system and nobody solves a 3-hop or 4-hop path in the top {k}, so "
        "the all-hops score mostly counts how many 1-hop questions there were. (The `hops` sweep "
        "reads each hop count on its own.)",
        f"- A point is a win for graph-rag when its graph evidence beats the better baseline by "
        f"at least {MARGIN:.2f}. A smaller gap is inside the spread between seeds.",
        "",
        "## Verdicts",
        "",
        *[f"- **{s.knob}**: {s.verdict()}" for s in sweeps],
        "",
    ]
    for s in sweeps:
        label = "path" if s.knob == "hops" else "2-hop path"
        unit = "questions at that hop count" if s.knob == "hops" else "2-hop questions"
        lines += [
            f"## {s.knob}",
            "",
            f"![{label}@{k} against {s.knob}]({chart_dir}/sweep-{s.knob}.svg)",
            "",
            f"{label}@{k}, mean over seeds; n is the {unit} behind each row:",
            "",
            f"| {s.knob} | n | " + " | ".join(SYSTEMS) + " | margin |",
            "|---|---|" + "---|" * (len(SYSTEMS) + 1),
        ]
        for p in s.points:
            hops = s.verdict_hops(p)
            cells = " | ".join(f"{s.path(p, sys, hops):.2f}" for sys in SYSTEMS)
            lines.append(
                f"| {_fmt(p.value)} | {p.comparison.questions_at(hops)} | {cells} | "
                f"{s.margin(p):+.2f} |"
            )
        if s.knob != "hops":
            lines += [
                "",
                f"Over every hop count (path@{k} / recall@{k}), for reference:",
                "",
                f"| {s.knob} | n | " + " | ".join(SYSTEMS) + " |",
                "|---|---|" + "---|" * len(SYSTEMS),
            ]
            for p in s.points:
                cells = " | ".join(
                    f"{s.path(p, sys):.2f} / {s.recall(p, sys):.2f}" for sys in SYSTEMS
                )
                lines.append(f"| {_fmt(p.value)} | {p.comparison.questions_at(None)} | {cells} |")
            lines += [
                "",
                "Worlds behind each point (realised, per seed):",
                "",
                "| " + s.knob + " | seed | entities | facts | mean degree | max degree "
                "| shared names | replaced | retracted |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for p in s.points:
                for w in p.realised:
                    lines.append(
                        f"| {_fmt(p.value)} | {w['seed']} | {w['entities']} | {w['facts']} | "
                        f"{w['degree_mean']:.2f} | {w['degree_max']} | {w['shared_names']} | "
                        f"{w['replaced_facts']} | {w['retracted_facts']} |"
                    )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Templated text with made-up names; graph-rag is the offline pipeline (see the compare "
        "report's caveats). A knob that moves the result here may not move it on real prose.",
        "- Questions are plain entity questions, equal weight on each hop count, so each point "
        f"has a quarter of its questions at 2 hops. Gaps under {MARGIN:.2f} are noise; the margin "
        "column shows how far each point is from that.",
        "- Only 2-hop paths separate the systems, so the verdicts say where graph-rag keeps its "
        "advantage on that one hop count and say nothing about 3 and 4 hops, where none of them "
        f"solves a path in the top {k}.",
        "",
    ]
    return "\n".join(lines)
