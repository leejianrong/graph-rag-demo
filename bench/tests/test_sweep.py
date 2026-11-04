"""Complexity sweeps: the verdict on where graph-rag stops winning, the chart, and a real run."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest
from kgbench import sweep as sw
from kgbench.cli import main
from kgbench.compare import SYSTEMS, Comparison, SeedRun
from kgbench.retrieval import HopScore, RetrievalReport

K = 5


def report(path: float, questions: int = 20) -> RetrievalReport:
    score = HopScore(questions, {K: path}, {K: path})
    return RetrievalReport((K,), score, {1: score, 2: score})


def sweep_of(margins: list[float], knob: str = "avg_degree") -> sw.Sweep:
    """A sweep whose graph-rag path@5 is ``margins[i]`` above both baselines at point i."""
    points = []
    for i, margin in enumerate(margins):
        paths = {s: 0.30 for s in SYSTEMS}
        paths[sw.CHALLENGER] = 0.30 + margin
        run = SeedRun(0, 100, 200, 50, 200, 20, {s: report(p) for s, p in paths.items()}, {})
        points.append(sw.Point(float(i + 1), Comparison("small", "fake", (K,), (run,)), None, ()))
    return sw.Sweep(knob, "small", {}, K, tuple(points), (0,))


def test_a_sweep_where_graph_rag_never_wins_says_so() -> None:
    verdict = sweep_of([-0.1, 0.0, 0.01]).verdict()
    assert "never beat" in verdict and "avg_degree" in verdict


def test_a_sweep_where_graph_rag_always_wins_names_the_smallest_margin() -> None:
    verdict = sweep_of([0.2, 0.05, 0.1]).verdict()
    assert "at every value" in verdict and "smallest at 2" in verdict and "+0.05" in verdict


def test_a_sweep_that_loses_from_some_value_on_names_where() -> None:
    verdict = sweep_of([0.2, 0.1, 0.0, -0.1]).verdict()
    assert "up to 2" in verdict and "stopped beating them from 3 on" in verdict


def test_a_sweep_that_wins_only_in_the_middle_lists_those_values() -> None:
    verdict = sweep_of([-0.1, 0.1, -0.1]).verdict()
    assert "only at 2" in verdict


def test_a_gap_under_the_margin_is_not_a_win() -> None:
    assert "never beat" in sweep_of([sw.MARGIN - 0.005]).verdict()
    assert "at every value" in sweep_of([sw.MARGIN + 0.005]).verdict()


def test_the_margin_is_measured_against_the_better_baseline() -> None:
    s = sweep_of([0.1])
    point = s.points[0]
    better = Comparison(
        "small",
        "fake",
        (K,),
        (
            SeedRun(
                0, 1, 1, 1, 1, 20, {**point.comparison.runs[0].reports, "Dense": report(0.5)}, {}
            ),
        ),
    )
    assert s.margin(sw.Point(1.0, better, None, ())) == pytest.approx(0.4 - 0.5)


def test_the_chart_is_valid_svg_with_a_line_per_system_and_a_label_per_value() -> None:
    svg = sw.chart_svg(sweep_of([0.1, 0.0, -0.1]))
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}polyline")) == len(SYSTEMS)
    labels = {t.text for t in root.iter(f"{ns}text")}
    assert {"1", "2", "3", "avg_degree", *SYSTEMS} <= labels


def test_an_unknown_knob_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown knob"):
        sw.run_sweep(None, "colour", (1,), (0,), preset="small", embedder=None, embedder_name="x")  # type: ignore[arg-type]


def run_cli(tmp_path, *extra) -> int:
    return main(
        [
            "sweep",
            "--set",
            "n_entities=120",
            "--seeds",
            "0",
            "1",
            "--questions",
            "12",
            "--out",
            str(tmp_path),
            *extra,
        ]
    )


def test_a_real_sweep_records_the_seed_and_realised_stats_of_every_point(tmp_path) -> None:
    assert run_cli(tmp_path, "--knob", "avg_degree", "--values", "2", "6") == 0
    data = json.loads((tmp_path / "sweep.json").read_text())
    assert len(data) == 1 and data[0]["knob"] == "avg_degree"
    points = data[0]["points"]
    assert [p["value"] for p in points] == [2, 6]
    for point in points:
        assert [w["seed"] for w in point["worlds"]] == [0, 1]
        assert all(w["entities"] == 120 for w in point["worlds"])
        assert set(point["path"]) == set(SYSTEMS)
    assert points[0]["worlds"][0]["degree_mean"] == pytest.approx(2.0)
    assert points[1]["worlds"][0]["degree_mean"] == pytest.approx(6.0)
    assert points[1]["worlds"][0]["facts"] == 360
    md = (tmp_path / "sweep.md").read_text()
    assert "## avg_degree" in md and "sweep-avg_degree.svg" in md and data[0]["verdict"] in md
    ET.fromstring((tmp_path / "sweep-avg_degree.svg").read_text())


def test_the_hops_sweep_reads_each_hop_count_from_one_run_and_skips_missing_ones(
    tmp_path,
) -> None:
    assert run_cli(tmp_path, "--knob", "hops", "--values", "1", "2", "9") == 0
    points = json.loads((tmp_path / "sweep.json").read_text())[0]["points"]
    assert [p["hops"] for p in points] == [1, 2]  # no 9-hop questions exist, so no point
    assert points[0]["worlds"] == points[1]["worlds"]  # one set of worlds behind every hop count
    assert points[0]["path"]["BM25"] > points[1]["path"]["BM25"]  # one hop is easier than two


def test_values_need_exactly_one_knob(tmp_path) -> None:
    assert run_cli(tmp_path, "--knob", "avg_degree", "hops", "--values", "2") == 2
