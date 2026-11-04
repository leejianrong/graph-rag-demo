"""The offline quality gate: floors are versioned, and a quality drop fails it (GRA-88)."""

from __future__ import annotations

import json

import pytest
from kgbench import gate
from kgbench.cli import main
from kgbench.gate import Failure, Setup, check, floors_from, format_report
from kggen.spec import PRESET_NAMES

TINY = Setup(preset="tiny", seed=0, questions=8, k=5, margin=0.02)


def test_floors_sit_a_margin_under_the_measured_values_and_skip_zeros() -> None:
    floors = floors_from({"a": 0.90, "b": 0.5, "zero": 0.0, "tiny": 0.015}, margin=0.02)
    assert floors == {"a": 0.88, "b": 0.48}  # nothing to protect at or below the margin


def test_a_floor_never_exceeds_the_value_it_was_taken_from() -> None:
    metrics = {"x": 0.9225589225589226, "y": 1 / 3, "z": 2 / 3}
    for margin in (0.0, 0.02):
        for metric, floor_value in floors_from(metrics, margin).items():
            assert floor_value <= metrics[metric]
    assert check(metrics, floors_from(metrics, 0.0)) == []


def test_a_metric_passes_at_its_floor_and_fails_below_it() -> None:
    assert check({"m": 0.88}, {"m": 0.88}) == []
    assert check({"m": 0.879}, {"m": 0.88}) == [Failure("m", 0.88, 0.879)]


def test_a_metric_that_was_not_measured_fails_rather_than_passing_silently() -> None:
    assert check({}, {"m": 0.5}) == [Failure("m", 0.5, None)]


def test_the_report_marks_each_metric() -> None:
    text = format_report({"good": 0.9, "bad": 0.1}, {"good": 0.8, "bad": 0.5, "gone": 0.3})
    lines = {line.split()[0]: line for line in text.splitlines()[1:]}
    assert lines["good"].endswith("ok") and lines["bad"].endswith("FAIL")
    assert "missing" in lines["gone"] and lines["gone"].endswith("FAIL")


@pytest.fixture(scope="module")
def tiny_metrics() -> dict[str, float]:
    return gate.measure(TINY)


def test_measure_is_deterministic_and_covers_every_gated_layer(tiny_metrics) -> None:
    assert gate.measure(TINY) == tiny_metrics
    kinds = {m.split("/")[0] for m in tiny_metrics}
    assert kinds == {"retrieval", "extraction", "linking"}
    systems = {m.split("/")[1] for m in tiny_metrics if m.startswith("retrieval")}
    assert systems == set(gate.GATED_SYSTEMS)
    assert all(0.0 <= v <= 1.0 for v in tiny_metrics.values())


def test_a_retrieval_regression_fails_the_gate(tiny_metrics, monkeypatch) -> None:
    floors = floors_from(tiny_metrics, margin=0.0)
    assert check(tiny_metrics, floors) == []

    from kgbench.graph_retriever import GraphRagRetriever

    monkeypatch.setattr(GraphRagRetriever, "retrieve", lambda self, question, k: [])
    failures = check(gate.measure(TINY), floors)
    assert failures and all(f.metric.startswith("retrieval/graph-rag") for f in failures)


def test_an_extraction_regression_fails_the_gate(tiny_metrics, monkeypatch) -> None:
    floors = floors_from(tiny_metrics, margin=0.0)
    from kgbench.extraction import Extraction

    monkeypatch.setattr(gate, "run_pipeline", lambda components, documents: Extraction((), ()))
    failed = {f.metric.split("/")[0] for f in check(gate.measure(TINY), floors)}
    assert {"extraction", "linking"} <= failed


def test_the_command_exits_nonzero_when_a_floor_is_missed(tmp_path, tiny_metrics, capsys) -> None:
    path = tmp_path / "gate.json"
    gate.save(path, TINY, floors_from(tiny_metrics, margin=0.0))
    assert main(["gate", "--thresholds", str(path)]) == 0
    assert "at or above their floor" in capsys.readouterr().out

    raised = {m: min(1.0, v + 0.1) for m, v in floors_from(tiny_metrics, margin=0.0).items()}
    raised["retrieval/graph-rag graph/hops=all/recall@5"] = 1.01  # unreachable
    gate.save(path, TINY, raised)
    assert main(["gate", "--thresholds", str(path)]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_update_rewrites_the_floors_from_the_current_numbers(tmp_path, tiny_metrics) -> None:
    path = tmp_path / "gate.json"
    gate.save(path, TINY, {"stale": 0.5})
    assert main(["gate", "--thresholds", str(path), "--update"]) == 0
    setup, floors = gate.load(path)
    assert setup == TINY and floors == floors_from(tiny_metrics, TINY.margin)
    assert "stale" not in floors


def test_the_committed_floors_are_well_formed() -> None:
    setup, floors = gate.load()
    data = json.loads(gate.DEFAULT_FLOORS.read_text())
    assert setup.preset in PRESET_NAMES and data["setup"]["margin"] > 0
    assert len(floors) >= 15 and all(0.0 < v <= 1.0 for v in floors.values())


def test_graph_rag_meets_the_committed_floors() -> None:
    """The gate itself: the small world, offline, under a minute, no provider."""
    setup, floors = gate.load()
    failures = check(gate.measure(setup), floors)
    assert not failures, "\n".join(
        f"{f.metric}: {f.value} is below the floor {f.floor}" for f in failures
    )
