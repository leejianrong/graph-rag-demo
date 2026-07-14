"""Fast E2E for the V8 benchmark harness + CLI — fakes only, no Docker ($0 gate).

Drives the whole benchmark over the offline, deterministic pipeline
(:func:`~graph_rag.benchmark.pipeline.build_offline_components`: heuristic
text-driven NER + KG, a :class:`~graph_rag.fakes.FakeLLMClient` coref, real EL over
in-memory ports + :class:`~graph_rag.fakes.FakeEmbedder`) against the in-repo mini
2Wiki fixture. Proves: it ingests → queries → produces metrics; the scores are
STABLE across two runs (determinism — fixed ingestion order + fakes); scoring is
non-LLM (a warm re-run reuses the pre-built graph and makes ZERO LLM calls); and
the CLI prints the metrics table. Doubles as the whole-pipeline offline smoke.

NOT marked contract/model/llm/benchmark — part of the fast, offline gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_rag.benchmark.cli import main as cli_main
from graph_rag.benchmark.dataset import load_examples, select_subset
from graph_rag.benchmark.harness import BenchmarkHarness
from graph_rag.benchmark.pipeline import build_offline_components

pytestmark = pytest.mark.e2e

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "wiki2_mini.json"


@pytest.fixture
def examples() -> list:
    """The fixed ``small`` subset of the mini 2Wiki fixture (deterministic order)."""
    return select_subset(load_examples(_FIXTURE), "small")


def test_harness_ingests_queries_and_produces_metrics(examples: list) -> None:
    """The harness runs end-to-end and produces a populated metrics result."""
    harness = BenchmarkHarness(build_offline_components())

    result = harness.run(examples, subset="small")

    assert result.num_questions == len(examples) == 5
    assert result.num_documents > 0  # the corpus graph was built
    assert len(result.per_question) == result.num_questions
    # Every metric key is present and in range.
    for key in (
        "exact_match",
        "token_f1",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
    ):
        assert 0.0 <= result.aggregate[key] <= 1.0
    # The offline pipeline answers at least one question and retrieves gold evidence.
    assert result.aggregate["exact_match"] > 0.0
    assert result.aggregate["supporting_fact_recall"] > 0.0


def test_scores_are_stable_across_runs_and_rerun_is_free(examples: list) -> None:
    """Determinism + ~$0 re-run: identical scores, and the warm re-run makes 0 LLM calls."""
    harness = BenchmarkHarness(build_offline_components())

    first = harness.run(examples, subset="small")
    second = harness.run(examples, subset="small")

    # Deterministic: fixed ingestion order + fakes -> byte-identical aggregates.
    assert first.aggregate == second.aggregate
    assert [s.as_metrics() for s in first.per_question] == [
        s.as_metrics() for s in second.per_question
    ]

    # The FIRST run pays the ingestion LLM cost (coref, via the FakeLLMClient);
    # the SECOND reuses the pre-built graph and runs only non-LLM retrieval, so it
    # is observably ~$0 (ADR-0009).
    assert first.llm_calls > 0
    assert second.llm_calls == 0


def test_two_independent_harnesses_agree(examples: list) -> None:
    """Reproducibility across processes: a fresh pipeline yields the same scores."""
    first = BenchmarkHarness(build_offline_components()).run(examples, subset="small")
    second = BenchmarkHarness(build_offline_components()).run(examples, subset="small")
    assert first.aggregate == second.aggregate


def test_cli_prints_metrics_table(capsys: pytest.CaptureFixture[str]) -> None:
    """`benchmark run` (via main()) on the fixture prints the metrics table, exit 0."""
    exit_code = cli_main(["run", "--subset", "small", "--dataset", str(_FIXTURE)])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "Benchmark results" in printed
    assert "Supporting-fact retrieval" in printed
    assert "exact match" in printed
    assert "token f1" in printed
