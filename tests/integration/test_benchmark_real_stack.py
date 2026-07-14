"""Opt-in, slow, real-stack benchmark smoke (TESTING §5 gap #5, ADR-0009).

Marked ``benchmark`` — EXCLUDED from the fast gate; run explicitly with
``uv run pytest -m benchmark``. It wires the REAL adapter stack (MinIO + ES +
Neo4j + spaCy + the LLM client) via
:func:`~graph_rag.benchmark.pipeline.build_real_components` and runs the whole
ingest → graph → query → score flow over the mini fixture. This is the ONLY place
all real adapters run together, so it doubles as the whole-pipeline real-stack
smoke test.

Skips CLEANLY when the infra / models / API key are absent (no Docker services, no
spaCy model, no key) — the point is that it never breaks the offline gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_rag.benchmark.dataset import load_examples, select_subset
from graph_rag.benchmark.harness import BenchmarkHarness

pytestmark = pytest.mark.benchmark

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "wiki2_mini.json"


def test_real_stack_benchmark_smoke() -> None:
    """Full real-stack ingest → graph → query → score over the fixture (opt-in)."""
    try:
        from graph_rag.benchmark.pipeline import build_real_components

        components = build_real_components()
    except Exception as exc:  # noqa: BLE001 - any missing dependency skips cleanly.
        pytest.skip(f"real stack unavailable (infra/model/key): {exc}")

    examples = select_subset(load_examples(_FIXTURE), "small")
    result = BenchmarkHarness(components).run(examples, subset="small")

    # The pipeline produced a well-formed scoring result over the whole stack.
    assert result.num_questions == len(examples)
    assert result.num_documents > 0
    for key in (
        "exact_match",
        "token_f1",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
    ):
        assert 0.0 <= result.aggregate[key] <= 1.0
