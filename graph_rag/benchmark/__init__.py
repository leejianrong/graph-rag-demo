"""V8 benchmark package (N18, U4, ADR-0009).

Measures the pipeline's multi-hop capability reproducibly at ~$0 on
2WikiMultihopQA, with **non-LLM** scoring (supporting-fact P/R/F1; answer EM /
token-F1 vs the entity ``name`` + ``aliases`` under standard normalization).

Modules:

* :mod:`~graph_rag.benchmark.metrics` — the pure, unit-testable scoring core.
* :mod:`~graph_rag.benchmark.dataset` — the 2Wiki loader + fixed named subsets (B8).
* :mod:`~graph_rag.benchmark.pipeline` — offline + real-stack pipeline wiring.
* :mod:`~graph_rag.benchmark.harness` — the ingest-once → query → score harness.
* :mod:`~graph_rag.benchmark.cli` — the ``benchmark run`` CLI (U4).
"""

from __future__ import annotations

from graph_rag.benchmark.harness import BenchmarkHarness, BenchmarkResult, QuestionScore

__all__ = ["BenchmarkHarness", "BenchmarkResult", "QuestionScore"]
