"""The benchmark CLI (U4) — ``benchmark run --subset small`` prints the metrics.

Console entry point wired in ``pyproject`` as ``benchmark`` (and runnable as
``python -m graph_rag.benchmark``). It loads a 2WikiMultihopQA dataset (the in-repo
mini fixture by default — no download), selects a FIXED subset (B8), builds the
graph once and scores the subset through V6 retrieval with **non-LLM** metrics
(ADR-0009), then prints a clean table.

``--dataset`` points at the real corpus (outside git, B8); ``--real`` wires the
real adapter stack (needs the running Docker services + models + an API key)
instead of the default offline fakes; ``--limit`` narrows the subset;
``--per-question`` also prints each question's scores. A second invocation reusing
a warm response cache + the pre-built graph is observably ~$0 (the printed
``llm calls`` drops to 0 on a re-run within one process; the real client's cache
serves re-runs across processes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_rag.benchmark.dataset import SUBSETS, load_examples, select_subset
from graph_rag.benchmark.harness import BenchmarkHarness
from graph_rag.benchmark.pipeline import build_offline_components, build_real_components

__all__ = ["main", "build_parser"]

# Default dataset: the in-repo mini fixture, so the CLI runs offline with no
# download. Real runs pass --dataset pointing at the (gitignored) corpus.
_DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "wiki2_mini.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the ``benchmark`` argument parser (``run`` sub-command)."""
    parser = argparse.ArgumentParser(prog="benchmark", description="Graph RAG benchmark harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a subset and print the metrics.")
    run.add_argument(
        "--subset",
        default="small",
        choices=sorted(SUBSETS),
        help="Named, fixed subset to score (default: small).",
    )
    run.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help="Path to a 2WikiMultihopQA JSON/JSONL file or directory "
        "(default: the in-repo mini fixture).",
    )
    run.add_argument(
        "--limit", type=int, default=None, help="Cap the number of questions (after subset)."
    )
    run.add_argument(
        "--real",
        action="store_true",
        help="Wire the real adapter stack (Docker + models + API key) instead of offline fakes.",
    )
    run.add_argument("--per-question", action="store_true", help="Also print per-question scores.")
    return parser


def _run(args: argparse.Namespace) -> int:
    """Execute the ``run`` sub-command; returns a process exit code."""
    examples = select_subset(load_examples(args.dataset), args.subset, limit=args.limit)
    if not examples:
        print(f"No examples selected from {args.dataset} (subset={args.subset}).")
        return 1

    components = build_real_components() if args.real else build_offline_components()
    harness = BenchmarkHarness(components)
    result = harness.run(examples, subset=args.subset)
    print(result.format_table(per_question=args.per_question))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return 1  # pragma: no cover - argparse enforces a valid sub-command.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
