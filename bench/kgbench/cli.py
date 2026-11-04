"""The ``kgbench`` command line.

``kgbench run --subset small`` loads a 2WikiMultihopQA dataset (the bundled mini fixture by
default, so there is no download), selects a FIXED subset, builds the graph once and scores the
subset through retrieval with **non-LLM** metrics (ADR-0009), then prints a table.

``--dataset`` points at the real corpus (outside git); ``--real`` wires the real adapter stack
(needs the Docker services, models and an API key) instead of the default offline fakes;
``--limit`` narrows the subset; ``--per-question`` also prints each question's scores. A re-run
that reuses a warm response cache and the pre-built graph costs about nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from graph_rag.wiring import build_offline_components, build_real_components
from kgbench import __version__
from kgbench.dataset import SUBSETS, load_examples, select_subset
from kgbench.harness import BenchmarkHarness

# The bundled mini fixture, so the command runs offline with no download.
_DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "wiki2_mini.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""
    parser = argparse.ArgumentParser(
        prog="kgbench",
        description="Evaluation bench for graph-rag (ADR-0011): scores a pipeline on kggen data.",
    )
    parser.add_argument("--version", action="version", version=f"kgbench {__version__}")
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="Run a subset and print the metrics.")
    run.add_argument(
        "--subset",
        default="small",
        choices=sorted(SUBSETS),
        help="Named, fixed subset to score (default: small).",
    )
    run.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help="A 2WikiMultihopQA JSON/JSONL file or directory (default: the bundled mini fixture).",
    )
    run.add_argument("--limit", type=int, help="Cap the number of questions (after the subset).")
    run.add_argument(
        "--real",
        action="store_true",
        help="Wire the real adapter stack (Docker + models + API key) instead of offline fakes.",
    )
    run.add_argument("--per-question", action="store_true", help="Also print per-question scores.")
    return parser


def _run(args: argparse.Namespace) -> int:
    examples = select_subset(load_examples(args.dataset), args.subset, limit=args.limit)
    if not examples:
        print(f"No examples selected from {args.dataset} (subset={args.subset}).")
        return 1
    components = build_real_components() if args.real else build_offline_components()
    result = BenchmarkHarness(components).run(examples, subset=args.subset)
    print(result.format_table(per_question=args.per_question))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; with no command, print help and exit 0."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 0
