"""The ``kgbench`` command line. Subcommands arrive with the slices that need them."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from kgbench import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""
    parser = argparse.ArgumentParser(
        prog="kgbench",
        description="Independent knowledge graph evaluation bench for graph-rag (ADR-0011).",
    )
    parser.add_argument("--version", action="version", version=f"kgbench {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; with no arguments, print help and exit 0."""
    parser = build_parser()
    parser.parse_args(list(argv) if argv is not None else None)
    parser.print_help()
    return 0
