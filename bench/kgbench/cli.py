"""The ``kgbench`` command line. Subcommands arrive with the slices that need them."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kgbench import __version__
from kgbench.spec import PRESET_NAMES, load_preset, load_spec
from kgbench.world import generate


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""
    parser = argparse.ArgumentParser(
        prog="kgbench",
        description="Independent knowledge graph evaluation bench for graph-rag (ADR-0011).",
    )
    parser.add_argument("--version", action="version", version=f"kgbench {__version__}")
    commands = parser.add_subparsers(dest="command")

    world = commands.add_parser("world", help="Work with generated ground-truth worlds.")
    world_commands = world.add_subparsers(dest="world_command", required=True)
    gen = world_commands.add_parser(
        "generate", help="Generate a world and print or save it as JSON."
    )
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESET_NAMES, help="A named complexity preset.")
    source.add_argument("--spec", type=Path, help="A WorldSpec YAML file.")
    gen.add_argument("--seed", type=int, help="Override the spec's seed.")
    gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; with no arguments, print help and exit 0."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "world":
        spec = load_preset(args.preset) if args.preset else load_spec(args.spec)
        if args.seed is not None:
            spec = spec.with_overrides(seed=args.seed)
        text = generate(spec).to_json()
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return 0
    parser.print_help()
    return 0
