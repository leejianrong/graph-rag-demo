"""The ``kggen`` command line. Subcommands arrive with the slices that need them."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kggen import __version__
from kggen.spec import PRESET_NAMES, load_preset, load_spec
from kggen.stats import report
from kggen.world import generate


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""
    parser = argparse.ArgumentParser(
        prog="kggen",
        description="Generate seeded synthetic knowledge graphs, documents and questions.",
    )
    parser.add_argument("--version", action="version", version=f"kggen {__version__}")
    commands = parser.add_subparsers(dest="command")

    world = commands.add_parser("world", help="Work with generated ground-truth worlds.")
    world_commands = world.add_subparsers(dest="world_command", required=True)
    gen = world_commands.add_parser(
        "generate", help="Generate a world and print or save it as JSON."
    )
    _add_source(gen)
    gen.add_argument("--out", type=Path, help="Write here (and a .stats.json beside it).")
    stats = world_commands.add_parser(
        "stats", help="Generate a world and print its realized statistics against the spec."
    )
    _add_source(stats)
    return parser


def _add_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESET_NAMES, help="A named complexity preset.")
    source.add_argument("--spec", type=Path, help="A WorldSpec YAML file.")
    parser.add_argument("--seed", type=int, help="Override the spec's seed.")


def _world(args: argparse.Namespace):
    spec = load_preset(args.preset) if args.preset else load_spec(args.spec)
    if args.seed is not None:
        spec = spec.with_overrides(seed=args.seed)
    return generate(spec)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; with no arguments, print help and exit 0."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "world":
        world = _world(args)
        if args.world_command == "stats":
            print(json.dumps(report(world), sort_keys=True, indent=1))
        elif args.out:
            args.out.write_text(world.to_json())
            args.out.with_suffix(".stats.json").write_text(
                json.dumps(report(world), sort_keys=True, indent=1) + "\n"
            )
        else:
            print(world.to_json(), end="")
        return 0
    parser.print_help()
    return 0
