"""The ``kggen`` command line. Subcommands arrive with the slices that need them."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kggen import __version__
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import RenderSpec, render_documents
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
    questions = commands.add_parser("questions", help="Derive multi-hop questions from a world.")
    q_commands = questions.add_subparsers(dest="questions_command", required=True)
    q_gen = q_commands.add_parser("generate", help="Generate questions with gold paths as JSON.")
    _add_source(q_gen)
    q_gen.add_argument("--n", type=int, default=50, help="How many questions.")
    q_gen.add_argument("--hops", default="1:1,2:1,3:1,4:1", help="Hop mix, e.g. 1:1,2:2,3:1.")
    q_gen.add_argument("--question-seed", type=int, default=0, help="Seed for question choice.")
    q_gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
    docs = commands.add_parser("docs", help="Render a world's facts as documents.")
    d_commands = docs.add_subparsers(dest="docs_command", required=True)
    d_gen = d_commands.add_parser("generate", help="Render template documents as JSON.")
    _add_source(d_gen)
    d_gen.add_argument("--render-seed", type=int, default=0, help="Seed for wording choices.")
    d_gen.add_argument("--max-facts", type=int, default=6, help="Most facts in one document.")
    d_gen.add_argument(
        "--alias-rate", type=float, default=0.5, help="Chance a mention is an alias."
    )
    d_gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
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
    if args.command == "questions":
        hop_mix = {int(h): float(w) for h, w in (p.split(":") for p in args.hops.split(","))}
        spec = QuestionSpec(seed=args.question_seed, n_questions=args.n, hop_mix=hop_mix)
        data = [q.to_dict() for q in generate_questions(_world(args), spec)]
        text = json.dumps(data, sort_keys=True, indent=1) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return 0
    if args.command == "docs":
        spec = RenderSpec(
            seed=args.render_seed, max_facts_per_doc=args.max_facts, alias_use_rate=args.alias_rate
        )
        data = [d.to_dict() for d in render_documents(_world(args), spec)]
        text = json.dumps(data, sort_keys=True, indent=1) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return 0
    parser.print_help()
    return 0
