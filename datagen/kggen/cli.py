"""The ``kggen`` command line. Subcommands arrive with the slices that need them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from kggen import __version__
from kggen.export import export_archive
from kggen.fixtures import write as write_fixtures
from kggen.llm import CachedChat, CacheMiss, OpenAICompatible, ResponseCache
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import RenderSpec, render_documents
from kggen.render_llm import InvalidRendering, render_documents_llm
from kggen.scenarios import ScenarioSpec, generate_scenario
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
    q_gen.add_argument(
        "--types",
        default="entity:1",
        help="Question type mix, e.g. entity:2,set:1,count:1,compare:1,superlative:1.",
    )
    q_gen.add_argument("--question-seed", type=int, default=0, help="Seed for question choice.")
    q_gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
    scenario = commands.add_parser(
        "scenario", help="Update, delete and reorder scenarios with the expected final graph."
    )
    sc_commands = scenario.add_subparsers(dest="scenario_command", required=True)
    sc_gen = sc_commands.add_parser("generate", help="Emit events, arrival orders and the result.")
    _add_source(sc_gen)
    sc_gen.add_argument("--scenario-seed", type=int, default=0, help="Seed for the scenario.")
    sc_gen.add_argument("--updates", type=int, default=5, help="Facts whose object changes.")
    sc_gen.add_argument("--deletes", type=int, default=5, help="Facts that are forgotten.")
    sc_gen.add_argument("--orders", type=int, default=3, help="Arrival orders to generate.")
    sc_gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
    docs = commands.add_parser("docs", help="Render a world's facts as documents.")
    d_commands = docs.add_subparsers(dest="docs_command", required=True)
    d_gen = d_commands.add_parser("generate", help="Render template documents as JSON.")
    _add_source(d_gen)
    d_gen.add_argument("--render-seed", type=int, default=0, help="Seed for wording choices.")
    d_gen.add_argument("--max-facts", type=int, default=6, help="Most facts in one document.")
    d_gen.add_argument(
        "--alias-rate", type=float, default=0.5, help="Chance a mention is an alias."
    )
    d_gen.add_argument(
        "--words", type=int, default=0, help="Words per document (e.g. 500 or 5000); 0 is short."
    )
    d_gen.add_argument(
        "--context-facts", type=int, default=3, help="Long documents: facts about neighbours."
    )
    d_gen.add_argument(
        "--renderer", choices=("template", "llm"), default="template", help="Who writes the text."
    )
    d_gen.add_argument("--cache-dir", type=Path, help="llm: where answers are cached (required).")
    d_gen.add_argument(
        "--base-url", help="llm: an OpenAI-compatible endpoint; omit to use only the cache."
    )
    d_gen.add_argument("--model", help="llm: the model name (required).")
    d_gen.add_argument("--out", type=Path, help="Write here instead of stdout.")
    export = commands.add_parser(
        "export", help="Write a reproducible dataset zip: world, questions, documents and specs."
    )
    export.add_argument("--spec", type=Path, required=True, help="A WorldSpec file (JSON or YAML).")
    export.add_argument("--questions-spec", type=Path, help="A QuestionSpec JSON file.")
    export.add_argument("--render-spec", type=Path, help="A RenderSpec JSON file.")
    export.add_argument("--seed", type=int, help="Override the world spec's seed.")
    export.add_argument("--out", type=Path, required=True, help="The zip to write.")
    fixtures = commands.add_parser("fixtures", help="The committed small and medium fixture sets.")
    f_commands = fixtures.add_subparsers(dest="fixtures_command", required=True)
    f_write = f_commands.add_parser("write", help="(Re)generate the fixture files.")
    f_write.add_argument("--out", type=Path, help="Directory to write (default: datagen/fixtures).")
    ui = commands.add_parser("ui", help="Start the web UI (needs the kggen[ui] extra).")
    ui.add_argument("--port", type=int, help="Port to serve on (default: first free).")
    ui.add_argument("--host", default="127.0.0.1", help="Address to bind.")
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


def _render_llm(args: argparse.Namespace, spec: RenderSpec):
    """LLM rendering: the key comes from KGGEN_LLM_API_KEY, never from a flag."""
    if not (args.cache_dir and args.model):
        raise SystemExit("--renderer llm needs --cache-dir and --model")
    client = (
        OpenAICompatible(args.base_url, args.model, os.environ.get("KGGEN_LLM_API_KEY"))
        if args.base_url
        else None
    )
    chat = CachedChat(ResponseCache(args.cache_dir), client, args.model)
    try:
        docs = render_documents_llm(_world(args), chat, spec)
    except (CacheMiss, InvalidRendering) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"llm renderer: {chat.calls} endpoint calls, {chat.hits} cache hits", file=sys.stderr)
    return docs


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
        type_mix = {t: float(w) for t, w in (p.split(":") for p in args.types.split(","))}
        spec = QuestionSpec(
            seed=args.question_seed, n_questions=args.n, hop_mix=hop_mix, type_mix=type_mix
        )
        data = [q.to_dict() for q in generate_questions(_world(args), spec)]
        text = json.dumps(data, sort_keys=True, indent=1) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return 0
    if args.command == "fixtures":
        write_fixtures(args.out) if args.out else write_fixtures()
        return 0
    if args.command == "scenario":
        scenario_spec = ScenarioSpec(
            seed=args.scenario_seed,
            n_updates=args.updates,
            n_deletes=args.deletes,
            n_orders=args.orders,
        )
        text = json.dumps(
            generate_scenario(_world(args), scenario_spec).to_dict(), sort_keys=True, indent=1
        )
        if args.out:
            args.out.write_text(text + "\n")
        else:
            print(text)
        return 0
    if args.command == "docs":
        render_spec = RenderSpec(
            seed=args.render_seed,
            max_facts_per_doc=args.max_facts,
            alias_use_rate=args.alias_rate,
            target_words=args.words,
            neighbour_facts=args.context_facts,
        )
        if args.renderer == "llm":
            docs = _render_llm(args, render_spec)
        else:
            docs = render_documents(_world(args), render_spec)
        data = [d.to_dict() for d in docs]
        text = json.dumps(data, sort_keys=True, indent=1) + "\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text, end="")
        return 0
    if args.command == "export":
        world_spec = load_spec(args.spec)
        if args.seed is not None:
            world_spec = world_spec.with_overrides(seed=args.seed)
        questions = (
            QuestionSpec.model_validate_json(args.questions_spec.read_text())
            if args.questions_spec
            else QuestionSpec()
        )
        render = (
            RenderSpec.model_validate_json(args.render_spec.read_text())
            if args.render_spec
            else RenderSpec()
        )
        args.out.write_bytes(export_archive(generate(world_spec), questions, render))
        return 0
    if args.command == "ui":
        try:
            from kggen_ui.app import serve
        except ImportError:
            print("The UI needs indah: install it with `pip install 'kggen[ui]'`.")
            return 1
        serve(port=args.port, host=args.host)
        return 0
    parser.print_help()
    return 0
