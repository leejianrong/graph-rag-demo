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
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from kggen.llm import CachedChat, CacheMiss, OpenAICompatible, ResponseCache
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import render_documents
from kggen.spec import PRESET_NAMES, WorldSpec, load_preset
from kggen.world import generate

from graph_rag.config import Settings
from graph_rag.wiring import build_embedder, build_offline_components, build_real_components
from kgbench import __version__, gate, judge_hard
from kgbench import hard as hards
from kgbench import judge as judges
from kgbench import order as orders
from kgbench import sweep as sweeps
from kgbench.compare import CachingEmbedder, compare
from kgbench.cost import Meter
from kgbench.dataset import SUBSETS, load_examples, select_subset
from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.harness import BenchmarkHarness
from kgbench.linking import DIFFICULTY, linking_gold, score_linking

# The bundled mini fixture, so the command runs offline with no download.
_DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "wiki2_mini.json"
_REPO = Path(__file__).resolve().parents[2]
_HARD_DIR = _REPO / "datagen" / "examples" / "hard-small"
_DATA = Path(__file__).resolve().parent / "data"


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
    run.add_argument("--json", type=Path, help="Write scores, cost and latency as JSON here.")
    extract = commands.add_parser(
        "extract", help="Score extraction (mentions, entities, triples) against generated gold."
    )
    extract.add_argument("--preset", choices=PRESET_NAMES, default="small", help="World to use.")
    extract.add_argument("--seed", type=int, help="Override the preset's seed.")
    extract.add_argument("--real", action="store_true", help="Score the real adapter stack.")
    extract.add_argument("--json", action="store_true", help="Print the report as JSON.")
    cmp = commands.add_parser(
        "compare", help="Compare BM25, dense and graph-rag retrieval on generated questions."
    )
    cmp.add_argument("--preset", choices=PRESET_NAMES, default="medium", help="Base world.")
    cmp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="World seeds to run.")
    cmp.add_argument("--questions", type=int, default=200, help="Questions per seed.")
    cmp.add_argument(
        "--embedder",
        choices=("fake", "bge"),
        default="fake",
        help="fake: deterministic hashing (fast, tests). bge: the real local model (embed extra).",
    )
    cmp.add_argument("--k", type=int, default=5, help="The k for the headline table.")
    cmp.add_argument("--out", type=Path, help="Write the markdown report here.")
    cmp.add_argument("--json", type=Path, help="Write every number as JSON here.")
    order = commands.add_parser(
        "order", help="Ingest the same corpus in several shuffled orders and report the spread."
    )
    order.add_argument("--preset", choices=PRESET_NAMES, default="small", help="Base world.")
    order.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTY),
        default="ambiguous",
        help="Ambiguity setting layered over the preset (default: ambiguous).",
    )
    order.add_argument("--orders", type=int, default=6, help="Shuffled orders (at least 5).")
    order.add_argument("--real", action="store_true", help="Run the real adapter stack.")
    order.add_argument("--out", type=Path, help="Write the markdown report here.")
    order.add_argument("--json", type=Path, help="Write every number as JSON here.")
    sw = commands.add_parser(
        "sweep", help="Vary one world knob at a time and score every system at each value."
    )
    sw.add_argument(
        "--knob",
        nargs="+",
        default=["all"],
        choices=["all", *sweeps.KNOBS],
        help="Knobs to sweep (default: all).",
    )
    sw.add_argument(
        "--values", type=float, nargs="+", help="Values for a single knob (default: its usual set)."
    )
    sw.add_argument("--preset", choices=PRESET_NAMES, default="small", help="Base world.")
    sw.add_argument(
        "--set",
        dest="overrides",
        nargs="+",
        default=[],
        metavar="KNOB=VALUE",
        help="Change the base world first, e.g. n_entities=600.",
    )
    sw.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="World seeds per point.")
    sw.add_argument("--questions", type=int, default=100, help="Questions per seed.")
    sw.add_argument("--embedder", choices=("fake", "bge"), default="fake", help="As in compare.")
    sw.add_argument("--k", type=int, default=5, help="The k for path@k.")
    sw.add_argument("--out", type=Path, help="Directory for sweep.md, one SVG per knob and JSON.")
    hd = commands.add_parser(
        "hard", help="Score the committed harder LLM-written text against template text."
    )
    hd.add_argument("--dir", type=Path, default=_HARD_DIR, help="The dataset folder.")
    hd.add_argument("--questions", type=int, default=300, help="Questions for retrieval.")
    hd.add_argument("--embedder", choices=("fake", "bge"), default="fake", help="As in compare.")
    hd.add_argument("--k", type=int, default=5, help="The k for the retrieval table.")
    hd.add_argument("--out", type=Path, help="Write the markdown report here.")
    hd.add_argument("--llm-cache", type=Path, help="Also run the real LLM stages, caching here.")
    hd.add_argument("--llm-base-url", help="OpenAI-compatible endpoint (semula); omit to replay.")
    hd.add_argument("--llm-model", default="openai/claude-haiku", help="LiteLLM model string.")
    hd.add_argument(
        "--spacy", metavar="MODEL", help="Also run real spaCy NER, e.g. en_core_web_sm."
    )
    hd.add_argument("--charts", type=Path, help="Write the extraction and retrieval charts here.")
    jd = commands.add_parser(
        "judge",
        help="Measure an LLM judge's error rates on labelled open-ended cases (opt-in).",
    )
    jd.add_argument(
        "--hard",
        action="store_true",
        help="The harder cases, written by a model: other default files, and --build asks it.",
    )
    jd.add_argument("--cases", type=Path, help="The labelled cases (JSON).")
    jd.add_argument("--writer-cache", type=Path, help="--hard --build: cached writer answers.")
    jd.add_argument(
        "--build", choices=PRESET_NAMES, help="Write the cases from this preset, then exit."
    )
    jd.add_argument("--cache-dir", type=Path, help="Cached judge answers.")
    jd.add_argument("--model", default="claude-haiku", help="The model name the cache is keyed by.")
    jd.add_argument("--base-url", help="An OpenAI-compatible endpoint; omit to replay the cache.")
    jd.add_argument("--out", type=Path, help="Write the markdown report here.")
    jd.add_argument("--json", type=Path, help="Write every number as JSON here.")
    jd.add_argument("--chart", type=Path, help="Write the accuracy-per-kind chart (SVG) here.")
    gate_cmd = commands.add_parser(
        "gate", help="Fail if graph-rag scores below the stored floors (offline, no provider)."
    )
    gate_cmd.add_argument(
        "--thresholds", type=Path, default=gate.DEFAULT_FLOORS, help="Floor file."
    )
    gate_cmd.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the floors from the current numbers (after a change meant to move them).",
    )
    linking = commands.add_parser(
        "linking", help="Score entity linking (B-cubed) per difficulty setting."
    )
    linking.add_argument("--preset", choices=PRESET_NAMES, default="small", help="Base world.")
    linking.add_argument(
        "--difficulty",
        nargs="+",
        choices=sorted(DIFFICULTY),
        default=list(DIFFICULTY),
        help="Ambiguity settings layered over the preset (default: all).",
    )
    linking.add_argument("--real", action="store_true", help="Score the real adapter stack.")
    linking.add_argument("--json", action="store_true", help="Print the reports as JSON.")
    return parser


def _compare(args: argparse.Namespace) -> int:
    ks = sorted({1, 3, 5, 10, args.k})
    result = compare(
        load_preset(args.preset),
        args.seeds,
        preset=args.preset,
        n_questions=args.questions,
        real_embedder=args.embedder == "bge",
        ks=ks,
    )
    print(result.format_table(args.k))
    if args.out:
        command = (
            f"uv run --extra embed kgbench compare --preset {args.preset} "
            f"--seeds {' '.join(map(str, args.seeds))} --questions {args.questions} "
            f"--embedder {args.embedder} --k {args.k} --out {args.out}"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.to_markdown(k=args.k, command=command))
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=1) + "\n")
    return 0


def _order(args: argparse.Namespace) -> int:
    if args.orders < 5:
        print("--orders must be at least 5, or the spread means little", file=sys.stderr)
        return 2
    spec = WorldSpec.model_validate(
        {**load_preset(args.preset).model_dump(), **DIFFICULTY[args.difficulty]}
    )
    world = generate(spec)
    report = orders.run_orders(
        world,
        render_documents(world),
        range(1, args.orders + 1),
        build_real_components if args.real else build_offline_components,
    )
    print(report.format_table())
    if args.out:
        command = (
            f"uv run kgbench order --preset {args.preset} --difficulty {args.difficulty} "
            f"--orders {args.orders}{' --real' if args.real else ''} --out {args.out}"
            + (f" --json {args.json}" if args.json else "")
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            orders.to_markdown(
                report, preset=args.preset, difficulty=args.difficulty, command=command
            )
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=1) + "\n")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    knobs = list(sweeps.KNOBS) if "all" in args.knob else args.knob
    if args.values and len(knobs) != 1:
        print("--values needs exactly one --knob", file=sys.stderr)
        return 2
    try:
        overrides = dict(item.split("=", 1) for item in args.overrides)
    except ValueError:
        print("--set takes KNOB=VALUE pairs", file=sys.stderr)
        return 2
    base = load_preset(args.preset).with_overrides(**overrides)
    real = args.embedder == "bge"
    name = (
        "bge-small-en-v1.5 (local sentence-transformers)" if real else "deterministic hashing fake"
    )
    embedder = CachingEmbedder(build_embedder(real=real))
    done = []
    for knob in knobs:
        values = args.values or sweeps.KNOBS[knob]
        done.append(
            sweeps.run_sweep(
                base,
                knob,
                values,
                args.seeds,
                preset=args.preset,
                embedder=embedder,
                embedder_name=name,
                n_questions=args.questions,
                k=args.k,
                overrides=overrides,
            )
        )
        print(f"{knob}: {done[-1].verdict()}", flush=True)
    if args.out:
        extra = f" --set {' '.join(args.overrides)}" if args.overrides else ""
        command = (
            f"uv run --extra embed kgbench sweep --knob {' '.join(args.knob)} "
            f"--preset {args.preset}{extra} --seeds {' '.join(map(str, args.seeds))} "
            f"--questions {args.questions} --embedder {args.embedder} --k {args.k} "
            f"--out {args.out}"
        )
        args.out.mkdir(parents=True, exist_ok=True)
        for one in done:
            (args.out / f"sweep-{one.knob}.svg").write_text(sweeps.chart_svg(one))
        (args.out / "sweep.md").write_text(sweeps.to_markdown(done, command=command))
        (args.out / "sweep.json").write_text(
            json.dumps([sweeps.to_dict(one) for one in done], indent=1) + "\n"
        )
    return 0


def _gate(args: argparse.Namespace) -> int:
    setup, floors = gate.load(args.thresholds)
    metrics = gate.measure(setup)
    if args.update:
        gate.save(args.thresholds, setup, gate.floors_from(metrics, setup.margin))
        print(f"Wrote {len(gate.floors_from(metrics, setup.margin))} floors to {args.thresholds}")
        return 0
    print(gate.format_report(metrics, floors))
    failures = gate.check(metrics, floors)
    for f in failures:
        got = "not measured" if f.value is None else f"{f.value:.3f}"
        print(f"REGRESSION {f.metric}: {got} is below the floor {f.floor:.3f}")
    print(f"{len(floors) - len(failures)}/{len(floors)} gated metrics at or above their floor")
    return 1 if failures else 0


def _linking(args: argparse.Namespace) -> int:
    base = load_preset(args.preset).model_dump()
    reports, costs = {}, {}
    for name in args.difficulty:
        world = generate(WorldSpec.model_validate({**base, **DIFFICULTY[name]}))
        documents = render_documents(world)
        components = build_real_components() if args.real else build_offline_components()
        with Meter(components) as meter:
            extraction = run_pipeline(components, documents)
        reports[name] = score_linking(linking_gold(world, documents), extraction)
        costs[name] = meter.cost
    if args.json:
        print(
            json.dumps(
                {n: {**r.to_dict(), "cost": costs[n].to_dict()} for n, r in reports.items()},
                indent=1,
            )
        )
    else:
        for name, report in reports.items():
            print(f"[{name}] {report.format_table()}\n  cost: {costs[name].format_line()}\n")
    return 0


def _hard(args: argparse.Namespace) -> int:
    ks = sorted({1, 3, 5, 10, args.k})
    real = args.embedder == "bge"
    embedder = CachingEmbedder(build_embedder(real=real))
    settings = None
    if args.llm_cache:
        if args.llm_base_url:
            os.environ["OPENAI_API_BASE"] = args.llm_base_url  # LiteLLM reads the endpoint here
        settings = Settings(
            coref_model=args.llm_model,
            kg_build_model=args.llm_model,
            llm_cache_dir=str(args.llm_cache),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "semula-local"),
        )
    result = hards.measure(args.dir, embedder, args.questions, ks, settings, args.spacy)
    command = (
        f"uv run --extra embed kgbench hard --dir {args.dir.resolve().relative_to(_REPO)} "
        f"--questions {args.questions} --embedder {args.embedder} --k {args.k}"
        + (f" --llm-cache {args.llm_cache.resolve().relative_to(_REPO)}" if args.llm_cache else "")
        + (f" --spacy {args.spacy}" if args.spacy else "")
    )
    prefix = f"{args.out.stem}-" if args.charts and args.out else None
    text = hards.to_markdown(
        result,
        args.k,
        command,
        "bge-small-en-v1.5" if real else "hashing-fake",
        chart_dir=prefix,
    )
    print(text)
    if args.charts:
        args.charts.mkdir(parents=True, exist_ok=True)
        stem = args.out.stem if args.out else "hard"
        name = args.dir.name
        (args.charts / f"{stem}-extraction-hard.svg").write_text(
            hards.extraction_chart(result, "hard", f"{name}: extraction and linking, hard text")
        )
        (args.charts / f"{stem}-retrieval-hard.svg").write_text(
            hards.retrieval_chart(result, "hard", args.k, f"{name}: whole-path@{args.k}, hard text")
        )
    if args.out:
        args.out.write_text(text)
    return 0


def _judge(args: argparse.Namespace) -> int:
    suffix = "_hard" if args.hard else ""
    cases_path = args.cases or _DATA / f"judge_cases{suffix}.json"
    cache_dir = args.cache_dir or _DATA / ("judge-cache-hard" if args.hard else "judge-cache")
    client = OpenAICompatible(args.base_url, args.model) if args.base_url else None
    if args.build:
        world = generate(load_preset(args.build))
        spec = QuestionSpec(n_questions=12, type_mix={"open_ended": 1.0}, hop_mix={1: 1.0, 2: 1.0})
        questions = generate_questions(world, spec)
        if args.hard:
            writer = args.writer_cache or _DATA / "judge-writer-cache-hard"
            chat = CachedChat(ResponseCache(writer), client, args.model)
            try:
                cases, rejected = judge_hard.build_hard_cases(world, questions, chat)
            except CacheMiss as exc:
                raise SystemExit(f"{exc}; pass --base-url to ask an endpoint") from exc
            print(f"{rejected} writer answers rejected (ask again to replace them)")
        else:
            cases = judges.build_cases(questions)
        cases_path.write_text(json.dumps(cases, indent=1) + "\n")
        print(f"wrote {len(cases)} labelled cases to {cases_path}")
        return 0
    cases = json.loads(cases_path.read_text())
    chat = CachedChat(ResponseCache(cache_dir), client, args.model)
    try:
        report = judges.evaluate(cases, judges.llm_judge(chat))
    except CacheMiss as exc:
        raise SystemExit(f"{exc}; pass --base-url to ask an endpoint") from exc
    text = report.format_table()
    print(text)
    if args.out:
        args.out.write_text(f"# LLM judge error rates\n\nModel `{args.model}`.\n\n{text}\n")
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=1) + "\n")
    if args.chart:
        kind = "harder" if args.hard else "plain"
        args.chart.write_text(
            judges.chart_svg(report, f"Judge accuracy by case kind ({kind} cases)")
        )
    return 0


def _extract(args: argparse.Namespace) -> int:
    spec = load_preset(args.preset)
    world = generate(spec.model_copy(update={"seed": args.seed}) if args.seed is not None else spec)
    documents = render_documents(world)
    components = build_real_components() if args.real else build_offline_components()
    with Meter(components) as meter:
        extraction = run_pipeline(components, documents)
    report = score(gold_from(world, documents), extraction)
    if args.json:
        print(json.dumps({**report.to_dict(), "cost": meter.cost.to_dict()}, indent=1))
    else:
        print(report.format_table() + f"\n\ncost: {meter.cost.format_line()}")
    return 0


def _run(args: argparse.Namespace) -> int:
    examples = select_subset(load_examples(args.dataset), args.subset, limit=args.limit)
    if not examples:
        print(f"No examples selected from {args.dataset} (subset={args.subset}).")
        return 1
    components = build_real_components() if args.real else build_offline_components()
    result = BenchmarkHarness(components).run(examples, subset=args.subset)
    print(result.format_table(per_question=args.per_question))
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=1) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; with no command, print help and exit 0."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run(args)
    if args.command == "extract":
        return _extract(args)
    if args.command == "linking":
        return _linking(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "order":
        return _order(args)
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "hard":
        return _hard(args)
    if args.command == "judge":
        return _judge(args)
    if args.command == "gate":
        return _gate(args)
    parser.print_help()
    return 0
