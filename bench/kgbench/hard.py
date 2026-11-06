"""The same world as template and as harder LLM-written text, through several pipelines (GRA-115).

Everything but the prose and the pipeline is identical (the world, its gold, the questions), so a
difference in a score is the effect of the text or of the stage swapped in. The harder text is a
committed dataset (``kggen docs generate --style rich``, in ``datagen/examples/hard-*``), and the
LLM stages answer from a committed cache, so a replay needs no model and no endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from kggen.render import Document, render_documents
from kggen.spec import load_spec
from kggen.world import generate

from graph_rag.config import Settings
from graph_rag.wiring import BenchmarkComponents, build_offline_components
from kgbench.charts import grouped_bars, wilson
from kgbench.compare import Comparison, run_seed
from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.linking import linking_gold, score_linking

TEXTS = ("template", "hard")


def load(folder: Path) -> tuple[Path, tuple[Document, ...]]:
    docs = json.loads((folder / "documents.json").read_text())
    return folder / "spec.yaml", tuple(Document.from_dict(d) for d in docs)


def pipeline_builders(
    embedder,
    settings: Settings | None = None,
    spacy_model: str | None = None,
    wanted: Sequence[str] | None = None,
) -> dict[str, Callable[[], BenchmarkComponents]]:
    """Builders for the pipelines to run, keyed by name suffix (``""`` is the offline one)."""

    def pipeline(llm: bool, spacy: bool) -> Callable[[], BenchmarkComponents]:
        base = settings or Settings()
        used = base.model_copy(update={"ner_model": spacy_model}) if spacy and spacy_model else base
        return lambda: build_offline_components(
            settings=used, embedder=embedder, llm_stages=llm, spacy_ner=spacy
        )

    pipelines = {"": pipeline(False, False)}
    if spacy_model:
        pipelines[" + spaCy"] = pipeline(False, True)
    if settings:
        pipelines[" + LLM"] = pipeline(True, False)
    if settings and spacy_model:
        pipelines[" + LLM + spaCy"] = pipeline(True, True)
    if wanted:
        pipelines = {k: v for k, v in pipelines.items() if _key(k) in wanted}
    return pipelines


def measure(
    folder: Path,
    embedder,
    n_questions: int,
    ks: Sequence[int],
    settings: Settings | None = None,
    spacy_model: str | None = None,
    texts_wanted: Sequence[str] = TEXTS,
    pipelines_wanted: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Per ``text + pipeline``: extraction, linking and the retrieval comparison.

    ``settings`` (the coref and graph-build models pointed at an endpoint and a cache) adds the
    real LLM stages; ``spacy_model`` adds real spaCy NER; both add the two together. The
    ``*_wanted`` arguments leave texts or pipelines out (the LLM ones cost a prompt each).
    """
    spec_path, hard = load(folder)
    spec = load_spec(spec_path)
    world = generate(spec)
    texts = {
        t: docs
        for t, docs in (("template", render_documents(world)), ("hard", hard))
        if t in texts_wanted
    }

    pipelines = pipeline_builders(embedder, settings, spacy_model, pipelines_wanted)
    out = {}
    for suffix, build in pipelines.items():
        for text, documents in texts.items():
            extraction = run_pipeline(build(), documents)
            run = run_seed(
                spec,
                n_questions=n_questions,
                embedder=embedder,
                ks=ks,
                documents=documents,
                pipeline=build,
            )
            out[text + suffix] = {
                "text": text,
                "pipeline": suffix.removeprefix(" + ") or "offline",
                "extraction": score(gold_from(world, documents), extraction),
                "linking": score_linking(linking_gold(world, documents), extraction),
                "retrieval": Comparison(folder.name, "", tuple(ks), (run,)),
                "words": sum(len(d.text.split()) for d in documents) // len(documents),
            }
    return out


PIPELINES = ("offline", "spacy", "llm", "llm+spacy")


def _key(suffix: str) -> str:
    return suffix.removeprefix(" + ").replace(" + ", "+").lower() or "offline"


def typed(report) -> bool:
    """Whether the system predicted any relation other than the ``RELATED_TO`` fallback."""
    return any(p.predicted for name, p in report.by_predicate.items() if name != "RELATED_TO")


def _metrics(r: dict) -> dict[str, float]:
    e = r["extraction"]
    return {
        "mention F1": e.mentions.f1,
        "entity F1": e.entities.f1,
        "entity-pair F1": e.edges.f1,
        "triple F1*": e.triples.f1 if typed(e) else 0.0,
        "linking F1": r["linking"].overall.f1,
    }


def extraction_chart(result: dict[str, dict], text: str, title: str) -> str:
    """Extraction and linking scores of every pipeline on one text."""
    rows = {r["pipeline"]: _metrics(r) for r in result.values() if r["text"] == text}
    names = list(next(iter(rows.values())))
    return grouped_bars(
        title, names, {p: [m[n] for n in names] for p, m in rows.items()}, ylabel="F1 against gold"
    )


def retrieval_chart(result: dict[str, dict], text: str, k: int, title: str) -> str:
    """Whole-path@k per hop count: BM25, dense and graph-rag under each pipeline, with intervals."""
    picked = {r["pipeline"]: r["retrieval"] for r in result.values() if r["text"] == text}
    first = next(iter(picked.values()))
    hops = first.hop_counts()
    ns = [first.questions_at(h) for h in hops]

    def column(comparison: Comparison, system: str):
        values = [comparison.mean(system, h, k, "path") for h in hops]
        return values, [wilson(round(v * n), n) for v, n in zip(values, ns, strict=True)]

    series, intervals = {}, {}
    for name, system in (("BM25", "BM25"), ("Dense", "Dense")):
        series[name], intervals[name] = column(first, system)
    for pipeline, comparison in picked.items():
        label = f"graph-rag ({pipeline})"
        series[label], intervals[label] = column(comparison, "graph-rag graph")
    return grouped_bars(
        title,
        [f"{h}-hop" for h in hops],
        series,
        ylabel=f"whole-path@{k}",
        intervals=intervals,
        counts=[f"n={n}" for n in ns],
    )


def to_markdown(
    result: dict[str, dict], k: int, command: str, embedder: str, chart_dir: str | None = None
) -> str:
    columns = list(result)
    lines = [
        f"# {Path(command.split('--dir ')[-1].split()[0]).name}: one world, two texts, "
        "several pipelines",
        "",
        f"`{command}`; graph-rag with {embedder} embeddings, one world, so the differences are one "
        "sample, not a distribution. `LLM` is the real coref and graph-build stages (Claude Haiku "
        "through semula, answers cached); `spaCy` is the real NER stage; the rest are the offline "
        "heuristics.",
        "",
        "| | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    lines.append(
        "| words per document | " + " | ".join(str(result[c]["words"]) for c in columns) + " |"
    )
    for name in _metrics(result[columns[0]]):
        label = "triple F1 (strict)*" if name == "triple F1*" else name
        cells = " | ".join(_cell(result[c], name) for c in columns)
        lines.append(f"| {label} | {cells} |")
    lines += [
        "",
        "\\* `n/a`: that pipeline's graph builder emits only the `RELATED_TO` fallback, so strict "
        "triple F1 is 0 by construction (drawn as 0 in the charts).",
    ]
    if chart_dir is not None:
        lines += [
            "",
            f"![extraction and linking, hard text]({chart_dir}extraction-hard.svg)",
            "",
            f"![whole-path@{k} by hop count, hard text]({chart_dir}retrieval-hard.svg)",
        ]
    for column in columns:
        r = result[column]["retrieval"]
        lines += ["", f"## Retrieval: {column}", "", "```", r.format_table(k), "```"]
    return "\n".join(lines) + "\n"


def _cell(r: dict, name: str) -> str:
    if name == "triple F1*" and not typed(r["extraction"]):
        return "n/a"
    return f"{_metrics(r)[name]:.3f}"
