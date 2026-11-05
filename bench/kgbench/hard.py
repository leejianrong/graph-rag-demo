"""The same world as template text and as harder LLM-written text, scored side by side (GRA-115).

Everything but the prose is identical (the world, its gold, the questions), so a difference in a
score is the effect of the text. The harder text is a committed dataset (``kggen docs generate
--style rich``, in ``datagen/examples/hard-small``), so this needs no model and no endpoint.
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
from kgbench.compare import Comparison, run_seed
from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.linking import linking_gold, score_linking

STYLES = ("template", "hard", "template + LLM", "hard + LLM")


def load(folder: Path) -> tuple[Path, tuple[Document, ...]]:
    docs = json.loads((folder / "documents.json").read_text())
    return folder / "spec.yaml", tuple(Document.from_dict(d) for d in docs)


def measure(
    folder: Path, embedder, n_questions: int, ks: Sequence[int], settings: Settings | None = None
) -> dict[str, dict]:
    """Per text style: extraction, linking and the retrieval comparison for the committed world.

    With ``settings`` (pointing the coref and graph-build models at an endpoint and a cache), each
    text is also run through the real LLM stages, so relations can be scored.
    """
    spec_path, hard = load(folder)
    spec = load_spec(spec_path)
    world = generate(spec)
    texts = {"template": render_documents(world), "hard": hard}
    pipelines: dict[str, Callable[[], BenchmarkComponents]] = {
        "": lambda: build_offline_components(embedder=embedder)
    }
    if settings:
        pipelines[" + LLM"] = lambda: build_offline_components(
            settings=settings, embedder=embedder, llm_stages=True
        )
    out = {}
    for suffix, build in pipelines.items():
        for style, documents in texts.items():
            extraction = run_pipeline(build(), documents)
            run = run_seed(
                spec,
                n_questions=n_questions,
                embedder=embedder,
                ks=ks,
                documents=documents,
                pipeline=build,
            )
            out[style + suffix] = {
                "extraction": score(gold_from(world, documents), extraction),
                "linking": score_linking(linking_gold(world, documents), extraction),
                "retrieval": Comparison(folder.name, "", tuple(ks), (run,)),
                "words": sum(len(d.text.split()) for d in documents) // len(documents),
            }
    return out


def typed(report) -> bool:
    """Whether the system predicted any relation other than the ``RELATED_TO`` fallback."""
    return any(p.predicted for name, p in report.by_predicate.items() if name != "RELATED_TO")


def to_markdown(result: dict[str, dict], k: int, command: str, embedder: str) -> str:
    styles = [s for s in STYLES if s in result]
    lines = [
        f"# {' vs '.join(styles)}: the same world as different text and pipelines",
        "",
        f"`{command}`; graph-rag with {embedder} embeddings, one world, so the differences are one "
        "sample, not a distribution. A `+ LLM` column runs the real coref and graph-build stages "
        "(Claude Haiku through semula, answers cached); the others use the offline heuristics.",
        "",
        "| | " + " | ".join(styles) + " |",
        "|---|" + "---|" * len(styles),
    ]
    show_triples = any(typed(result[s]["extraction"]) for s in styles)
    rows = [
        ("words per document", lambda r: str(r["words"])),
        ("mention F1", lambda r: f"{r['extraction'].mentions.f1:.3f}"),
        ("entity F1", lambda r: f"{r['extraction'].entities.f1:.3f}"),
        ("entity pair F1 (relation ignored)", lambda r: f"{r['extraction'].edges.f1:.3f}"),
        (
            "triple F1 (strict)",
            lambda r: f"{r['extraction'].triples.f1:.3f}" if typed(r["extraction"]) else "n/a",
        ),
        ("linking B-cubed F1", lambda r: f"{r['linking'].overall.f1:.3f}"),
    ]
    for name, cell in rows:
        if name.startswith("triple") and not show_triples:
            continue
        lines.append(f"| {name} | " + " | ".join(cell(result[s]) for s in styles) + " |")
    lines += [
        "",
        "`n/a`: that pipeline's graph builder emits only the `RELATED_TO` fallback, so strict "
        "triple F1 is 0 by construction.",
    ]
    for style in styles:
        r = result[style]["retrieval"]
        lines += ["", f"## Retrieval: {style}", "", "```", r.format_table(k), "```"]
    return "\n".join(lines) + "\n"
