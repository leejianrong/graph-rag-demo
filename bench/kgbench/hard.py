"""The same world as template text and as harder LLM-written text, scored side by side (GRA-115).

Everything but the prose is identical (the world, its gold, the questions), so a difference in a
score is the effect of the text. The harder text is a committed dataset (``kggen docs generate
--style rich``, in ``datagen/examples/hard-small``), so this needs no model and no endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from kggen.render import Document, render_documents
from kggen.spec import load_spec
from kggen.world import generate

from graph_rag.wiring import build_offline_components
from kgbench.compare import Comparison, run_seed
from kgbench.extraction import gold_from, run_pipeline, score
from kgbench.linking import linking_gold, score_linking

STYLES = ("template", "hard")


def load(folder: Path) -> tuple[Path, tuple[Document, ...]]:
    docs = json.loads((folder / "documents.json").read_text())
    return folder / "spec.yaml", tuple(Document.from_dict(d) for d in docs)


def measure(folder: Path, embedder, n_questions: int, ks: Sequence[int]) -> dict[str, dict]:
    """Per style: extraction, linking and the retrieval comparison for the committed world."""
    spec_path, hard = load(folder)
    spec = load_spec(spec_path)
    world = generate(spec)
    out = {}
    for style, documents in (("template", render_documents(world)), ("hard", hard)):
        extraction = run_pipeline(build_offline_components(embedder=embedder), documents)
        run = run_seed(spec, n_questions=n_questions, embedder=embedder, ks=ks, documents=documents)
        out[style] = {
            "extraction": score(gold_from(world, documents), extraction),
            "linking": score_linking(linking_gold(world, documents), extraction),
            "retrieval": Comparison("hard-small", "", tuple(ks), (run,)),
            "words": sum(len(d.text.split()) for d in documents) // len(documents),
        }
    return out


def to_markdown(result: dict[str, dict], k: int, command: str, embedder: str) -> str:
    lines = [
        "# The same world as template text and as harder LLM-written text",
        "",
        f"`{command}`; offline graph-rag, {embedder} embeddings, one world (the committed "
        "`hard-small` spec), so the differences are one sample, not a distribution.",
        "",
        "| | template | hard |",
        "|---|---|---|",
    ]
    t, h = result["template"], result["hard"]
    rows = [
        ("words per document", t["words"], h["words"]),
        ("mention F1", t["extraction"].mentions.f1, h["extraction"].mentions.f1),
        ("entity F1", t["extraction"].entities.f1, h["extraction"].entities.f1),
        ("triple F1 (strict)", t["extraction"].triples.f1, h["extraction"].triples.f1),
        ("entity pair F1 (relation ignored)", t["extraction"].edges.f1, h["extraction"].edges.f1),
        ("linking B-cubed F1", t["linking"].overall.f1, h["linking"].overall.f1),
    ]
    for name, a, b in rows:
        fmt = (lambda v: str(v)) if isinstance(a, int) else (lambda v: f"{v:.3f}")
        lines.append(f"| {name} | {fmt(a)} | {fmt(b)} |")
    for style in STYLES:
        r = result[style]["retrieval"]
        lines += ["", f"## Retrieval on {style} text", "", "```", r.format_table(k), "```"]
    return "\n".join(lines) + "\n"
