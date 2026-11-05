"""Do the systems rank the same on a real dataset as on ours? (GRA-127)

Our numbers come from invented worlds. This runs BM25, dense and graph-rag on a slice of
2WikiMultihopQA, which has real Wikipedia text and human-reviewed multi-hop questions, scored the
same way: a question is solved at k when every gold supporting sentence is in the top k. The
paragraphs of all the sampled questions form one corpus (as in HippoRAG's protocol), a paragraph
is a document and each sentence is its own "fact", so no gold comes from graph-rag.

The data is fetched on request from the Hugging Face datasets server and kept outside git
(``datasets/`` is ignored); only the script and the results are committed.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from kggen.questions import Question
from kggen.render import Document, Sentence

from graph_rag.ports import Embedder
from graph_rag.wiring import build_offline_components
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.dataset import BenchmarkExample, load_examples
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import RetrievalReport, Retriever, evaluate

DATASET = "framolfese/2WikiMultihopQA"
_ROWS = "https://datasets-server.huggingface.co/rows"
SYSTEMS = ("BM25", "Dense", "graph-rag passages", "graph-rag graph")


def fetch(n: int, dest: Path, split: str = "validation") -> None:
    """Write the first ``n`` rows of a split in the 2Wiki shape ``load_examples`` reads."""
    rows: list[dict] = []
    while len(rows) < n:
        url = (
            f"{_ROWS}?dataset={DATASET}&config=default&split={split}"
            f"&offset={len(rows)}&length={min(100, n - len(rows))}"
        )
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            rows += [r["row"] for r in json.load(response)["rows"]]
    shaped = [
        {
            "_id": r["id"],
            "type": r["type"],
            "question": r["question"],
            "answer": r["answer"],
            "supporting_facts": [
                [t, i]
                for t, i in zip(
                    r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"], strict=True
                )
            ],
            "context": [
                [t, s]
                for t, s in zip(r["context"]["title"], r["context"]["sentences"], strict=True)
            ],
        }
        for r in rows[:n]
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(shaped) + "\n")


def _fact_id(title: str, index: int) -> str:
    return f"{title}#{index}"


def corpus(examples: Sequence[BenchmarkExample]) -> tuple[Document, ...]:
    """One document per distinct paragraph title; each sentence carries its own fact id."""
    docs: dict[str, Document] = {}
    for example in examples:
        for paragraph in example.context:
            if paragraph.title in docs:
                continue
            sentences, offset = [], 0
            for i, text in enumerate(paragraph.sentences):
                sentences.append(
                    Sentence(text, offset, offset + len(text), (_fact_id(paragraph.title, i),), ())
                )
                offset += len(text) + 1
            docs[paragraph.title] = Document(
                f"P{len(docs):05d}",
                paragraph.title,
                " ".join(paragraph.sentences),
                tuple(sentences),
                False,
            )
    return tuple(docs.values())


def questions(examples: Sequence[BenchmarkExample]) -> tuple[Question, ...]:
    """A question per example; its hops are the distinct paragraphs holding its gold sentences."""
    return tuple(
        Question(
            e.id,
            e.question,
            len({t for t, _ in e.supporting_facts}),
            "",
            "",
            e.answer,
            tuple(e.gold_answers),
            tuple(_fact_id(t, i) for t, i in e.supporting_facts),
        )
        for e in examples
    )


def measure(
    examples: Sequence[BenchmarkExample], embedder: Embedder, ks: Sequence[int]
) -> dict[str, RetrievalReport]:
    """All four systems on one corpus."""
    docs, qs = corpus(examples), questions(examples)
    build: dict[str, Callable[[], Retriever]] = {
        "BM25": lambda: BM25Retriever(docs),
        "Dense": lambda: DenseRetriever(docs, embedder),
        "graph-rag passages": lambda: GraphRagRetriever(
            build_offline_components(embedder=embedder), docs, mode="passages"
        ),
        "graph-rag graph": lambda: GraphRagRetriever(
            build_offline_components(embedder=embedder), docs, mode="graph"
        ),
    }
    return {name: evaluate(make(), qs, ks) for name, make in build.items()}


def load(path: Path) -> list[BenchmarkExample]:
    return sorted(load_examples(path), key=lambda e: e.id)


def format_comparison(reports: dict[str, RetrievalReport], k: int) -> str:
    """Whole-path@k per system and hop count with 95% Wilson intervals, and the ranking."""
    from kgbench.charts import wilson

    groups = [("all", lambda r: r.overall)] + [
        (f"{h} paragraphs", lambda r, h=h: r.by_hops[h])
        for h in sorted(next(iter(reports.values())).by_hops)
    ]
    lines = ["| group | n | " + " | ".join(reports) + " |", "|---|---|" + "---|" * len(reports)]
    for name, pick in groups:
        n = pick(next(iter(reports.values()))).questions
        cells = []
        for report in reports.values():
            hit = round(pick(report).path_at[k] * n)
            low, high = wilson(hit, n)
            cells.append(f"{hit / n:.0%} ({hit}/{n}, {low:.0%}–{high:.0%})")
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    order = sorted(reports, key=lambda s: -reports[s].overall.path_at[k])
    ranking = " > ".join(order)
    return "\n".join(lines) + f"\n\nRanking by whole-path@{k} over all questions: {ranking}."
