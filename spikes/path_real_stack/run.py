"""Whole-path@5 on real 2Wiki text through the real stores (GRA-170, outside the product).

Retrieval only. The stores must already hold the corpus that ``kgbench run --real`` ingested
over the same questions (``docs/results/real-2wiki-medium-50.md``); see the README. It scores
BM25, dense, graph-rag's supporting sentences, the plain graph view (flag off) and the path view
(flag on) on the same questions, mapping each result back to a gold sentence by document key
and sentence text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from math import sqrt
from pathlib import Path

from kgbench import real as reals
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.compare import CachingEmbedder
from kgbench.harness import _object_key

from graph_rag.config import Settings
from graph_rag.ids import document_id
from graph_rag.models import QueryRequest
from graph_rag.wiring import build_embedder, build_real_components

K = 5
NEAREST = 2  # supporting sentences listed ahead of the chain, as in ``kgbench.graph_retriever``


def wilson(hits: int, n: int, z: float = 1.96) -> str:
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{hits}/{n} = {p:.0%} ({centre - half:.0%}-{centre + half:.0%})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="The 2Wiki JSON the corpus came from."
    )
    parser.add_argument("--limit", type=int, required=True, help="The first N questions, by id.")
    args = parser.parse_args()

    examples = reals.load(args.data)[: args.limit]
    docs, questions = reals.corpus(examples), reals.questions(examples)
    embedder = CachingEmbedder(build_embedder(real=True))
    bm25, dense = BM25Retriever(docs), DenseRetriever(docs, embedder)
    sentences = {p.title: p.sentences for e in examples for p in e.context}

    plain = build_real_components(Settings(path_retrieval=False))
    chains = build_real_components(Settings(path_retrieval=True))
    title_of = {document_id(plain.bucket, _object_key(t)): t for t in sentences}

    def fact(doc_hash: str, text: str, index: int) -> str | None:
        title = title_of.get(doc_hash)
        if title is None:
            return None
        for i, sentence in enumerate(sentences[title]):
            if sentence.strip() == text.strip():
                return f"{title}#{i}"
        return f"{title}#{index}" if 0 <= index < len(sentences[title]) else None

    def retrieve(components, question: str):  # type: ignore[no-untyped-def]
        r = components.retriever.retrieve(QueryRequest(question=question))
        evidence = [fact(s.document_id, s.text, s.sentence_index) for s in r.supporting_sentences]
        edges = [
            (
                e,
                fact(
                    e.provenance.source_doc_id,
                    e.provenance.source_sentence,
                    e.provenance.sentence_index,
                ),
            )
            for e in r.subgraph.edges
        ]
        rank = {n.canonical_id: i for i, n in enumerate(r.ranked_nodes)}
        worst = len(rank)
        by_rank = [
            f
            for _, (_, f) in sorted(
                enumerate(edges),
                key=lambda ie: (
                    min(rank.get(ie[1][0].subject_id, worst), rank.get(ie[1][0].object_id, worst)),
                    ie[0],
                ),
            )
        ]
        return evidence, [f for _, f in edges], by_rank

    def top(facts: list[str | None]) -> set[str]:
        ordered = list(dict.fromkeys(f for f in facts if f is not None))
        return set(ordered[:K])

    hits: dict[str, Counter] = {}
    for q in questions:
        need = set(q.supporting_facts)
        evidence, _, _ = retrieve(plain, q.question)
        _, _, graph_view = retrieve(plain, q.question)
        evidence_on, edges_on, _ = retrieve(chains, q.question)
        got = {
            "BM25": {f for s in bm25.retrieve(q.question, K) for f in s.fact_ids},
            "Dense": {f for s in dense.retrieve(q.question, K) for f in s.fact_ids},
            "graph-rag passages": top(evidence_on),
            "graph-rag graph (flag off)": top([*graph_view, *evidence]),
            "graph-rag path": top([*evidence_on[:NEAREST], *edges_on, *evidence_on[NEAREST:]]),
        }
        for name, facts in got.items():
            counts = hits.setdefault(name, Counter())
            for group in ("all", q.hops):
                counts[(group, "ok")] += int(need <= facts)
                counts[(group, "n")] += 1

    groups = ["all", *sorted({q.hops for q in questions})]
    print("| group | " + " | ".join(hits) + " |")
    print("|---|" + "---|" * len(hits))
    for g in groups:
        cells = (wilson(c[(g, "ok")], c[(g, "n")]) for c in hits.values())
        print(f"| {g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
