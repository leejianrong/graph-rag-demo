# ruff: noqa: E501
"""GRA-54 spike: one bench question, in-memory stores against Postgres stores.

Builds a ``small`` kggen world, ingests its documents through ``build_offline_components`` once
with the in-memory stores and once with the Postgres ones (the ``stores=`` seam), then asks the same
multi-hop questions through graph-rag's retriever. Prints one question's ranked answer side by side,
the whole-path@5 and recall@5 per hop count, and per-query latency.

    PG_DSN=postgresql://postgres:spike@127.0.0.1:55432/spike \\
      .venv/bin/python -m spikes.pg_stores.run_bench [--questions 60] [--hops 2]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import psycopg
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import RetrievedSentence, score_retrieval
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import render_documents
from kggen.spec import load_preset
from kggen.world import generate

from graph_rag.models import QueryRequest
from graph_rag.wiring import build_offline_components
from spikes.pg_stores.pg_stores import (
    PgDocumentStore,
    PgEntityStore,
    PgGraphStore,
    PgObjectStore,
    ensure_schema,
)


def _pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--dsn", default=os.environ.get("PG_DSN", ""))
    args = ap.parse_args()

    world = generate(load_preset(args.preset))
    documents = render_documents(world)
    questions = generate_questions(world, QuestionSpec(n_questions=args.questions))
    print(
        f"world: {len(world.entities)} entities, {len(world.edges)} edges, "
        f"{len(documents)} documents, {len(questions)} questions"
    )

    conn = psycopg.connect(args.dsn, autocommit=True)
    ensure_schema(conn, 384, reset=True)
    pg_stores = (
        PgObjectStore(conn),
        PgDocumentStore(conn),
        PgEntityStore(conn),
        PgGraphStore(conn),
    )

    runs = {}
    for name, stores in (("in-memory", None), ("postgres", pg_stores)):
        components = build_offline_components(stores=stores)
        t0 = time.perf_counter()
        retriever = GraphRagRetriever(components, documents, mode="graph")
        ingest_s = time.perf_counter() - t0
        rankings: dict[str, list[RetrievedSentence]] = {}
        latency: list[float] = []
        nodes: dict[str, list[str]] = {}
        for q in questions:
            t = time.perf_counter()
            rankings[q.id] = retriever.retrieve(q.question, 5)
            latency.append((time.perf_counter() - t) * 1000)
            nodes[q.id] = [
                n.name
                for n in components.retriever.retrieve(
                    QueryRequest(question=q.question)
                ).ranked_nodes[:3]
            ]
        runs[name] = (
            score_retrieval(questions, rankings, (5,)),
            ingest_s,
            latency,
            rankings,
            nodes,
        )

    for name, (report, ingest_s, latency, _, _) in runs.items():
        print(
            f"\n== {name}: ingest {ingest_s:.1f}s; query ms p50 {statistics.median(latency):.1f} "
            f"p95 {_pct(latency, 0.95):.1f}"
        )
        for hops, score in sorted(report.by_hops.items()):
            print(
                f"  {hops}-hop n={score.questions}: path@5 {score.path_at[5]:.3f} recall@5 {score.recall_at[5]:.3f}"
            )
        print(
            f"  all: path@5 {report.overall.path_at[5]:.3f} recall@5 {report.overall.recall_at[5]:.3f}"
        )

    mem, pg = runs["in-memory"], runs["postgres"]
    same = sum(mem[3][q.id] == pg[3][q.id] for q in questions)
    same_nodes = sum(mem[4][q.id] == pg[4][q.id] for q in questions)
    print(
        f"\nidentical top-5 sentences on {same}/{len(questions)} questions; "
        f"identical top-3 ranked nodes on {same_nodes}/{len(questions)}"
    )

    two = next(q for q in questions if q.hops == 2)
    print(f"\nquestion ({two.id}, 2 hops): {two.question}\n  gold: {two.answer_name}")
    for name, run in runs.items():
        print(f"  {name}: ranked nodes {run[4][two.id]}")
        for s in run[3][two.id][:2]:
            print(f"    - {s.text[:90]}")


if __name__ == "__main__":
    main()
