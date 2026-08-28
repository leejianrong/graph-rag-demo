#!/usr/bin/env python3
"""Bulk-import the eval-bench corpus into the real Graph RAG stack.

``eval-bench`` (a separate LLM-agent benchmark harness) ships a CSV corpus of
``(url, article, ...)`` rows. Loading it needs a bulk path that ingests many
documents in-process and synchronously — ``POST /ingest`` is unsuitable: it only
writes bytes to the object store and publishes a Kafka trigger, returning before
the actual pipeline runs (which happens later, off the Kafka consumer). Instead
this script writes each row's article bytes to the object store the same way
``graph_rag.demo.run_offline`` does for the bundled demo corpus, then calls
:meth:`~graph_rag.orchestrator.Orchestrator.process_document` directly — the
real per-document pipeline entrypoint — against a real, fully-wired
``Orchestrator`` (the same composition ``graph_rag.main.main`` builds, minus the
Kafka consumer / FastAPI server this script doesn't need).

The row's ``url`` is threaded through as
:attr:`~graph_rag.models.IngestTrigger.source_url` so the opt-in
:class:`~graph_rag.adapters.eval_bench_export.EvalBenchExportStage` (wired in only
when ``Settings.eval_bench_export_enabled`` is true) can compute its
``sha256(url)`` export identity.

Usage::

    uv run python scripts/import_eval_bench_corpus.py --csv path/to/corpus.csv
    uv run python scripts/import_eval_bench_corpus.py --csv path/to/corpus.csv --limit 5

Needs the real stack up (``make up``) and ``EVAL_BENCH_EXPORT_ENABLED=true`` (plus
an ``OPENAI_API_KEY`` — coref + KG-build call an LLM) in ``.env``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from graph_rag.adapters.embedder import SentenceTransformerEmbedder
from graph_rag.adapters.es_document_store import EsDocumentStore
from graph_rag.adapters.es_entity_store import EsEntityStore
from graph_rag.adapters.eval_bench_export import EvalBenchExportStage
from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.adapters.minio_object_store import MinioObjectStore
from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore
from graph_rag.config import Settings
from graph_rag.logging import configure_logging, get_logger
from graph_rag.models import IngestTrigger
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.kg_build import KgBuildStage
from graph_rag.stages.ner import SpacyNerStage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from graph_rag.ports import ObjectStore

__all__ = ["main", "build_parser"]

_logger = get_logger(__name__)

# Object-store key prefix for the imported corpus (this project's own object
# identity — decoupled from the eval-bench sha256(url) identity, see
# EvalBenchExportStage's module docstring).
_OBJECT_KEY_PREFIX = "eval-bench-corpus"


def _build_orchestrator(settings: Settings) -> tuple[Orchestrator, ObjectStore]:
    """Wire the SAME real adapter stack ``graph_rag.main.main`` composes.

    Returns ``(orchestrator, object_store)`` — the caller needs the object store
    directly to ``put`` each row's article bytes before triggering ingestion.
    Mirrors ``main.py``'s composition exactly (same adapters, same
    ``from_settings`` calls) but omits the Kafka consumer / FastAPI app / query
    retriever, none of which this offline bulk-import path uses.
    """
    object_store = MinioObjectStore.from_settings(settings)
    document_store = EsDocumentStore.from_settings(settings)
    document_store.ensure_index()

    ner_stage = SpacyNerStage.from_settings(settings)

    llm_client = LiteLLMClient.from_settings(settings)
    coref_stage = LLMCorefStage(llm_client)

    embedder = SentenceTransformerEmbedder.from_settings(settings)
    entity_store = EsEntityStore.from_settings(settings)
    entity_store.ensure_index()
    entity_linking_stage = EntityLinkingStage.from_settings(settings, entity_store, embedder)

    graph_store = Neo4jGraphStore.from_settings(settings)
    graph_store.init()
    kg_build_stage = KgBuildStage.from_settings(settings)

    if not settings.eval_bench_export_enabled:
        print(
            "EVAL_BENCH_EXPORT_ENABLED is not set to true in .env — this script's whole\n"
            "point is the eval-bench dual-write, so set it (and re-run `make up` if the\n"
            "app container needs the new env var) before importing the corpus.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    eval_bench_export = EvalBenchExportStage.from_settings(settings)
    eval_bench_export.ensure_index()

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner_stage,
        coref_stage=coref_stage,
        entity_linking_stage=entity_linking_stage,
        graph_store=graph_store,
        kg_build_stage=kg_build_stage,
        eval_bench_export=eval_bench_export,
    )
    return orchestrator, object_store


def _iter_rows(csv_path: Path, *, limit: int | None) -> Iterator[dict[str, str]]:
    """Yield ``{url, article, ...}`` rows from the corpus CSV, up to ``limit``."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                return
            yield row


def run(csv_path: Path, *, limit: int | None = None) -> int:
    """Import ``csv_path``'s corpus into the real stack. Returns a process exit code."""
    settings = Settings()
    configure_logging(settings.log_level)

    orchestrator, object_store = _build_orchestrator(settings)

    imported = 0
    dropped = 0
    skipped = 0
    started = time.monotonic()
    for i, row in enumerate(_iter_rows(csv_path, limit=limit)):
        url = (row.get("url") or "").strip()
        article = row.get("article") or ""
        if not url or not article.strip():
            _logger.warning("row %d: missing url/article, skipping", i)
            skipped += 1
            continue

        object_key = f"{_OBJECT_KEY_PREFIX}/{i:06d}.txt"
        object_store.put(settings.minio_bucket, object_key, article.encode("utf-8"))
        result = orchestrator.process_document(
            IngestTrigger(bucket=settings.minio_bucket, object_key=object_key, source_url=url)
        )
        if result is None:
            _logger.warning("row %d: dropped (see the error logged above) url=%s", i, url)
            dropped += 1
            continue
        imported += 1
        if imported % 10 == 0 or imported == 1:
            elapsed = time.monotonic() - started
            _logger.info(
                "imported %d document(s) so far (%.1fs elapsed, latest url=%s)",
                imported,
                elapsed,
                url,
            )

    elapsed = time.monotonic() - started
    _logger.info(
        "done: %d imported, %d dropped, %d skipped (%.1fs)",
        imported,
        dropped,
        skipped,
        elapsed,
    )
    return 0 if dropped == 0 and skipped == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the ``import_eval_bench_corpus.py`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Bulk-import the eval-bench CSV corpus into the real Graph RAG stack "
        "(needs `make up` + EVAL_BENCH_EXPORT_ENABLED=true in .env).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to the eval-bench corpus CSV (columns: url, article, ...).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only import the first N rows (handy for a quick smoke test).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    return run(args.csv, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
