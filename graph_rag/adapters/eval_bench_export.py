"""Eval-bench export adapter — an OPT-IN dual-write into an external project's shape.

``eval-bench`` (github.com/leejianrong/eval-bench, a separate LLM-agent benchmark
harness) has existing, unmodifiable client code
(``retrieval/backends.py``) that reads:

* **Elasticsearch**: a document indexed at ``_id = sha256(url)`` (the article's
  source URL), with the full article body text in a field findable via a bare
  ``query_string`` query with NO explicit ``fields:`` list — i.e. it relies on
  ES's default-field (``"*"``) behaviour, so the body text lives in a plain
  top-level ``text`` field (an ordinary, indexed, non-excluded ``text`` mapping —
  never buried behind an obscure name or an ``"index": false`` mapping that would
  drop it from that default expansion).
* **Neo4j**: ``:Document {doc_id}`` nodes (``doc_id = sha256(url)``);
  ``:Entity {name, description}`` nodes joined onto THIS project's EXISTING
  ``:Entity`` nodes via their existing ``canonical_id`` property (no second,
  parallel entity identity); a ``MENTIONS`` relationship from ``Document`` to
  every ``Entity`` mentioned in that document. This project's own typed
  ``Entity-[REL_TYPE]->Entity`` edges
  (:class:`~graph_rag.adapters.neo4j_graph_store.Neo4jGraphStore`) need no
  change and are untouched here.

This adapter is a self-contained, additive dual-write: it reuses this project's
own Elasticsearch/Neo4j connections (:class:`~graph_rag.config.Settings`) but
writes to a SEPARATE index (``settings.eval_bench_export_index``, default
``osint_reports``) so it never collides with this project's own ``documents`` /
``entities`` indices, and it MERGEs onto (never creates a second copy of) this
project's own ``:Entity`` nodes.

Wired OPT-IN: :class:`~graph_rag.orchestrator.Orchestrator` accepts
``eval_bench_export: EvalBenchExportStage | None = None`` and calls
:meth:`EvalBenchExportStage.write` only when it is supplied — omitted (the
default), V1-V5 behaviour is unaffected. ``main.py`` wires a real instance only
when ``settings.eval_bench_export_enabled`` is true.

Document identity here is INTENTIONALLY decoupled from this project's own
``document_id`` (``graph_rag.ids.document_id``, a hash of ``{bucket}/{object_key}``):
this adapter independently computes ``sha256(url)`` from the caller-supplied
``source_url`` (see :class:`~graph_rag.models.IngestTrigger.source_url`) and uses
it ONLY as the id it writes into its own ES doc / Neo4j ``:Document`` node.

Entity descriptions: :class:`~graph_rag.models.CanonicalEntity` carries no
description field, so one is synthesized here as "the first sentence this entity
was mentioned in", reconstructed from the already-persisted
``DocumentRecord.sentences`` + ``DocumentRecord.el_result`` (no changes to the
entity-linking stage's internals).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from elasticsearch import Elasticsearch
from neo4j import GraphDatabase

from graph_rag.logging import get_logger
from graph_rag.models import CanonicalEntity, DocumentRecord, Triple

if TYPE_CHECKING:
    from neo4j import Driver

    from graph_rag.config import Settings

__all__ = ["EvalBenchExportStage"]

_logger = get_logger(__name__)


def _osint_reports_mapping() -> dict[str, Any]:
    """The export index mapping.

    ``text`` is a plain ``text`` field with no custom analyzer/exclusion, so it is
    picked up by ES's default ``query_string`` field expansion (``"*"``) with no
    explicit ``fields:`` list — exactly the read path eval-bench's client relies
    on. ``url``/``doc_id`` are ``keyword`` (exact-match identity, not full-text).
    """
    return {
        "properties": {
            "doc_id": {"type": "keyword"},
            "url": {"type": "keyword"},
            "text": {"type": "text"},
            "source_document_id": {"type": "keyword"},
        }
    }


class EvalBenchExportStage:
    """Dual-writes one document into the eval-bench-compatible ES index + Neo4j shape.

    Constructed over its own Elasticsearch client + Neo4j driver (matching this
    project's existing adapter house style — see
    :class:`~graph_rag.adapters.es_document_store.EsDocumentStore` and
    :class:`~graph_rag.adapters.neo4j_graph_store.Neo4jGraphStore`, each of which
    also builds its own client/driver from :class:`~graph_rag.config.Settings`).
    """

    def __init__(
        self,
        es_client: Elasticsearch,
        es_index: str,
        driver: Driver,
        database: str = "neo4j",
    ) -> None:
        """Build the stage over an existing ES client + Neo4j driver.

        Args:
            es_client: A configured ``elasticsearch`` v8 client.
            es_index: The export index name (``settings.eval_bench_export_index``).
            driver: A configured ``neo4j`` driver (lazy — no connection opens
                until first use).
            database: The target Neo4j database name.
        """
        self._es_client = es_client
        self._es_index = es_index
        self._driver = driver
        self._database = database

    @classmethod
    def from_settings(cls, settings: Settings) -> EvalBenchExportStage:
        """Construct from :class:`~graph_rag.config.Settings`.

        Reuses ``settings.elasticsearch_url`` / ``settings.neo4j_uri`` +
        credentials (the same connections the rest of the real stack uses) but
        targets the separate ``settings.eval_bench_export_index``.
        """
        es_client = Elasticsearch(hosts=[settings.elasticsearch_url])
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return cls(es_client, settings.eval_bench_export_index, driver)

    def close(self) -> None:
        """Close the underlying Neo4j driver and its connection pool."""
        self._driver.close()

    def ensure_index(self) -> None:
        """Create the export ES index with its mapping if it does not exist.

        Idempotent: a no-op when the index is already present.
        """
        if self._es_client.indices.exists(index=self._es_index):
            _logger.debug("eval-bench export index %r already exists", self._es_index)
            return
        self._es_client.indices.create(index=self._es_index, mappings=_osint_reports_mapping())
        _logger.info("created eval-bench export index %r", self._es_index)

    def write(
        self,
        record: DocumentRecord,
        canonical_entities: list[CanonicalEntity],
        triples: list[Triple],  # noqa: ARG002 - kept for API symmetry / future edge export
        *,
        source_url: str | None,
    ) -> None:
        """Dual-write ``record``'s eval-bench-compatible shape (opt-in, best-effort).

        Args:
            record: This project's own (already-persisted) ``DocumentRecord`` —
                supplies the full raw ``text`` and, via ``sentences``/``el_result``,
                the raw material for entity descriptions.
            canonical_entities: Every entity mentioned in this document (the same
                deduplicated list the graph checkpoint upserts as nodes and passes
                to KG-build — see ``Orchestrator._doc_canonical_entities``).
            triples: This document's knowledge-graph edges (unused by the current
                eval-bench contract, which only wants ``MENTIONS``; accepted so the
                orchestrator's hook can hand over everything it has in scope
                without this adapter dictating what the caller computes).
            source_url: The document's real source URL (threaded through
                :class:`~graph_rag.models.IngestTrigger.source_url`). Required to
                compute the ``sha256(url)`` identity this adapter writes under; if
                absent (e.g. a document ingested via ``POST /ingest``, which never
                sets it) the write is skipped — logged, not raised, so an
                unconfigured caller never drops an otherwise-successful document
                (ADR-0001's log-and-drop is for pipeline failures, not a missing
                optional input).
        """
        if not source_url:
            _logger.warning(
                "eval-bench export skipped for document %s: no source_url on the trigger",
                record.document_id,
            )
            return
        doc_id = _sha256(source_url)
        self._write_document(doc_id, source_url, record.text, record.document_id)
        self._write_document_node(doc_id, source_url)
        self._write_entities(doc_id, record, canonical_entities)
        _logger.debug(
            "eval-bench export wrote document %s (doc_id=%s, %d entit(y/ies))",
            record.document_id,
            doc_id,
            len(canonical_entities),
        )

    # --- Elasticsearch --------------------------------------------------

    def _write_document(self, doc_id: str, url: str, text: str, source_document_id: str) -> None:
        """Index the article at ``_id = doc_id`` (overwrite-idempotent, R1.5-style)."""
        self._es_client.index(
            index=self._es_index,
            id=doc_id,
            document={
                "doc_id": doc_id,
                "url": url,
                "text": text,
                "source_document_id": source_document_id,
            },
            refresh=True,
        )

    # --- Neo4j -----------------------------------------------------------

    def _write_document_node(self, doc_id: str, url: str) -> None:
        """MERGE the ``:Document {doc_id}`` node (idempotent), setting ``url``.

        Written even for a zero-entity document — an entity-free document still
        exists in the corpus. ``MERGE`` on ``doc_id`` keeps re-ingest idempotent.
        """
        self._run("MERGE (d:Document {doc_id: $doc_id}) SET d.url = $url", doc_id=doc_id, url=url)

    def _write_entities(
        self, doc_id: str, record: DocumentRecord, entities: list[CanonicalEntity]
    ) -> None:
        """MERGE each ``:Entity`` node + its ``MENTIONS`` edge from ``:Document``.

        Entities MERGE onto THIS project's existing ``:Entity {canonical_id}``
        nodes (created by
        :class:`~graph_rag.adapters.neo4j_graph_store.Neo4jGraphStore.upsert_entities`
        at the graph checkpoint just before this hook runs) — no second identity
        is created. ``MERGE`` on the ``MENTIONS`` edge keeps re-ingest idempotent
        (no duplicate edges on re-import).
        """
        if not entities:
            return
        descriptions = _entity_descriptions(record, entities)
        rows = [
            {
                "canonical_id": entity.canonical_id,
                "name": entity.name,
                "description": descriptions.get(entity.canonical_id, entity.name),
            }
            for entity in entities
        ]
        self._run(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "UNWIND $rows AS row "
            "MERGE (e:Entity {canonical_id: row.canonical_id}) "
            "SET e.name = row.name, e.description = row.description "
            "MERGE (d)-[:MENTIONS]->(e)",
            doc_id=doc_id,
            rows=rows,
        )

    def _run(self, query: str, **params: Any) -> list[Any]:
        """Execute ``query`` with ``params`` in an auto-commit transaction; return records."""
        result = self._driver.execute_query(query, database_=self._database, **params)
        return list(result.records)


def _sha256(value: str) -> str:
    """Return the SHA-256 hex digest of ``value`` (the eval-bench document identity)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _entity_descriptions(record: DocumentRecord, entities: list[CanonicalEntity]) -> dict[str, str]:
    """Synthesize "the first sentence this entity was mentioned in" per entity.

    ``CanonicalEntity`` carries no description field, so this reconstructs one
    from data already persisted on ``record`` by the time the orchestrator's
    hook fires: for each entity, find its earliest-occurring
    :class:`~graph_rag.models.EntityLink` (by the char offset of the matching NER
    mention), then the sentence containing that mention's start offset. Falls
    back to the entity's own name when no matching mention/sentence is found
    (e.g. a coref-cluster surface with no exact ``Mention.text`` match).
    """
    # First occurrence (by mention char_start) of each canonical_id's linked
    # surface form, resolved via record.mentions (typed, char-offset mentions).
    mentions_by_text: dict[str, int] = {}
    for mention in record.mentions:
        earliest = mentions_by_text.get(mention.text)
        if earliest is None or mention.char_start < earliest:
            mentions_by_text[mention.text] = mention.char_start

    earliest_offset: dict[str, int] = {}
    for link in record.el_result:
        offset = mentions_by_text.get(link.mention_text)
        if offset is None:
            continue
        if link.canonical_id not in earliest_offset or offset < earliest_offset[link.canonical_id]:
            earliest_offset[link.canonical_id] = offset

    descriptions: dict[str, str] = {}
    for entity in entities:
        offset = earliest_offset.get(entity.canonical_id)
        sentence_text = _sentence_containing(record, offset) if offset is not None else None
        descriptions[entity.canonical_id] = sentence_text or entity.name
    return descriptions


def _sentence_containing(record: DocumentRecord, offset: int) -> str | None:
    """Return the text of the sentence whose ``[char_start, char_end)`` covers ``offset``."""
    for sentence in record.sentences:
        if sentence.char_start <= offset < sentence.char_end:
            return sentence.text
    return None
