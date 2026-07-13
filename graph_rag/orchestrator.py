"""The pipeline shell (N4) — the in-process orchestrator (ADR-0001).

V1 runs only the *read* stage (N5) and the ingestion checkpoint (N11): fetch the
document bytes via :class:`~graph_rag.ports.ObjectStore`, create the bare
``ES-Documents`` record with **raw text at ingestion, before processing**, and
persist it via :class:`~graph_rag.ports.DocumentStore`. Later slices add the
NER/coref/EL/KG-build stages behind this same shell.

Error handling is **log-and-drop per document** (ADR-0001): any exception while
processing one document is logged and swallowed (``process_document`` returns
``None``) so a single bad document never wedges the Kafka consumer loop — the
next trigger is processed normally. Idempotency comes from the deterministic
``document_id``: reprocessing overwrites (R1.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_rag.ids import document_id
from graph_rag.logging import get_logger
from graph_rag.models import DocumentRecord

if TYPE_CHECKING:
    from graph_rag.models import IngestTrigger
    from graph_rag.ports import DocumentStore, ObjectStore

__all__ = ["Orchestrator"]

_logger = get_logger(__name__)


class Orchestrator:
    """The single in-process pipeline shell, constructor-injected with its ports.

    V1 uses only ``object_store`` (read stage) and ``document_store`` (ingestion
    checkpoint). The remaining ports (EntityStore/GraphStore/LLMClient/Embedder)
    plug into this same shell in later slices without changing the V1 contract.
    """

    def __init__(self, object_store: ObjectStore, document_store: DocumentStore) -> None:
        """Wire the V1-active ports.

        Args:
            object_store: Reads a document's raw bytes (N5 / MinIO).
            document_store: Writes the ``ES-Documents`` record (N11 / Elasticsearch).
        """
        self._object_store = object_store
        self._document_store = document_store

    def process_document(self, trigger: IngestTrigger) -> DocumentRecord | None:
        """Process one ingest trigger end-to-end, log-and-drop on failure.

        Steps (V1): read bytes (N5) → compute deterministic ``document_id`` →
        decode to text → build the raw :class:`~graph_rag.models.DocumentRecord`
        → upsert at the ingestion checkpoint (N11). No enrichment in V1.

        Any exception is logged and swallowed so the consumer loop keeps going
        (ADR-0001); on failure this returns ``None`` instead of raising.

        Args:
            trigger: The ``{bucket, object_key}`` payload for one document.

        Returns:
            The persisted :class:`~graph_rag.models.DocumentRecord` on success, or
            ``None`` if processing this document failed (dropped).
        """
        try:
            # 1. Read stage (N5): fetch the raw bytes from the object store.
            data = self._object_store.get(trigger.bucket, trigger.object_key)

            # 2. Deterministic identity (ADR-0001): same location -> same id -> overwrite.
            doc_id = document_id(trigger.bucket, trigger.object_key)

            # 3. Decode to text. errors="replace" keeps a malformed byte from
            #    wedging the pipeline; the raw text is what V1 persists.
            text = data.decode("utf-8", errors="replace")

            # 4. Build the bare record (raw text only — no enrichment in V1) and
            #    persist it at the ingestion checkpoint (N11).
            record = DocumentRecord(
                document_id=doc_id,
                bucket=trigger.bucket,
                object_key=trigger.object_key,
                text=text,
            )
            self._document_store.upsert(record)

            _logger.info(
                "ingested document %s (%s/%s)",
                doc_id,
                trigger.bucket,
                trigger.object_key,
            )
            # 5. Return the record so callers/tests can assert on it.
            return record
        except Exception:  # noqa: BLE001 — log-and-drop per document (ADR-0001).
            _logger.exception(
                "dropping document %s/%s after processing error",
                trigger.bucket,
                trigger.object_key,
            )
            return None
