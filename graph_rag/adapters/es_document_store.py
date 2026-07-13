"""Elasticsearch adapter for the ``DocumentStore`` port (ADR-0005, ADR-0010).

Wraps the ``elasticsearch`` v8 client to persist the per-document
``ES-Documents`` record. V1 writes only the raw ``text`` (plus identity fields);
the enrichment fields (``mentions``/``coref_clusters``/``el_result``/``vectors``)
are added at the V4 entity-linking checkpoint, so the index mapping stays minimal
and ``dynamic`` friendly — later slices add fields without a migration.

Idempotency (R1.5, ADR-0001): :meth:`EsDocumentStore.upsert` indexes with
``id=record.document_id`` so re-ingesting the same object overwrites its record
rather than creating a duplicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch, NotFoundError

from graph_rag.logging import get_logger
from graph_rag.models import DocumentRecord

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["EsDocumentStore"]

_logger = get_logger(__name__)

# Minimal V1 mapping. ``dynamic: true`` lets the V4 EL checkpoint add
# enrichment fields (mentions/coref/el_result/vectors) without a migration.
_DOCUMENTS_MAPPING: dict[str, object] = {
    "dynamic": True,
    "properties": {
        "document_id": {"type": "keyword"},
        "bucket": {"type": "keyword"},
        "object_key": {"type": "keyword"},
        "text": {"type": "text"},
    },
}


class EsDocumentStore:
    """Elasticsearch-backed :class:`~graph_rag.ports.DocumentStore` (V1-active).

    Records are keyed by ``document_id`` so ``upsert`` overwrites (idempotent).
    Writes refresh the index so reads-after-write are immediately visible, which
    the contract tests rely on.
    """

    def __init__(self, client: Elasticsearch, index: str) -> None:
        """Build the store over an existing client and target index.

        Args:
            client: A configured ``elasticsearch`` v8 client.
            index: The ``ES-Documents`` index name.
        """
        self._client = client
        self._index = index

    @classmethod
    def from_settings(cls, settings: Settings) -> EsDocumentStore:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.elasticsearch_url`` and ``settings.documents_index``.
        """
        client = Elasticsearch(hosts=[settings.elasticsearch_url])
        return cls(client=client, index=settings.documents_index)

    def ensure_index(self) -> None:
        """Create the documents index with the V1 mapping if it does not exist.

        Idempotent: a no-op when the index is already present.
        """
        if self._client.indices.exists(index=self._index):
            _logger.debug("documents index %r already exists", self._index)
            return
        self._client.indices.create(index=self._index, mappings=_DOCUMENTS_MAPPING)
        _logger.info("created documents index %r", self._index)

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or overwrite the record, keyed by ``record.document_id``.

        Indexing with ``id=record.document_id`` makes re-ingest overwrite rather
        than duplicate (R1.5, ADR-0001). ``refresh=True`` makes the write visible
        to an immediately-following read.
        """
        self._client.index(
            index=self._index,
            id=record.document_id,
            document=record.model_dump(),
            refresh=True,
        )
        _logger.debug("upserted document %s", record.document_id)

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return the record for ``document_id``, or ``None`` if absent."""
        try:
            response = self._client.get(index=self._index, id=document_id)
        except NotFoundError:
            return None
        return DocumentRecord.model_validate(response["_source"])
