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

from typing import TYPE_CHECKING, Any

from elasticsearch import Elasticsearch, NotFoundError

from graph_rag.logging import get_logger
from graph_rag.models import DocumentRecord, Sentence, SupportingSentence

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["EsDocumentStore"]

_logger = get_logger(__name__)


def _documents_mapping(dims: int) -> dict[str, Any]:
    """The ``ES-Documents`` index mapping for a ``dims``-dimensional embedder.

    ``dynamic: true`` still lets the V4 EL checkpoint add enrichment fields
    (mentions/coref/el_result) without a migration. The V6 addition is the
    ``sentences`` **nested** field: one object per segmented sentence carrying its
    ``index``/``text``/``char_start``/``char_end`` and an indexed ``dense_vector``
    (dims pinned to the embedder, B1) with ``cosine`` similarity — so each sentence
    is individually kNN-searchable and returns its own offsets via ``inner_hits``
    (ADR-0007, B5). Storing the vectors inside the nested sentence keeps each
    passage vector paired with the offsets it belongs to.
    """
    return {
        "dynamic": True,
        "properties": {
            "document_id": {"type": "keyword"},
            "bucket": {"type": "keyword"},
            "object_key": {"type": "keyword"},
            "text": {"type": "text"},
            "sentences": {
                "type": "nested",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "text"},
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                },
            },
        },
    }


class EsDocumentStore:
    """Elasticsearch-backed :class:`~graph_rag.ports.DocumentStore` (V1-active).

    Records are keyed by ``document_id`` so ``upsert`` overwrites (idempotent).
    Writes refresh the index so reads-after-write are immediately visible, which
    the contract tests rely on.
    """

    def __init__(self, client: Elasticsearch, index: str, dims: int = 384) -> None:
        """Build the store over an existing client, target index and vector dims.

        Args:
            client: A configured ``elasticsearch`` v8 client.
            index: The ``ES-Documents`` index name.
            dims: The embedder's output dimension — pins the nested sentence
                ``dense_vector`` mapping (B1). Defaults to ``bge-small``'s 384 so
                pre-V6 call sites keep working unchanged.
        """
        self._client = client
        self._index = index
        self._dims = dims

    @classmethod
    def from_settings(cls, settings: Settings) -> EsDocumentStore:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.elasticsearch_url``, ``settings.documents_index`` and
        ``settings.embed_dim`` (the sentence ``dense_vector`` dims).
        """
        client = Elasticsearch(hosts=[settings.elasticsearch_url])
        return cls(
            client=client,
            index=settings.documents_index,
            dims=settings.embed_dim,
        )

    def ensure_index(self) -> None:
        """Create the documents index with the mapping if it does not exist.

        Idempotent: a no-op when the index is already present.
        """
        if self._client.indices.exists(index=self._index):
            _logger.debug("documents index %r already exists", self._index)
            return
        self._client.indices.create(index=self._index, mappings=_documents_mapping(self._dims))
        _logger.info(
            "created documents index %r (sentence dense_vector dims=%d)",
            self._index,
            self._dims,
        )

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or overwrite the record, keyed by ``record.document_id``.

        Indexing with ``id=record.document_id`` makes re-ingest overwrite rather
        than duplicate (R1.5, ADR-0001). ``refresh=True`` makes the write visible
        to an immediately-following read.

        The document's ``sentences`` and ``sentence_vectors`` are merged into the
        single ``sentences`` nested field (each sentence carrying its own vector)
        so per-sentence kNN with offsets works (V6, B5); the redundant top-level
        ``sentence_vectors`` is not stored separately (:meth:`get` rebuilds it from
        the nested vectors).
        """
        document: dict[str, Any] = record.model_dump()
        document.pop("sentence_vectors", None)
        document["sentences"] = self._nested_sentences(record)
        self._client.index(
            index=self._index,
            id=record.document_id,
            document=document,
            refresh=True,
        )
        _logger.debug("upserted document %s", record.document_id)

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return the record for ``document_id``, or ``None`` if absent.

        Rebuilds ``sentences`` and ``sentence_vectors`` from the nested field so
        the record round-trips (the two are stored merged; see :meth:`upsert`).
        """
        try:
            response = self._client.get(index=self._index, id=document_id)
        except NotFoundError:
            return None
        return self._to_record(response["_source"])

    def search_sentences(self, *, vector: list[float], top_k: int) -> list[SupportingSentence]:
        """Return the ``top_k`` sentences nearest ``vector`` by cosine (V6, B5).

        Runs an ES kNN search over the nested ``sentences.vector`` field with
        ``inner_hits`` so every matched sentence surfaces with its own offsets and
        per-sentence ``_score``. The ES ``_score`` for ``cosine`` similarity is
        ``(1 + cosine) / 2``; it is converted back to raw cosine in ``[-1, 1]`` so
        scores match the fake. Every matched inner hit is flattened and re-sorted
        deterministically in Python — score descending, then ``document_id``, then
        ``sentence_index`` — so the ordering is identical to
        :meth:`~graph_rag.fakes.InMemoryDocumentStore.search_sentences`, then
        truncated to ``top_k``.
        """
        knn: dict[str, Any] = {
            "field": "sentences.vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(top_k, 100),
            "inner_hits": {
                "size": top_k,
                "name": "sentences",
                "_source": {"excludes": ["sentences.vector"]},
            },
        }
        response = self._client.search(index=self._index, knn=knn, size=top_k)
        results: list[SupportingSentence] = []
        for hit in response["hits"]["hits"]:
            document_id = hit["_id"]
            inner = hit.get("inner_hits", {}).get("sentences", {}).get("hits", {}).get("hits", [])
            for inner_hit in inner:
                source = inner_hit["_source"]
                cosine = 2.0 * inner_hit["_score"] - 1.0
                results.append(
                    SupportingSentence(
                        document_id=document_id,
                        text=source["text"],
                        char_start=source["char_start"],
                        char_end=source["char_end"],
                        sentence_index=source["index"],
                        score=cosine,
                    )
                )
        results.sort(key=lambda s: (-s.score, s.document_id, s.sentence_index))
        return results[:top_k]

    def _nested_sentences(self, record: DocumentRecord) -> list[dict[str, Any]]:
        """Merge ``record.sentences`` with ``record.sentence_vectors`` for indexing.

        Each entry carries the sentence's ``index``/``text``/offsets and, when a
        positionally-aligned vector exists, its ``vector`` (so the passage is
        kNN-searchable). A sentence with no aligned vector is still indexed (its
        offsets remain queryable), it just never matches a kNN query.
        """
        vectors = record.sentence_vectors or []
        entries: list[dict[str, Any]] = []
        for i, sentence in enumerate(record.sentences):
            entry: dict[str, Any] = {
                "index": sentence.index,
                "text": sentence.text,
                "char_start": sentence.char_start,
                "char_end": sentence.char_end,
            }
            if i < len(vectors):
                entry["vector"] = list(vectors[i])
            entries.append(entry)
        return entries

    @staticmethod
    def _to_record(source: dict[str, Any]) -> DocumentRecord:
        """Rebuild a :class:`DocumentRecord` from an ES ``_source`` document.

        Reconstructs ``sentences`` (sorted by ``index`` for a stable order) and
        ``sentence_vectors`` from the merged nested field. ``sentence_vectors`` is
        set only when **every** sentence carries a vector (the orchestrator's
        invariant); otherwise it is ``None``, matching a raw/partial record.
        """
        data = dict(source)
        nested = data.pop("sentences", None) or []
        ordered = sorted(nested, key=lambda entry: entry.get("index", 0))
        sentences = [
            Sentence(
                text=entry["text"],
                char_start=entry["char_start"],
                char_end=entry["char_end"],
                index=entry["index"],
            )
            for entry in ordered
        ]
        vectors = [entry["vector"] for entry in ordered if entry.get("vector") is not None]
        data["sentences"] = [sentence.model_dump() for sentence in sentences]
        data["sentence_vectors"] = vectors if vectors and len(vectors) == len(sentences) else None
        return DocumentRecord.model_validate(data)
