"""Elasticsearch adapter for the ``EntityStore`` port (V4, ADR-0004/0005).

Wraps the ``elasticsearch`` v8 client to persist the corpus-local canonical
entity store — ``ES-Entities``, a **separate index from ``ES-Documents``** in the
same cluster (ADR-0005). It backs block-then-score entity linking: cheap
``type + normalized-name/alias`` blocking narrows candidates, then kNN over the
entity ``dense_vector`` confirms the match (and, later, seeds query-side entity
search in V6).

The real adapter is proved to behave identically to
:class:`~graph_rag.fakes.InMemoryEntityStore` at the seam by
``tests/contract/test_entity_store_contract.py``:

* ``upsert`` is idempotent, keyed by ``canonical_id`` (``id=canonical_id`` +
  ``refresh=True``) — re-upserting the same id overwrites, never duplicates;
* blocking uses the shared :func:`graph_rag.normalize.normalize_name` on both the
  entity ``name`` and every ``alias`` (stored at index time), so a mention blocks
  on name-or-alias exactly as the fake does;
* kNN ranks over a ``dense_vector`` with ``cosine`` similarity; the ES ``_score``
  (``(1 + cosine) / 2``) is converted back to raw cosine in ``[-1, 1]`` so the
  scores match the fake's brute-force cosine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elasticsearch import Elasticsearch, NotFoundError

from graph_rag.logging import get_logger
from graph_rag.models import CanonicalEntity
from graph_rag.normalize import normalize_name

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import CuratedType

__all__ = ["EsEntityStore"]

_logger = get_logger(__name__)

# Cap for the test/inspection helpers (`all`) and the kNN candidate pool — the
# corpus is small (demo), so a single non-paginated fetch is fine.
_MAX_RESULTS = 10_000


def _entities_mapping(dims: int) -> dict[str, Any]:
    """The ``ES-Entities`` index mapping for a ``dims``-dimensional embedder.

    ``name``/``aliases`` are stored verbatim (``keyword``) alongside their
    Python-``normalize_name`` forms (``normalized_name`` / ``aliases_normalized``)
    so blocking is an exact ``term`` filter on the shared normalization rule — not
    an ES analyzer that could drift from the fake. ``vector`` is an indexed
    ``dense_vector`` with ``cosine`` similarity (dims pinned to the embedder, B1).
    """
    return {
        "properties": {
            "canonical_id": {"type": "keyword"},
            "name": {"type": "keyword"},
            "normalized_name": {"type": "keyword"},
            "type": {"type": "keyword"},
            "aliases": {"type": "keyword"},
            "aliases_normalized": {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
        },
    }


class EsEntityStore:
    """Elasticsearch-backed :class:`~graph_rag.ports.EntityStore` (V4-active).

    Entities are keyed by ``canonical_id`` so ``upsert`` overwrites (idempotent,
    ADR-0004). Writes refresh the index so a read/search after a write is
    immediately visible, which the contract tests rely on.
    """

    def __init__(self, client: Elasticsearch, index: str, dims: int) -> None:
        """Build the store over an existing client, target index and vector dims.

        Args:
            client: A configured ``elasticsearch`` v8 client.
            index: The ``ES-Entities`` index name.
            dims: The embedder's output dimension (pins the ``dense_vector`` mapping).
        """
        self._client = client
        self._index = index
        self._dims = dims

    @classmethod
    def from_settings(cls, settings: Settings) -> EsEntityStore:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.elasticsearch_url``, ``settings.entities_index`` and
        ``settings.embed_dim``.
        """
        client = Elasticsearch(hosts=[settings.elasticsearch_url])
        return cls(client=client, index=settings.entities_index, dims=settings.embed_dim)

    def ensure_index(self) -> None:
        """Create the entities index with the ``dense_vector`` mapping if absent.

        Idempotent: a no-op when the index already exists.
        """
        if self._client.indices.exists(index=self._index):
            _logger.debug("entities index %r already exists", self._index)
            return
        self._client.indices.create(index=self._index, mappings=_entities_mapping(self._dims))
        _logger.info("created entities index %r (dense_vector dims=%d)", self._index, self._dims)

    def upsert(self, entity: CanonicalEntity) -> None:
        """Insert or overwrite a canonical entity, keyed by ``entity.canonical_id``.

        Stores the normalized forms of ``name`` and each ``alias`` so blocking
        matches name-or-alias (mirroring the fake). Indexing with
        ``id=entity.canonical_id`` makes a re-upsert overwrite rather than
        duplicate (idempotent). ``refresh=True`` makes the write visible to an
        immediately-following read/search.
        """
        document: dict[str, Any] = {
            "canonical_id": entity.canonical_id,
            "name": entity.name,
            "normalized_name": normalize_name(entity.name),
            "type": entity.type,
            "aliases": list(entity.aliases),
            "aliases_normalized": [normalize_name(alias) for alias in entity.aliases],
        }
        # Only index a vector when the entity has one — a vector-less entity is then
        # skipped by kNN (dense_vector search only ranks docs that have the field),
        # exactly as the fake skips entities with ``vector is None``.
        if entity.vector is not None:
            document["vector"] = list(entity.vector)

        self._client.index(
            index=self._index,
            id=entity.canonical_id,
            document=document,
            refresh=True,
        )
        _logger.debug("upserted canonical entity %s", entity.canonical_id)

    def get(self, canonical_id: str) -> CanonicalEntity | None:
        """Return the canonical entity for ``canonical_id``, or ``None`` if absent."""
        try:
            response = self._client.get(index=self._index, id=canonical_id)
        except NotFoundError:
            return None
        return self._to_entity(response["_source"])

    def block_candidates(
        self, *, entity_type: CuratedType, normalized_name: str
    ) -> list[CanonicalEntity]:
        """Return blocking candidates matching ``entity_type`` and name-or-alias (ADR-0004).

        A bool query filters ``type == entity_type`` **and** requires at least one
        of ``normalized_name`` or ``aliases_normalized`` to equal ``normalized_name``
        (the caller-supplied normalized key). ``normalized_name`` must already be
        produced by :func:`graph_rag.normalize.normalize_name` — the same rule the
        index was built with — so the fake and the adapter block identically.
        """
        query = {
            "bool": {
                "filter": [{"term": {"type": entity_type}}],
                "should": [
                    {"term": {"normalized_name": normalized_name}},
                    {"term": {"aliases_normalized": normalized_name}},
                ],
                "minimum_should_match": 1,
            }
        }
        response = self._client.search(index=self._index, query=query, size=_MAX_RESULTS)
        return [self._to_entity(hit["_source"]) for hit in response["hits"]["hits"]]

    def knn(
        self,
        *,
        vector: list[float],
        entity_type: CuratedType | None = None,
        top_k: int,
    ) -> list[tuple[CanonicalEntity, float]]:
        """Return the ``top_k`` nearest entities to ``vector`` by cosine, descending.

        Runs an ES kNN search over the ``vector`` ``dense_vector`` field, optionally
        pre-filtered to a single ``entity_type``. Entities with no stored vector are
        not indexed on that field and so never match (as the fake skips them). The
        ES ``_score`` for ``cosine`` similarity is ``(1 + cosine) / 2``; it is
        converted back to raw cosine in ``[-1, 1]`` so scores match the fake.
        """
        knn_query: dict[str, Any] = {
            "field": "vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": max(top_k, 100),
        }
        if entity_type is not None:
            knn_query["filter"] = {"term": {"type": entity_type}}

        response = self._client.search(index=self._index, knn=knn_query, size=top_k)
        results: list[tuple[CanonicalEntity, float]] = []
        for hit in response["hits"]["hits"]:
            cosine = 2.0 * hit["_score"] - 1.0
            results.append((self._to_entity(hit["_source"]), cosine))
        return results

    def count(self) -> int:
        """Return the number of canonical entities currently stored (test helper)."""
        return int(self._client.count(index=self._index)["count"])

    def all(self) -> list[CanonicalEntity]:
        """Return every stored canonical entity (test/inspection helper)."""
        response = self._client.search(
            index=self._index,
            query={"match_all": {}},
            size=_MAX_RESULTS,
        )
        return [self._to_entity(hit["_source"]) for hit in response["hits"]["hits"]]

    @staticmethod
    def _to_entity(source: dict[str, Any]) -> CanonicalEntity:
        """Rebuild a :class:`CanonicalEntity` from an ES ``_source`` document.

        Reads only the port-visible fields; the index-time blocking helpers
        (``normalized_name`` / ``aliases_normalized``) are dropped.
        """
        return CanonicalEntity(
            canonical_id=source["canonical_id"],
            name=source["name"],
            type=source["type"],
            aliases=list(source.get("aliases", [])),
            vector=source.get("vector"),
        )
