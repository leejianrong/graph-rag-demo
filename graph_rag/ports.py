"""The external-dependency ports — the single primary test seam (ADR-0010).

Everything outside the pipeline's control sits behind a narrow Python
``typing.Protocol`` here, constructor-injected into the orchestrator and query
service. Real adapters (``graph_rag.adapters``) wrap live services; in-memory
fakes (``graph_rag.fakes``) back the fast, $0, no-Docker suite.

Six ports map to the architecture's N10–N15 affordances. ``ObjectStore`` +
``DocumentStore`` are active from V1, ``LLMClient`` from V3, and ``EntityStore`` +
``Embedder`` from V4 (entity linking). ``GraphStore`` stays a minimal stub until
V5/V6. Signatures are fixed here so slices plug in without changing this contract.

``TriggerPublisher`` is the messaging seam: ``POST /ingest`` publishes through it
so the endpoint and its test can inject the fake instead of a real Kafka producer.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from graph_rag.models import CanonicalEntity, CuratedType, DocumentRecord, IngestTrigger

# Structured-output type variable: ``structured`` returns an instance of exactly
# the Pydantic model type the caller asked for (ADR-0008).
StructuredT = TypeVar("StructuredT", bound=BaseModel)

__all__ = [
    "ObjectStore",
    "DocumentStore",
    "EntityStore",
    "GraphStore",
    "LLMClient",
    "Embedder",
    "TriggerPublisher",
]


# --- V1-active ports ---------------------------------------------------------


@runtime_checkable
class ObjectStore(Protocol):
    """Read/write a document's raw bytes given ``{bucket, object_key}`` (MinIO).

    Active in V1.
    """

    def put(self, bucket: str, object_key: str, data: bytes) -> None:
        """Store ``data`` under ``(bucket, object_key)``, overwriting any prior object."""
        ...

    def get(self, bucket: str, object_key: str) -> bytes:
        """Return the bytes at ``(bucket, object_key)``.

        Raises:
            FileNotFoundError: If no object exists at that location.
        """
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Read/write the per-document ``ES-Documents`` record (Elasticsearch).

    Active in V1. ``upsert`` is idempotent, keyed by ``record.document_id``:
    re-upserting the same ID overwrites (ADR-0001).
    """

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or overwrite the record keyed by ``record.document_id``."""
        ...

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return the record for ``document_id``, or ``None`` if absent."""
        ...


# --- Stub ports for later slices (declared now so the contract is fixed) -----


@runtime_checkable
class EntityStore(Protocol):
    """Canonical-entity store over ``ES-Entities`` (upsert + blocking/kNN search).

    Active from V4 (ADR-0004/0005). Backs corpus-local entity linking —
    block-then-score EL at ingestion and, later, query-side kNN entity seeding
    (V6). ``upsert`` is idempotent, keyed by ``entity.canonical_id``: re-upserting
    the same ID overwrites (so merges that grow ``aliases``/refresh ``vector`` do
    not duplicate). The real adapter runs these against Elasticsearch; the fast
    suite injects :class:`~graph_rag.fakes.InMemoryEntityStore`.
    """

    def upsert(self, entity: CanonicalEntity) -> None:
        """Insert or overwrite a canonical entity, keyed by ``entity.canonical_id``."""
        ...

    def get(self, canonical_id: str) -> CanonicalEntity | None:
        """Return the canonical entity for ``canonical_id``, or ``None`` if absent."""
        ...

    def block_candidates(
        self, *, entity_type: CuratedType, normalized_name: str
    ) -> list[CanonicalEntity]:
        """Return the blocking candidates for an EL match (ADR-0004).

        The blocking filter narrows candidates cheaply before embedding scoring:
        an entity is a candidate iff its ``type`` equals ``entity_type`` **and**
        ``normalized_name`` matches the normalized form of its ``name`` or any of
        its ``aliases``. ``normalized_name`` must be produced by
        :func:`graph_rag.normalize.normalize_name` — the one shared rule both the
        fake and the real adapter block on.
        """
        ...

    def knn(
        self,
        *,
        vector: list[float],
        entity_type: CuratedType | None = None,
        top_k: int,
    ) -> list[tuple[CanonicalEntity, float]]:
        """Return the ``top_k`` nearest entities to ``vector`` by cosine similarity.

        Ranks over each entity's ``dense_vector``, descending by similarity, and
        returns ``(entity, score)`` pairs where ``score`` is cosine similarity in
        ``[-1, 1]``. Optionally restrict the search to a single ``entity_type``.
        Entities with no ``vector`` are skipped. Used both to confirm a blocked EL
        match and for query-side entity seeding (V6, B5).
        """
        ...

    def count(self) -> int:
        """Return the number of canonical entities currently stored (test helper)."""
        ...

    def all(self) -> list[CanonicalEntity]:
        """Return every stored canonical entity (test/inspection helper)."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph store over Neo4j (write triples, k-hop traversal).

    Stub for later slices (V5/V6); not exercised in V1.
    """

    def write_triples(self, triples: list[dict[str, Any]]) -> None:
        """Write nodes + provenance-carrying edges for the given triples."""
        ...

    def khop(self, seed_ids: list[str], hops: int) -> dict[str, Any]:
        """Expand ``hops`` hops from ``seed_ids`` and return the connected subgraph."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM client (LiteLLM + response cache + structured output).

    Active from V3 (coref is the first LLM use, ADR-0008). Both methods are served
    through a persistent ``sha256(model + prompt + params)`` response cache, so a
    repeated call is a cache hit that never touches the provider (observably $0).
    """

    def complete(self, prompt: str, **params: Any) -> str:
        """Return the model completion for ``prompt`` under the given ``params``."""
        ...

    def structured(self, prompt: str, schema: type[StructuredT], **params: Any) -> StructuredT:
        """Return a validated ``schema`` instance for ``prompt`` (ADR-0008).

        Requests structured/JSON output and validates it against the Pydantic
        ``schema``, retrying on parse/validation failure before raising. Cached
        like :meth:`complete`.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Local sentence-transformer embedder (``BAAI/bge-small-en-v1.5``, B1).

    Active from V4 (ADR-0004): produces the dense vectors EL scores over
    (mention-in-context + canonical entities) and, later, passage/sentence and
    query vectors (V6). The default model ``BAAI/bge-small-en-v1.5`` emits
    **384-dim** vectors (:attr:`dim`), which pins the ES ``dense_vector`` mapping.
    Embeddings must be **deterministic** for a given input text. The real adapter
    wraps ``sentence-transformers``; the fast suite injects
    :class:`~graph_rag.fakes.FakeEmbedder` (pure-Python, no model download).
    """

    @property
    def dim(self) -> int:
        """The embedding dimension (384 for ``bge-small-en-v1.5``)."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector (length :attr:`dim`) per input text, in order."""
        ...


# --- Messaging seam ----------------------------------------------------------


@runtime_checkable
class TriggerPublisher(Protocol):
    """Publish an ingest trigger onto the Kafka trigger topic.

    Active in V1: ``POST /ingest`` publishes through this port so the endpoint and
    its test can inject a fake instead of a live Kafka producer.
    """

    def publish(self, trigger: IngestTrigger) -> None:
        """Publish ``trigger`` to the ingest-trigger topic."""
        ...
