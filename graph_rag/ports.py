"""The external-dependency ports — the single primary test seam (ADR-0010).

Everything outside the pipeline's control sits behind a narrow Python
``typing.Protocol`` here, constructor-injected into the orchestrator and query
service. Real adapters (``graph_rag.adapters``) wrap live services; in-memory
fakes (``graph_rag.fakes``) back the fast, $0, no-Docker suite.

Six ports map to the architecture's N10–N15 affordances. **V1 actively uses only
``ObjectStore`` and ``DocumentStore``.** ``EntityStore``, ``GraphStore``,
``LLMClient`` and ``Embedder`` are declared with minimal, clearly-marked stub
method sets so later slices (V4/V5/V3/V6) plug in without changing this contract.

``TriggerPublisher`` is the messaging seam: ``POST /ingest`` publishes through it
so the endpoint and its test can inject the fake instead of a real Kafka producer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from graph_rag.models import DocumentRecord, IngestTrigger

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

    Stub for later slices (V4); not exercised in V1.
    """

    def upsert(self, entity: dict[str, Any]) -> None:
        """Insert or overwrite a canonical entity, keyed by its ``canonical_id``."""
        ...

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return up to ``top_k`` candidate entities nearest to ``vector`` (kNN)."""
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

    Stub for later slices (V3/V5/V7); not exercised in V1.
    """

    def complete(self, prompt: str, **params: Any) -> str:
        """Return the model completion for ``prompt`` under the given ``params``."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Local sentence-transformer embedder (``bge-small-en-v1.5``).

    Stub for later slices (V4/V6); not exercised in V1.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""
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
