"""In-memory fakes for every port (ADR-0010).

These back the fast, $0, no-Docker suite and are the fixture backbone of the test
suite (exposed as fixtures in ``tests/conftest.py``). Each fake implements the
matching ``Protocol`` in :mod:`graph_rag.ports`.

V1 actively uses :class:`InMemoryObjectStore`, :class:`InMemoryDocumentStore` and
:class:`InMemoryTriggerPublisher`. The remaining fakes exist so later slices can
instantiate and plug them in; their unused stub methods raise
``NotImplementedError``, but construction never does.
"""

from __future__ import annotations

from typing import Any

from graph_rag.models import DocumentRecord, IngestTrigger, Mention, Sentence
from graph_rag.stages.ner import NerResult

__all__ = [
    "InMemoryObjectStore",
    "InMemoryDocumentStore",
    "InMemoryTriggerPublisher",
    "InMemoryEntityStore",
    "InMemoryGraphStore",
    "FakeLLMClient",
    "FakeEmbedder",
    "FakeNerStage",
]


# --- V1-active fakes ---------------------------------------------------------


class InMemoryObjectStore:
    """In-memory :class:`~graph_rag.ports.ObjectStore` backed by a dict.

    Keyed by ``(bucket, object_key)`` -> ``bytes``.
    """

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put(self, bucket: str, object_key: str, data: bytes) -> None:
        """Store ``data`` under ``(bucket, object_key)``, overwriting any prior object."""
        self._objects[(bucket, object_key)] = data

    def get(self, bucket: str, object_key: str) -> bytes:
        """Return the bytes at ``(bucket, object_key)``.

        Raises:
            FileNotFoundError: If no object exists at that location.
        """
        try:
            return self._objects[(bucket, object_key)]
        except KeyError as exc:
            raise FileNotFoundError(f"No object at {bucket}/{object_key}") from exc


class InMemoryDocumentStore:
    """In-memory :class:`~graph_rag.ports.DocumentStore` backed by a dict.

    Keyed by ``document_id`` -> :class:`~graph_rag.models.DocumentRecord`. ``upsert``
    overwrites, so re-ingesting an object never duplicates (ADR-0001).
    """

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or overwrite the record keyed by ``record.document_id``."""
        self._records[record.document_id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return the record for ``document_id``, or ``None`` if absent."""
        return self._records.get(document_id)


class InMemoryTriggerPublisher:
    """In-memory :class:`~graph_rag.ports.TriggerPublisher`.

    Appends each published trigger to the public :attr:`published` list so tests
    can assert what ``POST /ingest`` published.
    """

    def __init__(self) -> None:
        self.published: list[IngestTrigger] = []

    def publish(self, trigger: IngestTrigger) -> None:
        """Record ``trigger`` on :attr:`published`."""
        self.published.append(trigger)


# --- Stub fakes for later slices (instantiable; unused methods raise) --------


class InMemoryEntityStore:
    """Trivial in-memory :class:`~graph_rag.ports.EntityStore`.

    Stub for later slices (V4); not exercised in V1. Construction is cheap; the
    stub methods raise until a slice needs them.
    """

    def __init__(self) -> None:
        self._entities: dict[str, dict[str, Any]] = {}

    def upsert(self, entity: dict[str, Any]) -> None:
        """Not implemented in V1."""
        raise NotImplementedError("EntityStore is a stub until V4")

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Not implemented in V1."""
        raise NotImplementedError("EntityStore is a stub until V4")


class InMemoryGraphStore:
    """Trivial in-memory :class:`~graph_rag.ports.GraphStore`.

    Stub for later slices (V5/V6); not exercised in V1.
    """

    def __init__(self) -> None:
        self._triples: list[dict[str, Any]] = []

    def write_triples(self, triples: list[dict[str, Any]]) -> None:
        """Not implemented in V1."""
        raise NotImplementedError("GraphStore is a stub until V5")

    def khop(self, seed_ids: list[str], hops: int) -> dict[str, Any]:
        """Not implemented in V1."""
        raise NotImplementedError("GraphStore is a stub until V6")


class FakeLLMClient:
    """Trivial :class:`~graph_rag.ports.LLMClient`.

    Stub for later slices (V3/V5/V7); not exercised in V1. A later fake will
    return canned structured responses keyed by prompt hash (ADR-0008/ADR-0010).
    """

    def complete(self, prompt: str, **params: Any) -> str:
        """Not implemented in V1."""
        raise NotImplementedError("LLMClient is a stub until V3")


class FakeEmbedder:
    """Trivial :class:`~graph_rag.ports.Embedder`.

    Stub for later slices (V4/V6); not exercised in V1.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Not implemented in V1."""
        raise NotImplementedError("Embedder is a stub until V4")


# --- V2-active fake ----------------------------------------------------------


class FakeNerStage:
    """Canned :class:`~graph_rag.stages.ner.NerStage` for the fast suite (V2-active).

    Returns configurable canned mentions + sentences with NO model download, so
    the fast E2E proves the wiring + the :class:`~graph_rag.models.PipelineResult`
    carry (not spaCy quality) — deterministic, ``$0``, no Docker. The last text
    passed to :meth:`analyze` is recorded on :attr:`last_text` for assertions.
    """

    def __init__(
        self,
        mentions: list[Mention] | None = None,
        sentences: list[Sentence] | None = None,
    ) -> None:
        """Configure the canned output.

        Args:
            mentions: Canned mentions to return from every :meth:`analyze` call.
            sentences: Canned sentences to return from every :meth:`analyze` call.
        """
        self._mentions = list(mentions or [])
        self._sentences = list(sentences or [])
        self.last_text: str | None = None

    def analyze(self, text: str) -> NerResult:
        """Return the canned mentions + sentences, ignoring ``text``'s content."""
        self.last_text = text
        return NerResult(
            mentions=list(self._mentions),
            sentences=list(self._sentences),
        )
