"""In-memory fakes for every port (ADR-0010).

These back the fast, $0, no-Docker suite and are the fixture backbone of the test
suite (exposed as fixtures in ``tests/conftest.py``). Each fake implements the
matching ``Protocol`` in :mod:`graph_rag.ports`.

V1 actively uses :class:`InMemoryObjectStore`, :class:`InMemoryDocumentStore` and
:class:`InMemoryTriggerPublisher`; V3 adds :class:`FakeLLMClient`; V4 adds the
full-fidelity :class:`FakeEmbedder` + :class:`InMemoryEntityStore`; V5 adds the
full-fidelity :class:`InMemoryGraphStore` (idempotent node upsert, doc-scoped edge
delete, BFS k-hop).
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Any, TypeVar

from graph_rag.models import (
    CanonicalEntity,
    CorefCluster,
    CuratedType,
    DocumentRecord,
    IngestTrigger,
    Mention,
    Sentence,
    Subgraph,
    SupportingSentence,
    Triple,
)
from graph_rag.normalize import normalize_name
from graph_rag.stages.ner import NerResult

if TYPE_CHECKING:
    from pydantic import BaseModel

_StructuredT = TypeVar("_StructuredT", bound="BaseModel")

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

    def search_sentences(self, *, vector: list[float], top_k: int) -> list[SupportingSentence]:
        """Return the ``top_k`` sentences nearest ``vector`` by cosine (V6, B5).

        Brute-force cosine over every stored record's ``sentence_vectors``,
        pairing each vector positionally with the record's ``sentences`` by index.
        A record with no ``sentence_vectors`` (``None``/empty) contributes nothing,
        as do vectors with no aligned :class:`~graph_rag.models.Sentence`. Ordering
        is deterministic — score descending, then ``document_id``, then
        ``sentence_index`` — mirroring the real adapter's re-sort so the contract
        test can prove equivalence.
        """
        scored: list[SupportingSentence] = []
        for record in self._records.values():
            vectors = record.sentence_vectors or []
            for idx, sentence_vector in enumerate(vectors):
                if idx >= len(record.sentences):
                    # A vector without an aligned sentence has no offsets to return.
                    continue
                sentence = record.sentences[idx]
                scored.append(
                    SupportingSentence(
                        document_id=record.document_id,
                        text=sentence.text,
                        char_start=sentence.char_start,
                        char_end=sentence.char_end,
                        sentence_index=sentence.index,
                        score=_cosine(vector, sentence_vector),
                    )
                )
        scored.sort(key=lambda s: (-s.score, s.document_id, s.sentence_index))
        return scored[:top_k]


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
    """In-memory :class:`~graph_rag.ports.EntityStore` (V4-active).

    Full-fidelity fake backing the EL fast E2E: a dict keyed by ``canonical_id``,
    type + normalized-name (or alias) blocking, and brute-force cosine kNN over
    entity vectors. Deterministic, ``$0``, no Docker — the same external behaviour
    the real Elasticsearch adapter is proved against by its contract test.
    """

    def __init__(self) -> None:
        self._entities: dict[str, CanonicalEntity] = {}

    def upsert(self, entity: CanonicalEntity) -> None:
        """Insert or overwrite the entity keyed by ``entity.canonical_id`` (idempotent)."""
        self._entities[entity.canonical_id] = entity

    def get(self, canonical_id: str) -> CanonicalEntity | None:
        """Return the canonical entity for ``canonical_id``, or ``None`` if absent."""
        return self._entities.get(canonical_id)

    def block_candidates(
        self, *, entity_type: CuratedType, normalized_name: str
    ) -> list[CanonicalEntity]:
        """Return entities of ``entity_type`` whose name/alias normalizes to ``normalized_name``.

        Blocking key uses the shared :func:`~graph_rag.normalize.normalize_name`
        so the fake and the real adapter block identically (ADR-0004).
        """
        candidates: list[CanonicalEntity] = []
        for entity in self._entities.values():
            if entity.type != entity_type:
                continue
            keys = {normalize_name(entity.name)}
            keys.update(normalize_name(alias) for alias in entity.aliases)
            if normalized_name in keys:
                candidates.append(entity)
        return candidates

    def knn(
        self,
        *,
        vector: list[float],
        entity_type: CuratedType | None = None,
        top_k: int,
    ) -> list[tuple[CanonicalEntity, float]]:
        """Return the ``top_k`` entities nearest ``vector`` by cosine, descending.

        Skips entities with no stored ``vector``; optionally restricts to a single
        ``entity_type``. Ties keep insertion order (Python's stable sort).
        """
        scored: list[tuple[CanonicalEntity, float]] = []
        for entity in self._entities.values():
            if entity_type is not None and entity.type != entity_type:
                continue
            if entity.vector is None:
                continue
            scored.append((entity, _cosine(vector, entity.vector)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return the number of canonical entities stored."""
        return len(self._entities)

    def all(self) -> list[CanonicalEntity]:
        """Return every stored canonical entity (insertion order)."""
        return list(self._entities.values())


class InMemoryGraphStore:
    """In-memory :class:`~graph_rag.ports.GraphStore` (V5-active).

    Full-fidelity fake backing the KG-build fast E2E: nodes in a dict keyed by
    ``canonical_id`` (the multi-label ``:Entity:Type`` node is modeled by storing
    the :class:`~graph_rag.models.CanonicalEntity`, whose ``type`` is the second
    label) and edges as a list of :class:`~graph_rag.models.Triple`. The
    load-bearing behaviours the tests rely on — idempotent node upsert, doc-scoped
    edge delete (so re-ingest overwrites), and a real BFS ``khop`` — mirror the
    external behaviour the real Neo4j adapter is proved against by its contract
    test. Deterministic, ``$0``, no Docker.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CanonicalEntity] = {}
        self._edges: list[Triple] = []

    def upsert_entities(self, entities: list[CanonicalEntity]) -> None:
        """Create/merge nodes keyed by ``canonical_id`` (idempotent overwrite)."""
        for entity in entities:
            self._nodes[entity.canonical_id] = entity

    def write_triples(self, triples: list[Triple]) -> None:
        """Append each triple as a provenance-carrying edge."""
        self._edges.extend(triples)

    def delete_document_edges(self, source_doc_id: str) -> None:
        """Remove every edge whose provenance ``source_doc_id`` matches (nodes kept).

        Called before re-writing a document's triples so re-ingest overwrites
        rather than duplicates (graph idempotency, TESTING.md gap #1).
        """
        self._edges = [
            edge for edge in self._edges if edge.provenance.source_doc_id != source_doc_id
        ]

    def khop(self, seed_ids: list[str], hops: int) -> Subgraph:
        """Return the connected subgraph within ``hops`` of ``seed_ids`` (BFS).

        Edges are traversed undirected. Seeds present in the graph are hop 0;
        each hop adds the nodes one edge further out. ``edges`` are every stored
        edge whose endpoints are both in the reached node set. Unknown seed IDs
        are ignored.
        """
        visited: set[str] = {sid for sid in seed_ids if sid in self._nodes}
        frontier: set[str] = set(visited)
        for _ in range(max(hops, 0)):
            next_frontier: set[str] = set()
            for edge in self._edges:
                if edge.subject_id in frontier and edge.object_id in self._nodes:
                    next_frontier.add(edge.object_id)
                if edge.object_id in frontier and edge.subject_id in self._nodes:
                    next_frontier.add(edge.subject_id)
            next_frontier -= visited
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        # Deterministic node order: node-insertion order, filtered to visited.
        nodes = [node for cid, node in self._nodes.items() if cid in visited]
        edges = [
            edge for edge in self._edges if edge.subject_id in visited and edge.object_id in visited
        ]
        return Subgraph(nodes=nodes, edges=edges)

    def node_count(self) -> int:
        """Return the number of nodes stored."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the number of edges stored."""
        return len(self._edges)

    def get_node(self, canonical_id: str) -> CanonicalEntity | None:
        """Return the node for ``canonical_id``, or ``None`` if absent."""
        return self._nodes.get(canonical_id)

    def get_node_edges(self, canonical_id: str) -> list[Triple]:
        """Return every edge incident to ``canonical_id`` (as subject or object)."""
        return [edge for edge in self._edges if canonical_id in (edge.subject_id, edge.object_id)]


class FakeLLMClient:
    """Canned :class:`~graph_rag.ports.LLMClient` for the fast suite (V3-active).

    Returns canned STRUCTURED responses so :class:`LLMCorefStage` runs against it
    deterministically with no provider call, and counts every call on the public
    :attr:`calls` counter so a test can assert how many LLM calls happened (e.g.
    that a cached / re-run path did not recompute). Keeping the fast suite on this
    fake makes the gate LLM-free and ``$0`` (ADR-0010): the canned response is the
    fixture, standing in for the ``sha256`` response cache the real client uses.
    """

    def __init__(
        self,
        clusters: list[CorefCluster] | None = None,
        structured_response: Any = None,
        completion: str = "",
    ) -> None:
        """Configure the canned output.

        Args:
            clusters: Canned coref clusters; wrapped into whatever schema
                :meth:`structured` is asked for (a ``ClusterMap``-shaped payload).
            structured_response: An explicit canned instance to return from
                :meth:`structured`, overriding ``clusters`` (for other schemas).
            completion: The canned string returned from :meth:`complete`.
        """
        self._clusters = clusters
        self._structured_response = structured_response
        self._completion = completion
        self.calls = 0

    def complete(self, prompt: str, **params: Any) -> str:
        """Return the canned completion string, counting the call."""
        self.calls += 1
        return self._completion

    def structured(self, prompt: str, schema: type[_StructuredT], **params: Any) -> _StructuredT:
        """Return a canned ``schema`` instance, counting the call.

        Uses an explicit ``structured_response`` if configured, else builds a
        ``ClusterMap``-shaped payload from the canned ``clusters``, else an empty
        instance — so the fast suite is deterministic and offline.
        """
        self.calls += 1
        if self._structured_response is not None:
            return self._structured_response
        if self._clusters is not None:
            payload = {"clusters": [c.model_dump() for c in self._clusters]}
            return schema.model_validate(payload)
        return schema()


class FakeEmbedder:
    """Deterministic :class:`~graph_rag.ports.Embedder` for the fast suite (V4-active).

    Pure-Python feature-hashing embedder — no torch, no model download, instant.
    Each text is tokenized with the shared :func:`~graph_rag.normalize.normalize_name`
    rule, each token is hashed into a signed bucket of a ``dim``-length vector, and
    the vector is L2-normalized. Consequences the fast suite relies on:

    * **Deterministic:** identical text → identical vector.
    * **Collidable:** texts that share normalized tokens have overlapping buckets
      and therefore high cosine similarity, so a test can make two surface forms
      "match" (score above threshold) or "not match" by construction — without a
      real model.
    """

    def __init__(self, dim: int = 384) -> None:
        """Configure the output dimension (defaults to B1's 384-dim ``bge-small``)."""
        self._dim = dim

    @property
    def dim(self) -> int:
        """The embedding dimension."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic L2-normalized vector per text, in order."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Feature-hash one text into a unit vector of length :attr:`dim`."""
        vec = [0.0] * self._dim
        tokens = normalize_name(text).split()
        if not tokens:
            # Empty/all-punctuation text: hash the raw text into one bucket so the
            # vector is still deterministic and non-zero.
            tokens = [text or "\x00"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dim
            sign = 1.0 if digest[8] & 1 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            # Pathological cancellation: seed a stable bucket so the result is a
            # valid unit vector rather than all-zeros.
            vec[len(text) % self._dim] = 1.0
            return vec
        return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


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
