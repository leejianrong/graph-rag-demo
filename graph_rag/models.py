"""Shared Pydantic v2 data models for the Graph RAG pipeline.

These models are part of the Slice V1 foundation contract. Adapters (Agent B) and
the orchestrator/tests (Agent C) code against them.

V1 models:

* :class:`IngestTrigger` — the Kafka trigger payload, carrying only
  ``{bucket, object_key}`` (ADR-0001).
* :class:`DocumentRecord` — the ``ES-Documents`` record. V1 writes ``text`` at
  ingestion; the enrichment fields (``mentions``, ``coref_clusters``,
  ``el_result``, ``sentence_vectors``) default empty/``None`` so raw-only V1–V3
  writes validate, and are populated together at the V4 entity-linking checkpoint.

V2 (NER) adds the in-memory enrichment carry (ADR-0002, ARCHITECTURE §4):

* :class:`Mention` / :class:`Sentence` — one NER mention (typed + char span) and
  one segmented sentence.
* :class:`PipelineResult` — the object the orchestrator RETURNS. It carries the
  raw :class:`DocumentRecord` plus the enrichment computed so far, held
  **in-memory** and NOT persisted to ES until the V4 EL checkpoint. Later slices
  extend it in place (V3 ``coref_clusters``, V4 ``el_result``).

V3 (coreference) adds the within-document coref cluster map (ADR-0003):

* :class:`CorefCluster` / :class:`ClusterMap` — a **non-destructive** grouping of
  coreferent mentions (including pronouns/repeats) onto a chosen in-document
  canonical surface form. ``ClusterMap`` is the Pydantic type the LLM structured
  output validates against; :class:`PipelineResult` carries the resulting
  ``coref_clusters`` in-memory (persisted at the V4 EL checkpoint). The original
  text is preserved — the map references surface forms, it never rewrites text.

V4 (entity linking) adds the corpus-local canonical store + per-document result
(ADR-0004/0005):

* :class:`CanonicalEntity` — one deduplicated ``ES-Entities`` record, keyed by
  ``canonical_id`` (the merge key / graph node identity), carrying the entity
  ``dense_vector`` that blocking + kNN search rank over.
* :class:`EntityLink` — one per-document EL result (doc-level entity →
  ``canonical_id``, with score + merge/create-new flag); :class:`PipelineResult`
  carries the list and it is persisted on the :class:`DocumentRecord` at the EL
  checkpoint.

V5 (knowledge-graph build) adds the graph triple + provenance (ADR-0006):

* :class:`EdgeProvenance` / :class:`Triple` — one graph edge
  ``(subject_id, predicate, object_id)`` over **canonical entity IDs** with a
  closed-set predicate, a DATE edge qualifier and per-edge provenance
  (source doc/sentence/span). :class:`Subgraph` is the ``khop`` traversal result
  (nodes + edges) V6 retrieval consumes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "IngestTrigger",
    "DocumentRecord",
    "CuratedType",
    "Mention",
    "Sentence",
    "CorefCluster",
    "ClusterMap",
    "CanonicalEntity",
    "EntityLink",
    "PipelineResult",
    "EdgeProvenance",
    "Triple",
    "Subgraph",
    "QueryRequest",
    "SupportingSentence",
    "RankedNode",
    "QueryResponse",
]

# The curated NER type set (ADR-0002): spaCy's OntoNotes labels narrowed to the
# types this graph needs. ``GPE`` and ``LOC`` both map to ``LOCATION``; labels
# outside this set are dropped. ``PRODUCT`` is optional-but-included.
CuratedType = Literal[
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "EVENT",
    "NORP",
    "PRODUCT",
]


class IngestTrigger(BaseModel):
    """The Kafka trigger payload — carries ONLY ``bucket`` and ``object_key``.

    Published by ``POST /ingest`` after the bytes land in the object store; the
    thin Kafka consumer resolves it to a ``process_document({bucket, object_key})``
    call (ADR-0001).
    """

    bucket: str
    object_key: str

    def to_json(self) -> str:
        """Serialize this trigger to a JSON string (Kafka message value)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> IngestTrigger:
        """Deserialize a Kafka message value (JSON ``str``/``bytes``) to a trigger."""
        return cls.model_validate_json(payload)


class DocumentRecord(BaseModel):
    """The ``ES-Documents`` record for one document.

    In V1 only ``document_id``, ``bucket``, ``object_key`` and ``text`` are set —
    the bare record created at ingestion, before processing. The enrichment
    fields default empty/``None`` so that raw-only V1–V3 writes still validate
    against one shared schema; they are populated together **in place at the V4
    entity-linking checkpoint** (ARCHITECTURE §4/§5, ADR-0001/0005) with the NER
    mentions, coref cluster map, per-document EL result and sentence vectors that
    the pipeline computed in-memory.

    The concrete sub-schemas are now pinned (V4): the fields carry
    :class:`Mention`, :class:`CorefCluster` and :class:`EntityLink` instances, so
    an enriched record round-trips through :meth:`to_json`/:meth:`from_json`.
    """

    document_id: str
    bucket: str
    object_key: str
    text: str  # raw document text, written at ingestion (V1)

    # --- Enrichment fields: written together at the EL checkpoint (V4) --------
    # Default-empty / None so a raw-only V1–V3 write validates unchanged; the EL
    # stage (Wave 2) sets them when it persists the enriched record in place.
    mentions: list[Mention] = Field(default_factory=list)  # NER (V2)
    coref_clusters: list[CorefCluster] = Field(default_factory=list)  # coref (V3)
    el_result: list[EntityLink] = Field(default_factory=list)  # per-doc EL (V4)
    # Per-sentence text + char offsets (from the V2 spaCy segmentation), persisted
    # so query-side passage/sentence kNN (V6, B5) can return the matched sentence
    # with its provenance offsets. Positionally aligned with ``sentence_vectors``
    # (``sentences[i]`` is embedded as ``sentence_vectors[i]``). Default-empty so a
    # raw-only V1–V3 write validates unchanged; the EL checkpoint sets it alongside
    # ``sentence_vectors``.
    sentences: list[Sentence] = Field(default_factory=list)  # segmentation (V2→V6)
    # Passage/sentence dense vectors for query-side seeding (ARCHITECTURE §5, B5);
    # None until the EL checkpoint embeds the document's sentences.
    sentence_vectors: list[list[float]] | None = None

    # Ignore anything from stored JSON that a later slice adds and this code
    # doesn't yet know about, rather than raising.
    model_config = {"extra": "ignore"}

    def to_json(self) -> str:
        """Serialize this record to a JSON string (e.g. for the document store)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> DocumentRecord:
        """Deserialize a JSON ``str``/``bytes`` document into a record."""
        return cls.model_validate_json(payload)


# --- V2 (NER) in-memory enrichment ------------------------------------------


class Mention(BaseModel):
    """One typed NER mention with character offsets into the raw document text.

    The offsets are half-open ``[char_start, char_end)`` slices of the *raw*
    ``DocumentRecord.text``, so ``text == raw[char_start:char_end]`` holds. They
    align coref mentions (V3), attach provenance to triples (V5) and drive UI
    highlighting (ADR-0002).
    """

    text: str
    type: CuratedType
    char_start: int
    char_end: int


class Sentence(BaseModel):
    """One segmented sentence with character offsets into the raw document text.

    Produced in the same spaCy pass as the mentions (ADR-0002). ``index`` is the
    zero-based position of the sentence in the document. ``text`` equals
    ``raw[char_start:char_end]``.
    """

    text: str
    char_start: int
    char_end: int
    index: int


# --- V3 (coreference) within-document cluster map ---------------------------


class CorefCluster(BaseModel):
    """One within-document coreference cluster (ADR-0003), non-destructively.

    Groups the surface forms that co-refer within a single document — including
    pronouns (``"she"``, ``"they"``, ``"it"``) and repeated names — onto a chosen
    in-document ``canonical`` surface form. This is a **map, not a rewrite**: the
    raw document text is preserved untouched, and ``canonical``/``members`` are
    verbatim surface strings drawn from it. Each document's clusters become the
    doc-level entities handed to entity linking at V4.
    """

    canonical: str  # the chosen in-document canonical surface form for the cluster
    members: list[str] = Field(
        default_factory=list
    )  # all coreferent surface forms (incl. pronouns/repeats), verbatim from the text


class ClusterMap(BaseModel):
    """The coref stage's structured output — the full set of clusters for a doc.

    This is the Pydantic type the LLM's structured/JSON output validates against
    (ADR-0008): a single JSON object wrapping the list of :class:`CorefCluster` s,
    so JSON-mode providers have an object (not a bare array) to return.
    """

    clusters: list[CorefCluster] = Field(default_factory=list)


# --- V4 (entity linking) canonical store + per-document EL result -----------


class CanonicalEntity(BaseModel):
    """One deduplicated corpus-wide entity — an ``ES-Entities`` record (ADR-0005).

    The corpus-local source of truth entity linking blocks/scores against
    (ADR-0004). ``canonical_id`` is the **merge key and graph node identity**:
    upsert is idempotent by it, and the same real-world entity mentioned across
    documents resolves to one ``CanonicalEntity``. ``name`` is the seed surface
    form (the first mention that created it); merged surface forms accumulate in
    ``aliases``. ``vector`` is the entity ``dense_vector`` (``bge-small-en-v1.5``,
    384-dim, B1) that the store's kNN search ranks over — ``None`` only for an
    entity created before its embedding is attached.
    """

    canonical_id: str
    name: str
    type: CuratedType
    aliases: list[str] = Field(default_factory=list)
    vector: list[float] | None = None


class EntityLink(BaseModel):
    """One per-document entity-linking result (ADR-0004/0005).

    Records that a doc-level entity (a coref cluster's canonical surface form,
    ``mention_text``) resolved to the canonical entity ``canonical_id`` of type
    ``entity_type``, with the embedding-similarity ``score`` that decided it and
    ``is_new`` telling merge (``False``) from create-new (``True``). The list of
    these is persisted on the :class:`DocumentRecord` at the EL checkpoint.
    """

    mention_text: str
    canonical_id: str
    entity_type: CuratedType
    score: float
    is_new: bool


class PipelineResult(BaseModel):
    """The object the orchestrator RETURNS — the in-memory enrichment carry.

    Bundles the raw :class:`DocumentRecord` (already persisted to ES at
    ingestion) with the enrichment computed so far in the pipeline. Per the write
    model (ARCHITECTURE §4, ADR-0001), this enrichment is held **in-memory** and
    is NOT persisted to ``ES-Documents`` until the V4 entity-linking checkpoint —
    in V2/V3 the ES record still stores raw text only.

    V2 populates ``mentions`` and ``sentences``; V3 adds ``coref_clusters`` (the
    non-destructive within-document cluster map); V4 adds ``el_result`` (the
    per-document entity-linking result); V5 adds ``triples`` (the knowledge-graph
    edges built + written at the graph checkpoint). At the EL checkpoint the
    orchestrator writes the enrichment back into ``record`` and persists it; the
    triples are carried in-memory for callers/tests (they live in Neo4j, not ES).
    """

    record: DocumentRecord
    mentions: list[Mention] = Field(default_factory=list)
    sentences: list[Sentence] = Field(default_factory=list)
    coref_clusters: list[CorefCluster] = Field(default_factory=list)
    el_result: list[EntityLink] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)  # KG edges (V5, in-memory)


# --- V5 (knowledge-graph build) triples + provenance ------------------------


class EdgeProvenance(BaseModel):
    """Per-edge provenance for one knowledge-graph triple (ADR-0006, §5c).

    Load-bearing for traceable answers: every edge records which document,
    sentence and exact span it came from, plus the model's original phrasing and
    confidence. The KG-build LLM cites only ``sentence_index`` per triple; the
    ``char_start``/``char_end`` offsets are resolved from **our own spaCy
    sentence segmentation** (ADR-0002), not by the LLM. ``raw_predicate`` is set
    only when the relation fell back to :attr:`~graph_rag.predicates.Predicate.RELATED_TO`
    (it preserves the original phrase; ``None`` on a clean predicate map).
    """

    source_doc_id: str
    sentence_index: int
    source_sentence: str
    raw_predicate: str | None = None
    confidence: float | None = None
    char_start: int | None = None
    char_end: int | None = None


class Triple(BaseModel):
    """One knowledge-graph edge: ``(subject_id, predicate, object_id)`` (ADR-0006).

    ``subject_id`` and ``object_id`` are **canonical entity IDs**
    (:attr:`CanonicalEntity.canonical_id`), NOT raw surface strings — grounding
    the graph in the EL store so the same entity across documents is one node and
    multi-hop traversal is reliable. ``predicate`` is a member of the closed set
    (:class:`graph_rag.predicates.Predicate`); when no primary relation fit it is
    ``"RELATED_TO"`` and ``provenance.raw_predicate`` holds the original phrase.
    ``date`` is the DATE **edge qualifier** (an ISO-ish date string) — dates are
    modeled as an attribute on the edge, never as standalone nodes.
    """

    subject_id: str
    predicate: str
    object_id: str
    provenance: EdgeProvenance
    date: str | None = None  # DATE qualifier on the edge (not a node)


class Subgraph(BaseModel):
    """A connected slice of the knowledge graph — the k-hop traversal result.

    Returned by :meth:`graph_rag.ports.GraphStore.khop`: the nodes reachable
    within ``hops`` of the seed IDs and every edge whose endpoints are both in
    that node set. V6 retrieval ranks over this (nodes + supporting-sentence
    provenance on the edges).
    """

    nodes: list[CanonicalEntity] = Field(default_factory=list)
    edges: list[Triple] = Field(default_factory=list)


# --- V6 (query) request/response schema (U3) --------------------------------


class QueryRequest(BaseModel):
    """The ``POST /query`` request body (U3, ADR-0007).

    Carries the natural-language ``question`` and the ``synthesize`` flag. The
    flag is the V7 gated-prose-synthesis switch; the field exists now so the
    request schema is stable, but V6 retrieval **ignores** it (the default,
    deterministic, ``$0`` retrieval path always runs).
    """

    question: str
    synthesize: bool = False

    def to_json(self) -> str:
        """Serialize this request to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> QueryRequest:
        """Deserialize a JSON ``str``/``bytes`` request body into a request."""
        return cls.model_validate_json(payload)


class SupportingSentence(BaseModel):
    """One retrieved supporting sentence with its provenance (U3, ADR-0007).

    The passage/sentence-anchored evidence for an answer: the sentence ``text``
    with its half-open ``[char_start, char_end)`` offsets into the source
    document's raw text, the document it came from, its zero-based
    ``sentence_index`` and the cosine ``score`` of its vector against the query
    vector. Returned by :meth:`graph_rag.ports.DocumentStore.search_sentences`.
    """

    document_id: str
    text: str
    char_start: int
    char_end: int
    sentence_index: int
    score: float


class RankedNode(BaseModel):
    """One scored knowledge-graph node in the ranked query result (U3, B4).

    A candidate answer entity: its ``canonical_id`` (graph node identity),
    ``name``, curated ``type`` and the ranking ``score`` assigned by
    :func:`graph_rag.query.ranking.rank_nodes`. The top-ranked node is the
    predicted entity answer for entity-typed questions (ADR-0007).
    """

    canonical_id: str
    name: str
    type: CuratedType
    score: float


class QueryResponse(BaseModel):
    """The ``POST /query`` response body (U3, ADR-0007).

    The deterministic retrieval result: the predicted ``answer`` (the top-ranked
    entity's name, or ``None`` when no entity was retrieved) and the same node as
    ``answer_entity``; the connected ``subgraph`` (nodes + provenance-carrying
    :class:`Triple` edges) the answer was read from; every candidate node in
    ``ranked_nodes`` (score-descending); and the ``supporting_sentences`` evidence
    with per-sentence provenance. Reusing :class:`Subgraph` means per-edge
    provenance rides along for traceable answers.

    ``prose`` is the OPTIONAL V7 gated LLM synthesis (ADR-0009): ``None`` on the
    default deterministic path (``synthesize=false``, no LLM call), and set to the
    grounded prose answer only when ``POST /query`` was called with
    ``synthesize=true`` and a synthesizer is wired. Defaulting it to ``None`` keeps
    the V6 response shape byte-for-byte unchanged when synthesis is off.
    """

    answer: str | None
    answer_entity: RankedNode | None
    subgraph: Subgraph
    ranked_nodes: list[RankedNode] = Field(default_factory=list)
    supporting_sentences: list[SupportingSentence] = Field(default_factory=list)
    # V7 gated synthesis: None unless ``synthesize=true`` produced grounded prose.
    prose: str | None = None

    def to_json(self) -> str:
        """Serialize this response to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str | bytes) -> QueryResponse:
        """Deserialize a JSON ``str``/``bytes`` response into a response."""
        return cls.model_validate_json(payload)


# ``DocumentRecord`` and ``PipelineResult`` annotate fields with model types
# defined later in this module (``from __future__ import annotations`` defers all
# annotations to strings). Rebuild them now that every referenced name is in the
# module namespace so Pydantic resolves the forward references.
DocumentRecord.model_rebuild()
PipelineResult.model_rebuild()
Triple.model_rebuild()
Subgraph.model_rebuild()
QueryResponse.model_rebuild()
