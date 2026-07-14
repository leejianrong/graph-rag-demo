"""The KG-build stage (N9) — LLM emits triples over canonical IDs (ADR-0006).

The fourth and final enrichment stage. It turns a processed document into graph
structure: the LLM receives the document text plus **that document's canonical
linked entities** (from V4 entity linking) and emits triples
``(subject_id, predicate, object_id)`` whose subject/object are **canonical entity
IDs** — never raw surface strings — so the graph is grounded in the EL store and
the same entity across documents is one node (ADR-0006, ARCHITECTURE §5c).

Four rules turn a raw LLM triple into a :class:`~graph_rag.models.Triple`:

1. **Predicate → closed set.** Each raw predicate phrase is mapped via
   :func:`~graph_rag.predicates.map_predicate` to the closest of the closed
   ~12-predicate set; when nothing fits it becomes ``RELATED_TO`` and the original
   phrase is preserved as :attr:`~graph_rag.models.EdgeProvenance.raw_predicate`
   (nothing is lost).
2. **Offsets from OUR segmentation, not the LLM.** The LLM cites only a
   ``sentence_index``; the ``source_sentence`` text and ``char_start``/``char_end``
   are resolved from **our own spaCy sentence segmentation** (the ``sentences``
   from N6, ADR-0002) — the LLM cannot count characters reliably. An out-of-range
   index is logged and the triple skipped.
3. **DATE is an edge qualifier.** A dated fact sets :attr:`~graph_rag.models.Triple.date`
   (an attribute on the edge), never a standalone DATE node.
4. **Canonical-ID validation.** A triple whose subject or object is not one of this
   document's canonical IDs is logged and dropped — the graph never grows a
   dangling edge to an unknown node.

Like the NER/coref/EL stages, the stage runs behind the :class:`KgStage` seam and
is constructor-injected (ADR-0010): the fast suite injects it over a
:class:`~graph_rag.fakes.FakeLLMClient` (canned triples, ``$0``, offline); the real
stack injects it over the LiteLLM client from :class:`~graph_rag.config.Settings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from graph_rag.logging import get_logger
from graph_rag.models import EdgeProvenance, Triple
from graph_rag.predicates import map_predicate

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import CanonicalEntity, Sentence
    from graph_rag.ports import LLMClient
    from graph_rag.stages.entity_linking import ELResult

__all__ = [
    "KgStage",
    "KgBuildStage",
    "LLMTriple",
    "TripleList",
    "build_kg_prompt",
]

_logger = get_logger(__name__)


class LLMTriple(BaseModel):
    """One triple as emitted by the KG-build LLM (ADR-0006).

    ``subject_id`` / ``object_id`` are **canonical entity IDs** drawn from the doc's
    linked entities (the model is handed the id↔name/type map, so it references IDs
    not surface strings). ``predicate`` is the model's free-text relation phrase
    (mapped to the closed set by :func:`~graph_rag.predicates.map_predicate`).
    ``sentence_index`` cites the source sentence in OUR segmentation — the stage,
    not the LLM, resolves char offsets from it. ``date`` is the optional DATE edge
    qualifier; ``confidence`` the optional model confidence.
    """

    subject_id: str
    predicate: str
    object_id: str
    sentence_index: int
    date: str | None = None
    confidence: float | None = None


class TripleList(BaseModel):
    """Structured-output wrapper — the list of triples the LLM emits for one doc.

    A single JSON object wrapping the list (not a bare array) so JSON-mode
    providers have an object to return, mirroring
    :class:`~graph_rag.models.ClusterMap` (ADR-0008). This is the Pydantic
    ``schema`` handed to :meth:`~graph_rag.ports.LLMClient.structured`.
    """

    triples: list[LLMTriple] = Field(default_factory=list)


def build_kg_prompt(text: str, canonical_entities: list[CanonicalEntity]) -> str:
    """Render the KG-build prompt from the doc text + its canonical linked entities.

    A pure function so the prompt is deterministic (stable cache key) and
    unit-inspectable. It hands the model the **canonical id↔name/type map** and
    asks it to emit triples that reference those IDs (not surface strings), citing
    the sentence index each relation came from (ADR-0006).

    Args:
        text: The raw document text (the sentence indices reference its segmentation).
        canonical_entities: This document's canonical linked entities (the allowed
            subject/object IDs).

    Returns:
        The rendered prompt string (the structured-output schema instruction is
        appended by the client's ``structured`` call).
    """
    listed = (
        "\n".join(
            f"- {entity.canonical_id}: {entity.name} ({entity.type})"
            for entity in canonical_entities
        )
        or "- (none)"
    )
    return (
        "You are building a knowledge graph from a single document. Extract the "
        "relations between the CANONICAL ENTITIES listed below and emit them as "
        "triples (subject_id, predicate, object_id). The subject_id and object_id "
        "MUST be canonical entity IDs from the list — never surface strings. Use a "
        "short predicate phrase for the relation. For each triple cite the "
        "zero-based sentence_index of the sentence it came from. If a relation is "
        "dated, put the date on the triple's `date` field (do NOT emit a date as "
        "an entity). Only emit relations actually stated in the document.\n\n"
        f"Document text:\n{text}\n\n"
        f"Canonical entities (id: name (type)):\n{listed}"
    )


@runtime_checkable
class KgStage(Protocol):
    """The KG-build stage seam: doc enrichment → knowledge-graph triples.

    Constructor-injected into the orchestrator; the fast suite injects a
    :class:`KgBuildStage` over a :class:`~graph_rag.fakes.FakeLLMClient` instead of
    calling a live provider. Implementations return triples over **canonical IDs**
    with a closed-set predicate and per-edge provenance (offsets resolved from the
    supplied segmentation, never the LLM).
    """

    def build(
        self,
        document_id: str,
        text: str,
        sentences: list[Sentence],
        el_result: ELResult,
        canonical_entities: list[CanonicalEntity],
    ) -> list[Triple]:
        """Return the knowledge-graph triples for one document."""
        ...


class KgBuildStage:
    """Real :class:`KgStage` — LLM-backed structured triple extraction (ADR-0006).

    Constructor-injected with the :class:`~graph_rag.ports.LLMClient` it emits
    triples through, so the provider/model, response cache and retry all live in
    the client (V5-active). Deterministic for a fixed client + inputs.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Wire the stage to an LLM client.

        Args:
            llm_client: The client to call. Defaults to a real
                :class:`~graph_rag.adapters.llm_client.LiteLLMClient` built from
                :class:`~graph_rag.config.Settings`; the fast suite injects a
                :class:`~graph_rag.fakes.FakeLLMClient` so no provider is called.
        """
        if llm_client is None:
            from graph_rag.adapters.llm_client import LiteLLMClient
            from graph_rag.config import get_settings

            llm_client = LiteLLMClient.from_settings(get_settings())
        self._llm = llm_client

    @classmethod
    def from_settings(cls, settings: Settings) -> KgBuildStage:
        """Construct with a :class:`LiteLLMClient` on ``settings.kg_build_model`` (B6).

        Builds the client directly (rather than ``LiteLLMClient.from_settings``,
        which pins the coref model) so KG-build uses its own ``kg_build_model``
        while sharing the LLM cache dir, retry budget and env-sourced API key.
        """
        from graph_rag.adapters.llm_client import LiteLLMClient

        return cls(
            LiteLLMClient(
                model=settings.kg_build_model,
                cache_dir=settings.llm_cache_dir,
                max_retries=settings.llm_max_retries,
                api_key=settings.openai_api_key,
            )
        )

    def build(
        self,
        document_id: str,
        text: str,
        sentences: list[Sentence],
        el_result: ELResult,
        canonical_entities: list[CanonicalEntity],
    ) -> list[Triple]:
        """Emit the document's triples over canonical IDs with resolved provenance.

        Runs the LLM once (structured output) to emit raw triples, then for each:
        validate its subject/object are known canonical IDs (drop + log otherwise);
        resolve the cited ``sentence_index`` against OUR segmentation to fill the
        provenance sentence text + char span (drop + log an out-of-range index);
        map the raw predicate to the closed set (preserving the raw phrase on the
        ``RELATED_TO`` fallback); and carry any DATE as the edge qualifier.

        Args:
            document_id: The source document's deterministic id (edge provenance).
            text: The raw document text (fed to the model; offsets index into it).
            sentences: OUR spaCy sentence segmentation (N6) — the offset source.
            el_result: The per-document entity-linking result (unused directly here;
                accepted so the seam carries the full EL output).
            canonical_entities: This document's canonical linked entities — the only
                allowed subject/object IDs.

        Returns:
            The list of :class:`~graph_rag.models.Triple` s referencing canonical IDs.
        """
        known_ids = {entity.canonical_id for entity in canonical_entities}
        if not known_ids:
            # No canonical entities → no valid edges possible; skip the LLM call.
            return []

        prompt = build_kg_prompt(text, canonical_entities)
        response = self._llm.structured(prompt, TripleList)

        sentences_by_index = {sentence.index: sentence for sentence in sentences}
        triples: list[Triple] = []
        for raw in response.triples:
            if raw.subject_id not in known_ids or raw.object_id not in known_ids:
                _logger.debug(
                    "dropping triple with unknown canonical id(s): %s -[%s]-> %s",
                    raw.subject_id,
                    raw.predicate,
                    raw.object_id,
                )
                continue

            sentence = sentences_by_index.get(raw.sentence_index)
            if sentence is None:
                _logger.debug(
                    "dropping triple citing out-of-range sentence_index %d (doc %s)",
                    raw.sentence_index,
                    document_id,
                )
                continue

            predicate, raw_predicate = map_predicate(raw.predicate)
            triples.append(
                Triple(
                    subject_id=raw.subject_id,
                    predicate=predicate,
                    object_id=raw.object_id,
                    date=raw.date,  # DATE is an edge qualifier, never a node.
                    provenance=EdgeProvenance(
                        source_doc_id=document_id,
                        sentence_index=raw.sentence_index,
                        # Offsets resolved from OUR segmentation, not the LLM.
                        source_sentence=sentence.text,
                        char_start=sentence.char_start,
                        char_end=sentence.char_end,
                        raw_predicate=raw_predicate,
                        confidence=raw.confidence,
                    ),
                )
            )
        return triples
