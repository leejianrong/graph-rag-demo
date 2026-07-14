"""Offline, deterministic pipeline wiring for the benchmark (fakes-first, ADR-0010).

The benchmark harness (N18) is agnostic to which stages are wired — it drives the
V1 ingestion path (:meth:`~graph_rag.orchestrator.Orchestrator.process_document`)
and the V6 retriever (:meth:`~graph_rag.query.retriever.QueryRetriever.retrieve`)
through the port seam. This module supplies the **offline** wiring: in-memory
stores + a :class:`~graph_rag.fakes.FakeEmbedder` + light, deterministic,
text-driven stages, so ``benchmark run`` and the fast suite build a real graph and
answer real queries with **no Docker, no model download, and no provider call**
(deterministic, ``$0``).

The offline stages are demo-grade heuristics (NOT the real spaCy/LLM stages):

* :class:`HeuristicNerStage` — regex sentence segmentation + capitalized-span
  entity detection (leading stop-words stripped), typed uniformly (type does not
  affect the non-LLM retrieval scoring — ADR-0009 — but a uniform type keeps
  cross-document entity identity stable through the name-based EL merge below).
* coref — an :class:`~graph_rag.stages.coref.LLMCorefStage` over a
  :class:`~graph_rag.fakes.FakeLLMClient` returning an empty cluster map. It makes
  ONE LLM call per document, so the FakeLLMClient's ``calls`` counter models the
  ingestion cost the warm response cache eliminates on re-run (ADR-0008/0009).
* entity linking — the REAL :class:`~graph_rag.stages.entity_linking.EntityLinkingStage`
  over in-memory ports, pinned to merge-by-name (threshold ``-1``, kNN off) so the
  SAME surface across documents resolves to ONE canonical id — the cross-document
  bridge multi-hop retrieval needs — deterministically under the FakeEmbedder.
* :class:`HeuristicKgBuildStage` — a co-occurrence graph builder: entities sharing
  a sentence get a ``RELATED_TO`` edge with that sentence's provenance, so k-hop
  traversal has real structure to walk. No LLM.

The real-stack wiring for the opt-in integration run lives in ``main.py`` /
:func:`build_real_components`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.models import EdgeProvenance, Mention, Sentence, Triple
from graph_rag.orchestrator import Orchestrator
from graph_rag.predicates import Predicate
from graph_rag.query.retriever import QueryRetriever
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.ner import NerResult

if TYPE_CHECKING:
    from graph_rag.models import CanonicalEntity
    from graph_rag.stages.entity_linking import ELResult

__all__ = [
    "HeuristicNerStage",
    "HeuristicKgBuildStage",
    "BenchmarkComponents",
    "build_offline_components",
    "build_real_components",
]

# A capitalized span: one or more Title-Case tokens (allowing internal ' or -).
_TITLE_TOKEN = r"[A-Z][A-Za-z0-9]*(?:['\-][A-Za-z0-9]+)*"
_ENTITY_RE = re.compile(rf"{_TITLE_TOKEN}(?:\s+{_TITLE_TOKEN})*")
# One sentence: a run of non-terminator characters, optionally closed by . ! or ?.
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
# Leading tokens stripped off a captured span (sentence-initial articles /
# prepositions / pronouns that are not part of a proper name).
_LEADING_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "of",
        "for",
        "and",
        "but",
        "or",
        "he",
        "she",
        "it",
        "they",
        "his",
        "her",
        "its",
        "their",
        "who",
        "what",
        "when",
        "where",
        "which",
        "why",
        "how",
        "is",
        "was",
        "were",
        "are",
        "be",
        "after",
        "before",
        "during",
        "by",
        "to",
        "from",
        "as",
    }
)
# The uniform curated type assigned to every heuristic mention (see module docstring).
_HEURISTIC_TYPE = "ORG"


class HeuristicNerStage:
    """Deterministic, text-driven :class:`~graph_rag.stages.ner.NerStage` for offline runs.

    Segments sentences and detects capitalized-span mentions with a regex — no
    spaCy model. Pure and deterministic for a fixed input, so it backs a
    reproducible ``$0`` benchmark over arbitrary corpus text.
    """

    def analyze(self, text: str) -> NerResult:
        """Return regex-segmented sentences + capitalized-span mentions for ``text``."""
        sentences = self._segment(text)
        mentions = self._entities(text)
        return NerResult(mentions=mentions, sentences=sentences)

    @staticmethod
    def _segment(text: str) -> list[Sentence]:
        """Split ``text`` into sentences with exact char offsets (``text[s:e] == sent``)."""
        sentences: list[Sentence] = []
        index = 0
        for match in _SENTENCE_RE.finditer(text):
            raw = match.group()
            stripped = raw.strip()
            if not stripped:
                continue
            start = match.start() + (len(raw) - len(raw.lstrip()))
            end = start + len(stripped)
            sentences.append(
                Sentence(text=text[start:end], char_start=start, char_end=end, index=index)
            )
            index += 1
        return sentences

    @staticmethod
    def _entities(text: str) -> list[Mention]:
        """Detect capitalized spans, stripping leading stop-words; uniform type."""
        mentions: list[Mention] = []
        for match in _ENTITY_RE.finditer(text):
            start = match.start()
            span = match.group()
            # Strip leading stop-word tokens (e.g. a sentence-initial "The").
            while True:
                token_match = re.match(r"\S+\s*", span)
                if token_match is None:
                    break
                token = token_match.group().strip()
                if token.lower() in _LEADING_STOPWORDS:
                    consumed = token_match.end()
                    start += consumed
                    span = span[consumed:]
                    continue
                break
            span = span.strip()
            if not span or span.lower() in _LEADING_STOPWORDS:
                continue
            mentions.append(
                Mention(
                    text=span,
                    type=_HEURISTIC_TYPE,
                    char_start=start,
                    char_end=start + len(span),
                )
            )
        return mentions


class HeuristicKgBuildStage:
    """Deterministic co-occurrence :class:`~graph_rag.stages.kg_build.KgStage` (offline).

    Connects canonical entities that appear together in a sentence with a
    ``RELATED_TO`` edge carrying that sentence's provenance — giving k-hop
    traversal real structure to walk — with no LLM call. Deterministic: edges are
    emitted in sorted ``(subject_id, object_id, sentence_index)`` order.
    """

    def build(
        self,
        document_id: str,
        text: str,
        sentences: list[Sentence],
        el_result: ELResult,
        canonical_entities: list[CanonicalEntity],
    ) -> list[Triple]:
        """Emit ``RELATED_TO`` edges between entities co-occurring in a sentence."""
        known_ids = {entity.canonical_id for entity in canonical_entities}
        # canonical_id -> the surface strings that resolved to it (from the EL links).
        surfaces: dict[str, set[str]] = {}
        for link in el_result.links:
            if link.canonical_id in known_ids:
                surfaces.setdefault(link.canonical_id, set()).add(link.mention_text)

        triples: list[Triple] = []
        for sentence in sentences:
            present = sorted(
                cid
                for cid, texts in surfaces.items()
                if any(surface in sentence.text for surface in texts)
            )
            for i, subject_id in enumerate(present):
                for object_id in present[i + 1 :]:
                    triples.append(
                        Triple(
                            subject_id=subject_id,
                            predicate=Predicate.RELATED_TO,
                            object_id=object_id,
                            provenance=EdgeProvenance(
                                source_doc_id=document_id,
                                sentence_index=sentence.index,
                                source_sentence=sentence.text,
                                char_start=sentence.char_start,
                                char_end=sentence.char_end,
                            ),
                        )
                    )
        return triples


@dataclass
class BenchmarkComponents:
    """The wired pieces the :class:`~graph_rag.benchmark.harness.BenchmarkHarness` drives.

    Bundles the object store to write the corpus into, the ingestion orchestrator,
    the query retriever, the corpus bucket name and a ``llm_calls`` probe returning
    the cumulative provider-call count — so a re-run's ~$0 cost (no new ingestion
    calls, non-LLM retrieval) is observable.
    """

    object_store: InMemoryObjectStore | object
    orchestrator: Orchestrator
    retriever: QueryRetriever
    bucket: str
    llm_calls: Callable[[], int]


def build_offline_components(*, settings: Settings | None = None) -> BenchmarkComponents:
    """Wire the fully offline, deterministic benchmark pipeline (fakes + heuristics).

    Shares ONE :class:`~graph_rag.fakes.FakeEmbedder` and ONE set of in-memory
    stores between ingestion and retrieval (so the retriever reads exactly what was
    ingested), and threads a single :class:`~graph_rag.fakes.FakeLLMClient` through
    coref so its ``calls`` counter is the observable ingestion cost.

    Args:
        settings: Optional settings for the retrieval knobs (seeding depths, k-hop,
            ranking weights). Defaults to :class:`~graph_rag.config.Settings`.

    Returns:
        The wired :class:`BenchmarkComponents`.
    """
    settings = settings or Settings()

    object_store = InMemoryObjectStore()
    document_store = InMemoryDocumentStore()
    entity_store = InMemoryEntityStore()
    graph_store = InMemoryGraphStore()
    embedder = FakeEmbedder(dim=settings.embed_dim)
    llm_client = FakeLLMClient()  # empty coref clusters; counts one call per document.

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=HeuristicNerStage(),
        coref_stage=LLMCorefStage(llm_client),
        # Merge-by-name: threshold -1 + kNN off → same surface ⇒ one canonical id
        # (the deterministic cross-document bridge, robust under the FakeEmbedder).
        entity_linking_stage=EntityLinkingStage(
            entity_store, embedder, threshold=-1.0, knn_top_k=0
        ),
        graph_store=graph_store,
        kg_build_stage=HeuristicKgBuildStage(),
    )
    retriever = QueryRetriever.from_settings(
        settings,
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    )
    return BenchmarkComponents(
        object_store=object_store,
        orchestrator=orchestrator,
        retriever=retriever,
        bucket=settings.minio_bucket,
        llm_calls=lambda: llm_client.calls,
    )


def build_real_components(settings: Settings | None = None) -> BenchmarkComponents:
    """Wire the real-stack benchmark pipeline (MinIO + ES + Neo4j + spaCy + LLM).

    Mirrors ``graph_rag.main`` — real adapters behind every port, the real
    spaCy/LLM stages, and a retriever that reuses the same embedder + stores — for
    the opt-in, slow integration benchmark run. The LLM call count is not tracked
    at this seam (the real client's response cache makes a re-run ~$0 by serving
    cached completions), so ``llm_calls`` reports ``0``.

    Args:
        settings: The runtime settings; defaults to environment-driven
            :class:`~graph_rag.config.Settings`.

    Returns:
        The wired :class:`BenchmarkComponents` over the real adapters.
    """
    from graph_rag.adapters.embedder import SentenceTransformerEmbedder
    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore
    from graph_rag.adapters.llm_client import LiteLLMClient
    from graph_rag.adapters.minio_object_store import MinioObjectStore
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore
    from graph_rag.stages.kg_build import KgBuildStage
    from graph_rag.stages.ner import SpacyNerStage

    settings = settings or Settings()

    object_store = MinioObjectStore.from_settings(settings)
    document_store = EsDocumentStore.from_settings(settings)
    document_store.ensure_index()
    entity_store = EsEntityStore.from_settings(settings)
    entity_store.ensure_index()
    graph_store = Neo4jGraphStore.from_settings(settings)
    graph_store.init()
    embedder = SentenceTransformerEmbedder.from_settings(settings)

    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=SpacyNerStage.from_settings(settings),
        coref_stage=LLMCorefStage(LiteLLMClient.from_settings(settings)),
        entity_linking_stage=EntityLinkingStage.from_settings(settings, entity_store, embedder),
        graph_store=graph_store,
        kg_build_stage=KgBuildStage.from_settings(settings),
    )
    retriever = QueryRetriever.from_settings(
        settings,
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    )
    return BenchmarkComponents(
        object_store=object_store,
        orchestrator=orchestrator,
        retriever=retriever,
        bucket=settings.minio_bucket,
        llm_calls=lambda: 0,
    )
