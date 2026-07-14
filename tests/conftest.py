"""Shared pytest fixtures.

Exposes the in-memory fakes (ADR-0010) as fixtures — the backbone of the fast,
$0, no-Docker suite. Both entry points are driven through this port seam.
"""

from __future__ import annotations

import pytest

from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    FakeNerStage,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
    InMemoryTriggerPublisher,
)
from graph_rag.query.retriever import QueryRetriever
from graph_rag.stages.coref import FakeCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.kg_build import KgBuildStage


@pytest.fixture
def object_store() -> InMemoryObjectStore:
    """A fresh in-memory ObjectStore fake (V1-active)."""
    return InMemoryObjectStore()


@pytest.fixture
def document_store() -> InMemoryDocumentStore:
    """A fresh in-memory DocumentStore fake (V1-active)."""
    return InMemoryDocumentStore()


@pytest.fixture
def trigger_publisher() -> InMemoryTriggerPublisher:
    """A fresh in-memory TriggerPublisher fake (V1-active); records to ``.published``."""
    return InMemoryTriggerPublisher()


@pytest.fixture
def entity_store() -> InMemoryEntityStore:
    """A fresh in-memory EntityStore fake (stub until V4)."""
    return InMemoryEntityStore()


@pytest.fixture
def graph_store() -> InMemoryGraphStore:
    """A fresh in-memory GraphStore fake (V5-active).

    Full-fidelity: idempotent node upsert, doc-scoped edge delete (re-ingest
    overwrite) and BFS k-hop. Backs the KG-build fast E2E over the port seam.
    """
    return InMemoryGraphStore()


@pytest.fixture
def llm_client() -> FakeLLMClient:
    """A fresh LLMClient fake (V3-active); canned structured responses, counts calls.

    Returns no canned clusters by default (an empty ``ClusterMap``); tests that
    need canned clusters construct their own
    :class:`~graph_rag.fakes.FakeLLMClient(clusters=...)`.
    """
    return FakeLLMClient()


@pytest.fixture
def embedder() -> FakeEmbedder:
    """A fresh Embedder fake (stub until V4)."""
    return FakeEmbedder()


@pytest.fixture
def ner_stage() -> FakeNerStage:
    """A fresh NER-stage fake (V2-active); returns no canned output by default.

    Injecting this into the orchestrator keeps the fast suite model-free: no
    spaCy model is loaded. Tests that need canned mentions/sentences construct
    their own :class:`~graph_rag.fakes.FakeNerStage` with the desired output.
    """
    return FakeNerStage()


@pytest.fixture
def coref_stage() -> FakeCorefStage:
    """A fresh coref-stage fake (V3-active); returns no canned clusters by default.

    Injecting this into the orchestrator keeps the fast suite LLM-free and ``$0``:
    no provider is called. Tests that need canned clusters construct their own
    :class:`~graph_rag.stages.coref.FakeCorefStage` with the desired output.
    """
    return FakeCorefStage()


@pytest.fixture
def entity_linking_stage(
    entity_store: InMemoryEntityStore, embedder: FakeEmbedder
) -> EntityLinkingStage:
    """A real EL stage (V4-active) over the in-memory EntityStore + FakeEmbedder.

    The stage itself is the real :class:`~graph_rag.stages.entity_linking.EntityLinkingStage`
    (fakes-first, ADR-0010): only its ports are faked, so the fast suite exercises
    the actual block/score/merge logic + the EL checkpoint — no Docker, no model,
    no LLM. Gated tie-breaker/NIL stay off (their defaults). Tests that need a
    tuned threshold or per-doc state construct their own instance.
    """
    return EntityLinkingStage(entity_store, embedder)


@pytest.fixture
def kg_build_stage(llm_client: FakeLLMClient) -> KgBuildStage:
    """A real KG-build stage (V5-active) over the ``FakeLLMClient`` (no provider).

    The stage itself is the real :class:`~graph_rag.stages.kg_build.KgBuildStage`
    (fakes-first, ADR-0010): only its LLM port is faked, so the fast suite
    exercises the actual predicate mapping + offset resolution + canonical-ID
    validation — no Docker, no model, no LLM. The default ``FakeLLMClient`` returns
    an empty ``TripleList``; tests that need canned triples construct their own
    :class:`~graph_rag.stages.kg_build.KgBuildStage(FakeLLMClient(structured_response=...))`.
    """
    return KgBuildStage(llm_client)


@pytest.fixture
def retriever(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
) -> QueryRetriever:
    """A V6 :class:`~graph_rag.query.retriever.QueryRetriever` over the in-memory fakes.

    Built from default :class:`~graph_rag.config.Settings` (the pinned B3/B4/B5
    knobs) over the SAME store fixtures, so a test seeds those stores and then
    drives the retriever (or ``POST /query``) against them — deterministic, ``$0``,
    no Docker/model/LLM (ADR-0010). No LLM client is wired anywhere on this path.
    """
    return QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    )
