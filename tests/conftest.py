"""Shared pytest fixtures.

Exposes the in-memory fakes (ADR-0010) as fixtures — the backbone of the fast,
$0, no-Docker suite. Both entry points are driven through this port seam.
"""

from __future__ import annotations

import pytest

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
    """A fresh in-memory GraphStore fake (stub until V5)."""
    return InMemoryGraphStore()


@pytest.fixture
def llm_client() -> FakeLLMClient:
    """A fresh LLMClient fake (stub until V3)."""
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
