"""Fast E2E for the V7 gated prose synthesis path — fakes only, no Docker ($0 gate).

Seeds a small multi-hop scenario into the in-memory stores (ADR-0010), wires a
:class:`~graph_rag.query.retriever.QueryRetriever` with an
:class:`~graph_rag.query.synthesis.AnswerSynthesizer` over a
:class:`~graph_rag.fakes.FakeLLMClient` (canned prose, ``$0``, offline), and drives
``POST /query`` via ``TestClient``. Proves the V7 payoff and its gate:

* ``synthesize=true`` → the response carries non-null ``prose`` grounded in the
  retrieved evidence, and the LLM WAS called (``FakeLLMClient.calls >= 1``);
* the DEFAULT request (no flag) → ``prose is None`` AND the LLM was NOT called
  (``calls == 0``), and the rest of the response equals the V6 result.

NOT marked ``contract``/``model``/``llm`` — part of the fast, offline gate.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
    InMemoryTriggerPublisher,
)
from graph_rag.models import (
    CanonicalEntity,
    DocumentRecord,
    EdgeProvenance,
    QueryRequest,
    QueryResponse,
    Sentence,
    Triple,
)
from graph_rag.query.retriever import QueryRetriever
from graph_rag.query.synthesis import AnswerSynthesizer

pytestmark = pytest.mark.e2e

_QUESTION = "Who was Ada Lovelace?"
_CANNED_PROSE = "Ada Lovelace worked on the Analytical Engine designed by Charles Babbage."
_SENT_D1 = "Ada Lovelace worked on the Analytical Engine."
_SENT_D2 = "The Analytical Engine was designed by Charles Babbage."


def _seed_stores(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
) -> None:
    """Seed the multi-hop, cross-document scenario into the fakes (A—B—C)."""
    entity_store.upsert(
        CanonicalEntity(
            canonical_id="e-ada",
            name="Ada Lovelace",
            type="PERSON",
            vector=embedder.embed([_QUESTION])[0],
        )
    )
    node_a = CanonicalEntity(canonical_id="e-ada", name="Ada Lovelace", type="PERSON")
    node_b = CanonicalEntity(canonical_id="e-engine", name="Analytical Engine", type="PRODUCT")
    node_c = CanonicalEntity(canonical_id="e-babbage", name="Charles Babbage", type="PERSON")
    graph_store.upsert_entities([node_a, node_b, node_c])
    graph_store.write_triples(
        [
            Triple(
                subject_id="e-ada",
                predicate="RELATED_TO",
                object_id="e-engine",
                provenance=EdgeProvenance(
                    source_doc_id="doc1", sentence_index=0, source_sentence=_SENT_D1
                ),
            ),
            Triple(
                subject_id="e-engine",
                predicate="PRODUCES",
                object_id="e-babbage",
                provenance=EdgeProvenance(
                    source_doc_id="doc2", sentence_index=0, source_sentence=_SENT_D2
                ),
            ),
        ]
    )
    document_store.upsert(
        DocumentRecord(
            document_id="doc1",
            bucket="documents",
            object_key="doc1.md",
            text=_SENT_D1,
            sentences=[Sentence(text=_SENT_D1, char_start=0, char_end=len(_SENT_D1), index=0)],
            sentence_vectors=[embedder.embed([_SENT_D1])[0]],
        )
    )
    document_store.upsert(
        DocumentRecord(
            document_id="doc2",
            bucket="documents",
            object_key="doc2.md",
            text=_SENT_D2,
            sentences=[Sentence(text=_SENT_D2, char_start=0, char_end=len(_SENT_D2), index=0)],
            sentence_vectors=[embedder.embed([_SENT_D2])[0]],
        )
    )


def _client_and_llm(synthesizer_wired: bool) -> tuple[TestClient, FakeLLMClient]:
    """Build a seeded ``POST /query`` app; optionally wire the V7 synthesizer.

    Returns the client and the shared :class:`FakeLLMClient` so a test can assert
    its ``.calls`` counter (proving whether the LLM was touched).
    """
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    _seed_stores(embedder, entity_store, document_store, graph_store)

    llm = FakeLLMClient(completion=_CANNED_PROSE)
    synthesizer = AnswerSynthesizer(llm) if synthesizer_wired else None
    retriever = QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
        synthesizer=synthesizer,
    )
    app = create_app(
        InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings(), retriever=retriever
    )
    return TestClient(app), llm


def test_query_with_synthesize_returns_prose_and_calls_llm() -> None:
    """``POST /query {synthesize: true}`` → non-null prose grounded in evidence; LLM called."""
    client, llm = _client_and_llm(synthesizer_wired=True)

    http_response = client.post("/query", json={"question": _QUESTION, "synthesize": True})

    assert http_response.status_code == 200
    body = http_response.json()
    response = QueryResponse.model_validate(body)

    # Prose is present and is the canned, evidence-grounded synthesis.
    assert response.prose == _CANNED_PROSE
    # The deterministic V6 core is unchanged underneath the prose.
    assert response.answer == "Ada Lovelace"
    # The LLM WAS called (the gate fired).
    assert llm.calls >= 1


def test_query_without_flag_makes_no_llm_call_and_matches_v6() -> None:
    """Default request → prose None, LLM NOT called, response equals the V6 result.

    Compares the gated app's default-request body against a retriever with NO
    synthesizer wired (the pure V6 path) minus the additive ``prose`` field.
    """
    client, llm = _client_and_llm(synthesizer_wired=True)

    http_response = client.post("/query", json={"question": _QUESTION})

    assert http_response.status_code == 200
    body = http_response.json()
    assert body["prose"] is None
    assert llm.calls == 0  # gate defaults OFF — no LLM call on the default path

    # The rest of the response equals the V6 result: build the same scenario with
    # NO synthesizer wired and compare everything except the additive prose field.
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    _seed_stores(embedder, entity_store, document_store, graph_store)
    v6 = QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    ).retrieve(QueryRequest(question=_QUESTION))

    gated = QueryResponse.model_validate(body)
    assert gated.prose is None
    assert gated.model_dump(exclude={"prose"}) == v6.model_dump(exclude={"prose"})
