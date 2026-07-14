"""Fast E2E for the V6 ``POST /query`` retrieval path — fakes only, no Docker ($0 gate).

Seeds a small **cross-document, multi-hop** scenario into the in-memory stores
(ADR-0010) — entity A linked to B in one document, B linked to C in another — and
drives BOTH the :class:`~graph_rag.query.retriever.QueryRetriever` directly AND the
FastAPI ``POST /query`` endpoint over the same fakes. Proves the payoff slice:
answer a multi-hop question with a CONNECTED subgraph + supporting sentences with
provenance, **no LLM** (no ``LLMClient`` is wired anywhere on this path), $0.

NOT marked ``contract``/``model``/``llm`` — part of the fast, offline gate.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
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

pytestmark = pytest.mark.e2e

# The cross-document, multi-hop fixture (A —doc1— B —doc2— C).
_QUESTION = "Who was Ada Lovelace?"
_ID_A = "e-ada"  # Ada Lovelace   (PERSON) — the intended, unambiguous answer
_ID_B = "e-engine"  # Analytical Engine (PRODUCT) — the bridge node
_ID_C = "e-babbage"  # Charles Babbage (PERSON) — two hops from the seed
_SENT_D1 = "Ada Lovelace worked on the Analytical Engine."
_SENT_D2 = "The Analytical Engine was designed by Charles Babbage."


def _seed_stores(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
) -> None:
    """Seed the multi-hop, cross-document scenario into the fakes.

    Only entity A carries an entity vector, so the question kNN seeds A alone; B
    and C are reached purely by graph expansion (A→B one hop, B→C two hops). The
    two edges' provenance point at two different documents, so the retrieved
    subgraph spans documents.
    """
    # --- ES-Entities: only A is vector-seedable, and its vector == the question's
    #     embedding, so its seed cosine is 1.0 and A is unambiguously top-ranked.
    entity_store.upsert(
        CanonicalEntity(
            canonical_id=_ID_A,
            name="Ada Lovelace",
            type="PERSON",
            vector=embedder.embed([_QUESTION])[0],
        )
    )

    # --- Knowledge graph: three nodes, two provenance-carrying edges across docs.
    node_a = CanonicalEntity(canonical_id=_ID_A, name="Ada Lovelace", type="PERSON")
    node_b = CanonicalEntity(canonical_id=_ID_B, name="Analytical Engine", type="PRODUCT")
    node_c = CanonicalEntity(canonical_id=_ID_C, name="Charles Babbage", type="PERSON")
    graph_store.upsert_entities([node_a, node_b, node_c])
    graph_store.write_triples(
        [
            Triple(
                subject_id=_ID_A,
                predicate="RELATED_TO",
                object_id=_ID_B,
                provenance=EdgeProvenance(
                    source_doc_id="doc1", sentence_index=0, source_sentence=_SENT_D1
                ),
            ),
            Triple(
                subject_id=_ID_B,
                predicate="PRODUCES",
                object_id=_ID_C,
                provenance=EdgeProvenance(
                    source_doc_id="doc2", sentence_index=0, source_sentence=_SENT_D2
                ),
            ),
        ]
    )

    # --- ES-Documents: per-sentence text + offsets + vectors for passage kNN.
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


def _assert_multihop_response(response: QueryResponse) -> None:
    """Assert the response is the connected, provenance-carrying multi-hop answer."""
    # 1. The subgraph is CONNECTED: it contains the A–B–C path, not isolated nodes.
    node_ids = {node.canonical_id for node in response.subgraph.nodes}
    assert node_ids == {_ID_A, _ID_B, _ID_C}
    edge_pairs = {(e.subject_id, e.object_id) for e in response.subgraph.edges}
    assert (_ID_A, _ID_B) in edge_pairs  # doc1 hop
    assert (_ID_B, _ID_C) in edge_pairs  # doc2 hop — spans documents

    # 2. The answer is the top-ranked node — unambiguously entity A (seed cosine 1.0
    #    at hop 0 beats every non-seed node's proximity-only score).
    assert response.answer_entity is not None
    assert response.answer_entity.canonical_id == _ID_A
    assert response.answer == "Ada Lovelace"
    assert response.ranked_nodes[0].canonical_id == _ID_A

    # 3. Supporting sentences come back WITH provenance (doc_id + offsets).
    assert response.supporting_sentences
    top_sentence = response.supporting_sentences[0]
    assert top_sentence.document_id in {"doc1", "doc2"}
    assert top_sentence.char_start == 0
    assert top_sentence.char_end == len(top_sentence.text)
    assert top_sentence.text in {_SENT_D1, _SENT_D2}


def test_retriever_answers_multihop_question(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
    retriever: QueryRetriever,
) -> None:
    """The retriever (driven directly) answers the multi-hop, cross-doc question.

    The ``retriever`` fixture shares these store fixtures, so seeding them here
    seeds the retriever. No LLM client is wired anywhere on this path.
    """
    _seed_stores(embedder, entity_store, document_store, graph_store)

    response = retriever.retrieve(QueryRequest(question=_QUESTION))

    _assert_multihop_response(response)


def test_query_endpoint_returns_response(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
    retriever: QueryRetriever,
) -> None:
    """POST /query returns 200 + JSON matching ``QueryResponse`` for the same scenario."""
    _seed_stores(embedder, entity_store, document_store, graph_store)
    app = create_app(
        InMemoryObjectStore(),
        InMemoryTriggerPublisher(),
        Settings(),
        retriever=retriever,
    )
    client = TestClient(app)

    http_response = client.post("/query", json={"question": _QUESTION})

    assert http_response.status_code == 200
    # The body round-trips through the response schema (HTTP contract).
    response = QueryResponse.model_validate(http_response.json())
    _assert_multihop_response(response)


def test_query_endpoint_ignores_synthesize_flag(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
    retriever: QueryRetriever,
) -> None:
    """``synthesize=True`` does not error and does not change the deterministic path (V6)."""
    _seed_stores(embedder, entity_store, document_store, graph_store)
    app = create_app(
        InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings(), retriever=retriever
    )
    client = TestClient(app)

    http_response = client.post("/query", json={"question": _QUESTION, "synthesize": True})

    assert http_response.status_code == 200
    assert http_response.json()["answer"] == "Ada Lovelace"


def test_query_endpoint_503_without_retriever() -> None:
    """With no retriever wired, POST /query responds 503 (feature not configured)."""
    app = create_app(InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings())
    client = TestClient(app)

    http_response = client.post("/query", json={"question": _QUESTION})

    assert http_response.status_code == 503


def test_ingest_endpoints_unaffected_by_query_wiring(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
    retriever: QueryRetriever,
) -> None:
    """Wiring a retriever leaves ``GET /health`` (and the existing surface) working."""
    app = create_app(
        InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings(), retriever=retriever
    )
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
