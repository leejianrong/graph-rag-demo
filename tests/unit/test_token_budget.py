"""A recall result cut to a token budget keeps the top of the ranking (GRA-38)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.models import (
    CanonicalEntity,
    EdgeProvenance,
    QueryResponse,
    RankedNode,
    Subgraph,
    SupportingSentence,
    Triple,
)
from graph_rag.query.budget import CHARS_PER_TOKEN, evidence_tokens, pack
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain


def big_response(n: int = 30) -> QueryResponse:
    nodes = [
        CanonicalEntity(canonical_id=f"e{i}", name=f"Entity {i}", type="ORG") for i in range(n)
    ]
    ranked = [
        RankedNode(canonical_id=e.canonical_id, name=e.name, type="ORG", score=1 - i / n)
        for i, e in enumerate(nodes)
    ]
    edges = [
        Triple(
            subject_id=f"e{i}",
            predicate="RELATED_TO",
            object_id=f"e{i + 1}",
            provenance=EdgeProvenance(
                source_doc_id="d",
                sentence_index=i,
                source_sentence=f"Entity {i} relates to {i + 1}.",
            ),
        )
        for i in range(n - 1)
    ]
    sentences = [
        SupportingSentence(
            document_id="d",
            text=f"Sentence number {i} says something.",
            char_start=i,
            char_end=i + 30,
            sentence_index=i,
            score=1 - i / n,
        )
        for i in range(n)
    ]
    return QueryResponse(
        answer="Entity 0",
        answer_entity=ranked[0],
        subgraph=Subgraph(nodes=nodes, edges=edges),
        ranked_nodes=ranked,
        supporting_sentences=sentences,
        prose="kept whole",
    )


@pytest.mark.parametrize("budget", [1, 5, 20, 100, 400, 1000, 5000])
def test_the_output_never_exceeds_the_budget(budget: int) -> None:
    packed, _ = pack(big_response(), budget)
    assert evidence_tokens(packed) <= budget


def test_what_is_kept_is_the_top_of_each_ranking() -> None:
    full = big_response()
    packed, cut = pack(full, 400)
    assert cut
    for name in ("supporting_sentences", "ranked_nodes"):
        kept, original = getattr(packed, name), getattr(full, name)
        assert 0 < len(kept) < len(original)
        assert kept == original[: len(kept)]  # a prefix: nothing lower ranked jumped the queue


def test_a_bigger_budget_keeps_at_least_as_much() -> None:
    sizes = []
    for budget in (50, 200, 800, 3200):
        packed, _ = pack(big_response(), budget)
        sizes.append(
            (len(packed.supporting_sentences), len(packed.ranked_nodes), len(packed.subgraph.edges))
        )
    assert sizes == sorted(sizes)


def test_a_budget_that_holds_everything_changes_nothing() -> None:
    full = big_response()
    packed, cut = pack(full, 10**6)
    assert not cut and packed == full


def test_the_answer_and_prose_are_never_cut_and_edges_keep_their_nodes() -> None:
    packed, cut = pack(big_response(), 30)
    assert cut and packed.answer == "Entity 0" and packed.prose == "kept whole"
    assert packed.answer_entity is not None
    ids = {n.canonical_id for n in packed.subgraph.nodes}
    for edge in packed.subgraph.edges:
        assert {edge.subject_id, edge.object_id} <= ids


def test_the_sentences_cannot_take_the_whole_budget() -> None:
    packed, _ = pack(big_response(), 400)
    sentence_chars = sum(len(s.model_dump_json()) for s in packed.supporting_sentences)
    assert sentence_chars <= 400 * CHARS_PER_TOKEN * 0.6
    assert packed.ranked_nodes or packed.subgraph.edges


def test_a_budget_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        pack(big_response(), 0)


# --- through the API -------------------------------------------------------------------------


@pytest.fixture
def client():
    components = build_offline_components()
    queue = Queue()
    app = create_app(components.object_store, queue, Settings(), retriever=components.retriever)
    http = TestClient(app)
    text = " ".join(
        f"Person{i} works for Acme Corp. Acme Corp is based in London." for i in range(8)
    )
    http.post("/v1/remember", json={"text": text, "source_id": "n"})
    drain(components, queue)
    return http


def test_recall_without_a_budget_reports_usage_and_no_truncation(client) -> None:
    body = client.post("/v1/recall", json={"question": "Where is Acme Corp based?"}).json()
    assert body["truncated"] is False and body["tokens_used"] > 0


def test_recall_with_a_budget_is_cut_and_says_so(client) -> None:
    ask = {"question": "Where is Acme Corp based?"}
    full = client.post("/v1/recall", json=ask).json()
    small = client.post("/v1/recall", json=ask | {"token_budget": 60}).json()
    assert small["truncated"] is True and small["tokens_used"] <= 60 < full["tokens_used"]
    assert len(small["supporting_sentences"]) < len(full["supporting_sentences"])
    assert (
        small["supporting_sentences"]
        == full["supporting_sentences"][: len(small["supporting_sentences"])]
    )
    assert small["answer"] == full["answer"]


@pytest.mark.parametrize("budget", [0, -5, "many"])
def test_a_budget_of_zero_or_less_is_rejected(client, budget) -> None:
    response = client.post("/v1/recall", json={"question": "x", "token_budget": budget})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"
    assert "token_budget" in response.json()["error"]["message"]
