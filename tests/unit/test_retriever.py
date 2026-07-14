"""Unit tests for the V6 :class:`~graph_rag.query.retriever.QueryRetriever` (fast, offline).

Covers the retriever's own composition logic — the parts NOT already unit-tested by
the pinned ranking function (``tests/unit/test_ranking.py``):

* **hop_distance BFS** over the returned subgraph (seeds = 0, unreachable omitted);
* **seed-merge (B5)** — entity kNN seeds anchor the graph while sentence kNN seeds
  anchor the evidence, and both ride the response;
* **top-node answer selection** on a crafted, seeded scenario.

Deterministic, ``$0``, no Docker / model / LLM (ADR-0010).
"""

from __future__ import annotations

from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
)
from graph_rag.models import (
    CanonicalEntity,
    DocumentRecord,
    EdgeProvenance,
    QueryRequest,
    Sentence,
    Subgraph,
    Triple,
)
from graph_rag.query.retriever import QueryRetriever


def _node(canonical_id: str, name: str = "n", type_: str = "PERSON") -> CanonicalEntity:
    """Build a bare graph node."""
    return CanonicalEntity(canonical_id=canonical_id, name=name, type=type_)


def _edge(subject_id: str, object_id: str, doc_id: str = "d") -> Triple:
    """Build an edge with minimal provenance."""
    return Triple(
        subject_id=subject_id,
        predicate="RELATED_TO",
        object_id=object_id,
        provenance=EdgeProvenance(source_doc_id=doc_id, sentence_index=0, source_sentence="s"),
    )


# --- hop_distance BFS -------------------------------------------------------


def test_hop_distances_single_seed_chain() -> None:
    """BFS over a chain: seed 0, one hop 1, two hops 2; isolated node omitted."""
    subgraph = Subgraph(
        nodes=[_node("a"), _node("b"), _node("c"), _node("d")],
        edges=[_edge("a", "b"), _edge("b", "c")],  # d has no edge -> unreachable
    )
    distances = QueryRetriever._hop_distances(subgraph, {"a": 0.9})
    assert distances == {"a": 0.0, "b": 1.0, "c": 2.0}
    assert "d" not in distances  # unreachable -> omitted (ranker treats as inf)


def test_hop_distances_is_undirected() -> None:
    """Edges are traversed undirected, so an incoming edge still reaches the seed's neighbour."""
    subgraph = Subgraph(
        nodes=[_node("a"), _node("b")],
        edges=[_edge("b", "a")],  # edge points b->a; seed is a
    )
    distances = QueryRetriever._hop_distances(subgraph, {"a": 0.5})
    assert distances == {"a": 0.0, "b": 1.0}


def test_hop_distances_multiple_seeds_takes_nearest() -> None:
    """With two seeds, each node gets its distance to the NEAREST seed."""
    subgraph = Subgraph(
        nodes=[_node("a"), _node("b"), _node("c")],
        edges=[_edge("a", "b"), _edge("b", "c")],
    )
    distances = QueryRetriever._hop_distances(subgraph, {"a": 0.9, "c": 0.8})
    assert distances == {"a": 0.0, "c": 0.0, "b": 1.0}


def test_hop_distances_seed_absent_from_subgraph_ignored() -> None:
    """A seed id with no node in the subgraph is not a BFS root."""
    subgraph = Subgraph(nodes=[_node("a")], edges=[])
    distances = QueryRetriever._hop_distances(subgraph, {"ghost": 1.0, "a": 0.7})
    assert distances == {"a": 0.0}


# --- seed-merge (B5) + top-node answer (crafted, seeded) --------------------


def _retriever(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    document_store: InMemoryDocumentStore,
    graph_store: InMemoryGraphStore,
) -> QueryRetriever:
    """Build a retriever over the given fakes with default (pinned) settings."""
    return QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    )


def test_seed_merge_combines_entity_and_sentence_seeds() -> None:
    """Entity seeds anchor the graph; sentence seeds anchor evidence — both returned.

    The entity kNN seed determines which nodes carry a seed similarity term (so the
    seeded node outranks an equally-close non-seed node), while the sentence kNN
    seed populates ``supporting_sentences`` independently — the B5 merge.
    """
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    question = "alpha"

    # Entity seed: only "alpha" is vector-seedable (its vector == the question's),
    # so it alone enters seed_scores. "beta" is a graph node but not a seed.
    entity_store.upsert(
        CanonicalEntity(
            canonical_id="alpha", name="alpha", type="PERSON", vector=embedder.embed([question])[0]
        )
    )
    graph_store.upsert_entities([_node("alpha", "alpha"), _node("beta", "beta")])
    graph_store.write_triples([_edge("alpha", "beta")])

    # Sentence seed: a separate document sentence anchors the supporting evidence.
    sentence = "alpha met beta at the workshop"
    document_store.upsert(
        DocumentRecord(
            document_id="d1",
            bucket="documents",
            object_key="d1.md",
            text=sentence,
            sentences=[Sentence(text=sentence, char_start=0, char_end=len(sentence), index=0)],
            sentence_vectors=[embedder.embed([sentence])[0]],
        )
    )

    response = _retriever(embedder, entity_store, document_store, graph_store).retrieve(
        QueryRequest(question=question)
    )

    # Graph side: both nodes present; the entity seed (alpha) carries the seed term
    # AND is hop 0, so it outranks the non-seed neighbour beta.
    ranked_by_id = {n.canonical_id: n for n in response.ranked_nodes}
    assert set(ranked_by_id) == {"alpha", "beta"}
    assert ranked_by_id["alpha"].score > ranked_by_id["beta"].score
    # Evidence side (merged, independent): the sentence seed rode along with provenance.
    assert response.supporting_sentences
    assert response.supporting_sentences[0].document_id == "d1"
    assert response.supporting_sentences[0].char_start == 0


def test_top_node_is_the_answer() -> None:
    """The predicted answer is the top-ranked node (no type filter in V6)."""
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    question = "grace hopper"

    entity_store.upsert(
        CanonicalEntity(
            canonical_id="gh",
            name="Grace Hopper",
            type="PERSON",
            vector=embedder.embed([question])[0],
        )
    )
    graph_store.upsert_entities([_node("gh", "Grace Hopper"), _node("navy", "US Navy", "ORG")])
    graph_store.write_triples([_edge("gh", "navy")])

    response = _retriever(embedder, entity_store, document_store, graph_store).retrieve(
        QueryRequest(question=question)
    )

    assert response.answer_entity is not None
    assert response.answer_entity.canonical_id == "gh"
    assert response.answer == "Grace Hopper"
    assert response.ranked_nodes[0].canonical_id == "gh"


def test_empty_stores_return_no_answer() -> None:
    """With nothing seeded, retrieval returns a null answer + empty subgraph (no error)."""
    embedder = FakeEmbedder()
    response = _retriever(
        embedder, InMemoryEntityStore(), InMemoryDocumentStore(), InMemoryGraphStore()
    ).retrieve(QueryRequest(question="anything"))

    assert response.answer is None
    assert response.answer_entity is None
    assert response.subgraph.nodes == []
    assert response.ranked_nodes == []
    assert response.supporting_sentences == []
