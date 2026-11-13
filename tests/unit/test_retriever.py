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
from graph_rag.scope import DEFAULT_SCOPE


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
        DEFAULT_SCOPE,
        CanonicalEntity(
            canonical_id="alpha", name="alpha", type="PERSON", vector=embedder.embed([question])[0]
        ),
    )
    graph_store.upsert_entities(DEFAULT_SCOPE, [_node("alpha", "alpha"), _node("beta", "beta")])
    graph_store.write_triples(DEFAULT_SCOPE, [_edge("alpha", "beta")])

    # Sentence seed: a separate document sentence anchors the supporting evidence.
    sentence = "alpha met beta at the workshop"
    document_store.upsert(
        DEFAULT_SCOPE,
        DocumentRecord(
            document_id="d1",
            bucket="documents",
            object_key="d1.md",
            text=sentence,
            sentences=[Sentence(text=sentence, char_start=0, char_end=len(sentence), index=0)],
            sentence_vectors=[embedder.embed([sentence])[0]],
        ),
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


def test_answer_is_not_the_entity_the_question_names() -> None:
    """The question's own entity ranks first; the answer is the other end of the edge (GRA-161)."""
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    question = "grace hopper"

    entity_store.upsert(
        DEFAULT_SCOPE,
        CanonicalEntity(
            canonical_id="gh",
            name="Grace Hopper",
            type="PERSON",
            vector=embedder.embed([question])[0],
        ),
    )
    graph_store.upsert_entities(
        DEFAULT_SCOPE, [_node("gh", "Grace Hopper"), _node("navy", "US Navy", "ORG")]
    )
    graph_store.write_triples(DEFAULT_SCOPE, [_edge("gh", "navy")])

    response = _retriever(embedder, entity_store, document_store, graph_store).retrieve(
        QueryRequest(question=question)
    )

    assert response.ranked_nodes[0].canonical_id == "gh"
    assert response.answer_entity is not None
    assert response.answer_entity.canonical_id == "navy"
    assert response.answer == "US Navy"


def test_empty_stores_return_no_answer() -> None:
    """With nothing seeded, retrieval returns a null answer + empty subgraph (no error)."""
    embedder = FakeEmbedder()
    response = _retriever(
        embedder, InMemoryEntityStore(), InMemoryDocumentStore(), InMemoryGraphStore()
    ).retrieve(QueryRequest(question="anything"))

    assert response.answer is None
    assert response.answer_entity is None
    assert response.matched_entity_ids == []
    assert response.subgraph.nodes == []
    assert response.ranked_nodes == []
    assert response.supporting_sentences == []


def _seeded(names: list[str], question: str) -> QueryRetriever:
    """A retriever over entities ``names`` (vector = embedding of the name), all linked to a hub."""
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    graph_store = InMemoryGraphStore()
    for name in names:
        entity_store.upsert(
            DEFAULT_SCOPE,
            CanonicalEntity(
                canonical_id=name, name=name, type="PERSON", vector=embedder.embed([name])[0]
            ),
        )
    graph_store.upsert_entities(DEFAULT_SCOPE, [_node(n, n) for n in [*names, "hub"]])
    graph_store.write_triples(DEFAULT_SCOPE, [_edge(n, "hub") for n in names])
    return _retriever(embedder, entity_store, InMemoryDocumentStore(), graph_store)


def test_matched_entities_one_match() -> None:
    """A question matching one entity lists exactly that id, not the hub it links to."""
    response = _seeded(["alpha"], "alpha").retrieve(QueryRequest(question="alpha"))
    assert response.matched_entity_ids == ["alpha"]
    assert "hub" in {n.canonical_id for n in response.ranked_nodes}


def test_matched_entities_several_best_first() -> None:
    """Several matches come back best first, and the exact match leads."""
    response = _seeded(["beta", "alpha", "gamma"], "alpha").retrieve(QueryRequest(question="alpha"))
    assert response.matched_entity_ids[0] == "alpha"
    assert set(response.matched_entity_ids) == {"alpha", "beta", "gamma"}
    assert "hub" not in response.matched_entity_ids


def test_matched_entities_none() -> None:
    """A question that matches nothing lists no ids."""
    response = _seeded([], "alpha").retrieve(QueryRequest(question="alpha"))
    assert response.matched_entity_ids == []


# --- the answer is not the question's own entity (GRA-161) -------------------
#
# The run5 graph: 11 entities and 12 edges as a real stack built them from four documents.

_RUN5_ENTITIES = {
    "klaus": ("Klaus Weber", "PERSON"),
    "hamburg": ("Hamburg", "LOCATION"),
    "y2021": ("2021", "DATE"),
    "northwind": ("Northwind Shipping", "ORG"),
    "acme": ("Acme Logistics", "ORG"),
    "delft": ("Delft University", "ORG"),
    "pieter": ("Pieter de Vries", "PERSON"),
    "y2015": ("2015", "DATE"),
    "rotterdam": ("Rotterdam", "LOCATION"),
    "helix": ("Helix Robotics", "ORG"),
    "maria": ("Maria Santos", "PERSON"),
}
_S_MARIA = "Maria Santos is the chief engineer at Helix Robotics, a company based in Rotterdam."
_S_PROTO = "Maria Santos presented the prototype in Hamburg."
_S_NORTH = "Northwind Shipping operates out of Hamburg, and its CEO Klaus Weber moved to Acme Logistics after the deal."  # noqa: E501
_S_PIETER = "Pieter de Vries studied at Delft University and later joined Helix Robotics."
_S_FOUND = "She founded the company in 2015 with Pieter de Vries."
_S_SIGNED = "Helix Robotics signed a contract with Acme Logistics to build warehouse robots."
_S_BOUGHT = "Acme Logistics acquired Northwind Shipping in 2021."
_RUN5_EDGES = [
    ("helix", "LOCATED_IN", "rotterdam", _S_MARIA),
    ("maria", "LOCATED_IN", "hamburg", _S_PROTO),
    ("northwind", "LOCATED_IN", "hamburg", _S_NORTH),
    ("pieter", "WORKS_FOR", "helix", _S_PIETER),
    ("pieter", "FOUNDED", "helix", _S_FOUND),
    ("maria", "FOUNDED", "helix", _S_FOUND),
    ("maria", "WORKS_FOR", "helix", _S_MARIA),
    ("pieter", "AFFILIATED_WITH", "delft", _S_PIETER),
    ("helix", "AFFILIATED_WITH", "acme", _S_SIGNED),
    ("klaus", "WORKS_FOR", "acme", _S_NORTH),
    ("klaus", "WORKS_FOR", "northwind", _S_NORTH),
    ("acme", "OWNS", "northwind", _S_BOUGHT),
]  # fmt: skip


def _run5(question: str, seeds: list[str], best_sentence: str) -> QueryRetriever:
    """The run5 graph; ``seeds`` sit nearest the question, ``best_sentence`` is the top evidence."""
    embedder = FakeEmbedder()
    question_vector = embedder.embed([question])[0]
    entity_store = InMemoryEntityStore()
    graph_store = InMemoryGraphStore()
    document_store = InMemoryDocumentStore()
    for cid, (name, type_) in _RUN5_ENTITIES.items():
        vector = question_vector if cid in seeds else embedder.embed([f"unrelated {name}"])[0]
        entity_store.upsert(
            DEFAULT_SCOPE,
            CanonicalEntity(canonical_id=cid, name=name, type=type_, vector=vector),  # type: ignore[arg-type]
        )
    graph_store.upsert_entities(
        DEFAULT_SCOPE,
        [_node(cid, name, type_) for cid, (name, type_) in _RUN5_ENTITIES.items()],
    )
    graph_store.write_triples(
        DEFAULT_SCOPE,
        [
            Triple(
                subject_id=s_,
                predicate=p_,
                object_id=o_,
                provenance=EdgeProvenance(source_doc_id="d", sentence_index=0, source_sentence=t_),
            )
            for s_, p_, o_, t_ in _RUN5_EDGES
        ],
    )
    sentences = sorted({t for *_, t in _RUN5_EDGES})
    document_store.upsert(
        DEFAULT_SCOPE,
        DocumentRecord(
            document_id="d",
            bucket="documents",
            object_key="d.md",
            text=" ".join(sentences),
            sentences=[
                Sentence(text=t, char_start=0, char_end=len(t), index=i)
                for i, t in enumerate(sentences)
            ],
            sentence_vectors=[
                question_vector if t == best_sentence else embedder.embed([t])[0] for t in sentences
            ],
        ),
    )
    return _retriever(embedder, entity_store, document_store, graph_store)


def test_who_founded_answers_a_founder_not_the_company() -> None:
    """'Who founded Helix Robotics?' answers a founder; the company the question names is not it."""
    question = "Who founded Helix Robotics?"
    response = _run5(
        question,
        ["helix", "delft"],
        _S_PIETER,
    ).retrieve(QueryRequest(question=question))
    assert response.matched_entity_ids and "helix" in response.matched_entity_ids
    assert response.answer in {"Maria Santos", "Pieter de Vries"}


def test_where_and_who_answers_the_place_not_the_company() -> None:
    """'Where is Northwind Shipping based and who is its CEO?' answers Hamburg (the first ask)."""
    question = "Where is Northwind Shipping based and who is its CEO?"
    response = _run5(
        question,
        ["northwind", "acme", "klaus", "hamburg"],
        _S_BOUGHT,
    ).retrieve(QueryRequest(question=question))
    assert response.answer == "Hamburg"


def test_which_company_answers_the_other_party_to_the_contract() -> None:
    """'Which company signed a contract with Acme ...' answers Helix, not Acme or its subsidiary."""
    question = (
        "Which company signed a contract with Acme Logistics, and who presented the prototype?"
    )
    response = _run5(
        question,
        ["acme", "northwind", "klaus", "hamburg"],
        _S_SIGNED,
    ).retrieve(QueryRequest(question=question))
    assert response.answer == "Helix Robotics"


def test_question_naming_nothing_keeps_the_top_node() -> None:
    """With no entity named in the question the answer is still the top-ranked node."""
    question = "robotics company"
    response = _run5(question, ["helix"], "").retrieve(QueryRequest(question=question))
    assert response.answer == "Helix Robotics"
