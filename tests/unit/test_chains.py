"""Path retrieval lists the best chain of edges first, and is off unless asked for (GRA-170)."""

from __future__ import annotations

import math

from graph_rag.config import Settings
from graph_rag.fakes import InMemoryDocumentStore, InMemoryEntityStore, InMemoryGraphStore
from graph_rag.models import CanonicalEntity, EdgeProvenance, QueryRequest, Subgraph, Triple
from graph_rag.query.chains import chain_first
from graph_rag.query.retriever import QueryRetriever
from graph_rag.scope import DEFAULT_SCOPE

VOCAB = "who owns makes widget acme zeta paris sells river the of company".split()


def embed(texts: list[str]) -> list[list[float]]:
    """A bag-of-words embedder over a tiny vocabulary, unit length, so similarity is readable."""
    out = []
    for text in texts:
        words = text.lower().replace("?", "").replace(".", "").split()
        vector = [float(words.count(w)) for w in VOCAB]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        out.append([x / norm for x in vector])
    return out


def edge(subject: str, object_: str, sentence: str) -> Triple:
    return Triple(
        subject_id=subject,
        predicate="RELATED_TO",
        object_id=object_,
        provenance=EdgeProvenance(source_doc_id="d", sentence_index=0, source_sentence=sentence),
    )


MAKES = edge("widget", "acme", "Acme makes the widget.")
OWNS = edge("acme", "zeta", "Zeta owns the company Acme.")
SELLS = edge("widget", "paris", "The widget sells in Paris.")
RIVER = edge("paris", "river", "The river of Paris.")
QUESTION = "Who owns the company that makes the widget?"


def subgraph() -> Subgraph:
    # A distractor first, so the original order is not already the right one.
    return Subgraph(nodes=[], edges=[SELLS, RIVER, MAKES, OWNS])


def sentences(edges: list[Triple]) -> list[str]:
    return [e.provenance.source_sentence for e in edges]


def test_the_chain_that_answers_the_question_comes_first() -> None:
    """Owner-of-maker is two edges; Zeta's sentence does not touch the seed, but the chain does."""
    ordered = chain_first(subgraph(), ["widget"], embed([QUESTION])[0], embed)

    assert sentences(ordered)[:2] == [
        MAKES.provenance.source_sentence,
        OWNS.provenance.source_sentence,
    ]
    assert sorted(sentences(ordered)) == sorted(
        sentences(subgraph().edges)
    )  # nothing lost or added


def test_a_chain_of_one_edge_cannot_reach_the_second_hop() -> None:
    """Revert proof: with depth 1 each edge stands alone and the owner edge is not on any chain."""
    ordered = chain_first(subgraph(), ["widget"], embed([QUESTION])[0], embed, depth=1)

    assert sentences(ordered)[0] == MAKES.provenance.source_sentence
    assert sentences(ordered).index(OWNS.provenance.source_sentence) > 1


def test_no_seed_in_the_subgraph_leaves_the_order_alone() -> None:
    ordered = chain_first(subgraph(), ["nowhere"], embed([QUESTION])[0], embed)

    assert ordered == subgraph().edges


# --- the flag ----------------------------------------------------------------------------------


def test_path_retrieval_is_off_by_default() -> None:
    assert Settings().path_retrieval is False


def retrieve(path_retrieval: bool) -> list[Triple]:
    entity_store, document_store, graph_store = (
        InMemoryEntityStore(),
        InMemoryDocumentStore(),
        InMemoryGraphStore(),
    )
    entity_store.upsert(
        DEFAULT_SCOPE,
        CanonicalEntity(
            canonical_id="widget", name="widget", type="PRODUCT", vector=embed(["widget"])[0]
        ),
    )
    graph_store.upsert_entities(
        DEFAULT_SCOPE,
        [
            CanonicalEntity(canonical_id=i, name=i, type="PRODUCT")
            for i in ("widget", "acme", "zeta", "paris", "river")
        ],
    )
    graph_store.write_triples(DEFAULT_SCOPE, subgraph().edges)

    class Embedder:
        dim = len(VOCAB)

        def embed(self, texts: list[str]) -> list[list[float]]:
            return embed(texts)

    retriever = QueryRetriever.from_settings(
        Settings(path_retrieval=path_retrieval, seed_top_k_entities=1),
        embedder=Embedder(),  # type: ignore[arg-type]
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
    )
    return retriever.retrieve(QueryRequest(question=QUESTION)).subgraph.edges


def test_the_retriever_lists_the_chain_first_only_when_the_flag_is_on() -> None:
    off = sentences(retrieve(False))
    on = sentences(retrieve(True))

    assert sorted(off) == sorted(on)  # same evidence either way
    assert on[:2] == [MAKES.provenance.source_sentence, OWNS.provenance.source_sentence]
    assert off[:2] != on[:2]
