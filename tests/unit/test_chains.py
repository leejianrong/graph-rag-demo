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


# --- hubs --------------------------------------------------------------------------------------


def hub_subgraph(fillers: int = 60) -> Subgraph:
    """The widget is a hub: many irrelevant edges, and the two that answer the question."""
    noise = [edge("widget", f"n{i}", f"Filler sentence {i}.") for i in range(fillers)]
    return Subgraph(nodes=[], edges=[*noise, MAKES, OWNS])


class Counting:
    """An embedder that remembers how many texts it was asked for, in each batch."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return embed(texts)


def test_a_hub_is_pruned_to_its_closest_edges_and_the_chain_is_still_found() -> None:
    counted = Counting()

    ordered = chain_first(
        hub_subgraph(), ["widget"], embed([QUESTION])[0], counted, beam=4, fanout=3
    )

    assert sentences(ordered)[:2] == [
        MAKES.provenance.source_sentence,
        OWNS.provenance.source_sentence,
    ]
    assert max(counted.batches[1:]) <= 4 * 3  # chains per level are at most beam x fanout


def test_without_a_cap_every_edge_at_a_hub_is_a_candidate() -> None:
    """Revert proof: with no cap the first level embeds a chain for every edge of the hub."""
    counted = Counting()

    chain_first(hub_subgraph(), ["widget"], embed([QUESTION])[0], counted, fanout=10_000)

    assert counted.batches[0] == 61  # 60 fillers and the maker edge: all touch the hub


def test_a_sentence_is_embedded_once_across_questions() -> None:
    cache: dict[str, list[float]] = {}
    first, second = Counting(), Counting()

    chain_first(
        hub_subgraph(), ["widget"], embed([QUESTION])[0], first, fanout=3, sentence_vectors=cache
    )
    chain_first(
        hub_subgraph(), ["widget"], embed([QUESTION])[0], second, fanout=3, sentence_vectors=cache
    )

    assert sum(first.batches) > sum(second.batches)  # the second question reuses the 61 sentences
    assert len(cache) == 61


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
