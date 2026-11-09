"""Merging graph nodes (GRA-30): the edges that move, and the ones that must not."""

from __future__ import annotations

from graph_rag.fakes import InMemoryGraphStore
from graph_rag.merging import rewire
from graph_rag.models import CanonicalEntity, EdgeProvenance, Triple
from graph_rag.scope import DEFAULT_SCOPE


def edge(
    s: str, o: str, doc: str = "d1", sentence: int = 0, predicate: str = "WORKS_FOR"
) -> Triple:
    return Triple(
        subject_id=s,
        predicate=predicate,
        object_id=o,
        provenance=EdgeProvenance(source_doc_id=doc, sentence_index=sentence, source_sentence="S."),
        valid_from="2020-01-01",
    )


def test_an_edge_moves_onto_the_survivor_and_keeps_everything_else() -> None:
    [moved] = rewire([edge("dup", "org")], "keep", {"dup"}, present=[])
    assert (moved.subject_id, moved.object_id) == ("keep", "org")
    assert moved.provenance.source_doc_id == "d1" and moved.valid_from == "2020-01-01"


def test_an_edge_between_the_merged_nodes_is_dropped_rather_than_left_as_a_loop() -> None:
    assert rewire([edge("keep", "dup"), edge("dup", "keep")], "keep", {"dup"}, present=[]) == []


def test_an_edge_the_survivor_already_has_is_not_repeated() -> None:
    present = [edge("keep", "org")]
    assert rewire([edge("dup", "org")], "keep", {"dup"}, present) == []
    # a different sentence of the same document is a different edge
    assert len(rewire([edge("dup", "org", sentence=1)], "keep", {"dup"}, present)) == 1


def test_the_same_edge_seen_from_both_ends_is_moved_once() -> None:
    twice = [edge("dup1", "org"), edge("dup1", "org")]
    assert len(rewire(twice, "keep", {"dup1", "dup2"}, present=[])) == 1


def test_the_store_folds_nodes_and_keeps_unrelated_edges_in_place() -> None:
    store = InMemoryGraphStore()
    nodes = [CanonicalEntity(canonical_id=i, name=i, type="PERSON") for i in ("keep", "dup", "x")]
    store.upsert_entities(DEFAULT_SCOPE, nodes)
    store.write_triples(DEFAULT_SCOPE, [edge("dup", "x"), edge("x", "keep", "d2")])

    store.merge_nodes(DEFAULT_SCOPE, "keep", ["dup"])

    assert store.get_node(DEFAULT_SCOPE, "dup") is None and store.node_count(DEFAULT_SCOPE) == 2
    assert {(e.subject_id, e.object_id) for e in store.get_node_edges(DEFAULT_SCOPE, "x")} == {
        ("keep", "x"),
        ("x", "keep"),
    }
