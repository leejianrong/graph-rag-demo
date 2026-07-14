"""Unit tests for the V5 graph fake (fast, offline, ``$0``).

Exercises :class:`~graph_rag.fakes.InMemoryGraphStore` at the port seam — the
backbone the KG-build fast E2E runs against (ADR-0010): idempotent multi-label
node upsert, provenance edge write/read, doc-scoped edge delete (re-ingest
overwrite, TESTING.md gap #1) and BFS k-hop traversal. No Neo4j, no Docker.
"""

from __future__ import annotations

from graph_rag.fakes import InMemoryGraphStore
from graph_rag.models import CanonicalEntity, EdgeProvenance, Triple


def _entity(canonical_id: str, name: str, type_: str = "ORG") -> CanonicalEntity:
    """A minimal canonical entity (node) for the tests."""
    return CanonicalEntity(canonical_id=canonical_id, name=name, type=type_)  # type: ignore[arg-type]


def _triple(subject_id: str, object_id: str, doc_id: str = "doc-1") -> Triple:
    """A minimal provenance-carrying edge for the tests."""
    return Triple(
        subject_id=subject_id,
        predicate="RELATED_TO",
        object_id=object_id,
        provenance=EdgeProvenance(
            source_doc_id=doc_id,
            sentence_index=0,
            source_sentence=f"{subject_id} and {object_id}.",
        ),
    )


# --- node upsert ------------------------------------------------------------


def test_upsert_entities_is_idempotent_by_canonical_id() -> None:
    """Re-upserting the same ``canonical_id`` overwrites, never duplicates."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity("e-1", "Apple")])
    store.upsert_entities([_entity("e-1", "Apple Inc.")])  # same id, new name
    assert store.node_count() == 1
    node = store.get_node("e-1")
    assert node is not None
    assert node.name == "Apple Inc."


def test_upsert_preserves_type_the_second_label() -> None:
    """The node keeps its ``type`` (the multi-label ``:Entity:Type`` second label)."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity("e-p", "Ada", type_="PERSON")])
    node = store.get_node("e-p")
    assert node is not None
    assert node.type == "PERSON"


def test_get_node_missing_returns_none() -> None:
    """An unknown canonical id yields ``None``."""
    assert InMemoryGraphStore().get_node("nope") is None


# --- edge write / read ------------------------------------------------------


def test_write_and_read_edges_carry_provenance() -> None:
    """Written edges are retrievable and keep their provenance."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity("a", "A"), _entity("b", "B")])
    store.write_triples([_triple("a", "b")])
    assert store.edge_count() == 1
    edges = store.get_node_edges("a")
    assert len(edges) == 1
    assert edges[0].provenance.source_doc_id == "doc-1"
    # incident lookup finds the edge from either endpoint
    assert store.get_node_edges("b") == edges


# --- doc-scoped delete (re-ingest overwrite) --------------------------------


def test_delete_document_edges_removes_only_that_docs_edges() -> None:
    """Delete-by-doc removes exactly the target doc's edges; others survive."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity("a", "A"), _entity("b", "B"), _entity("c", "C")])
    store.write_triples(
        [
            _triple("a", "b", doc_id="doc-1"),
            _triple("b", "c", doc_id="doc-2"),
        ]
    )
    store.delete_document_edges("doc-1")
    assert store.edge_count() == 1
    remaining = store._edges  # type: ignore[attr-defined]
    assert remaining[0].provenance.source_doc_id == "doc-2"
    # nodes are untouched by an edge delete
    assert store.node_count() == 3


def test_reingest_overwrite_does_not_duplicate_edges() -> None:
    """delete-then-rewrite a doc's edges keeps the edge count stable (idempotent)."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity("a", "A"), _entity("b", "B")])
    store.write_triples([_triple("a", "b", doc_id="doc-1")])
    # re-ingest doc-1: checkpoint deletes then re-writes
    store.delete_document_edges("doc-1")
    store.write_triples([_triple("a", "b", doc_id="doc-1")])
    assert store.edge_count() == 1


# --- k-hop BFS --------------------------------------------------------------


def _line_graph() -> InMemoryGraphStore:
    """Build a-b-c-d plus e-a: BFS fan-out from ``a`` is e/b (h1), c (h2), d (h3)."""
    store = InMemoryGraphStore()
    store.upsert_entities([_entity(x, x.upper()) for x in ("a", "b", "c", "d", "e")])
    store.write_triples(
        [
            _triple("a", "b"),
            _triple("b", "c"),
            _triple("c", "d"),
            _triple("e", "a"),
        ]
    )
    return store


def test_khop_depth_1_returns_immediate_neighbours() -> None:
    """Depth 1 from ``a`` reaches a, b, e and the edges among them only."""
    sub = _line_graph().khop(["a"], hops=1)
    assert {n.canonical_id for n in sub.nodes} == {"a", "b", "e"}
    edge_pairs = {(e.subject_id, e.object_id) for e in sub.edges}
    assert edge_pairs == {("a", "b"), ("e", "a")}


def test_khop_depth_2_expands_one_further() -> None:
    """Depth 2 from ``a`` additionally reaches c (via b), not d."""
    sub = _line_graph().khop(["a"], hops=2)
    assert {n.canonical_id for n in sub.nodes} == {"a", "b", "c", "e"}
    edge_pairs = {(e.subject_id, e.object_id) for e in sub.edges}
    assert edge_pairs == {("a", "b"), ("b", "c"), ("e", "a")}


def test_khop_depth_0_is_seed_nodes_only() -> None:
    """Depth 0 returns just the seed nodes (no expansion), and no edges out."""
    sub = _line_graph().khop(["a"], hops=0)
    assert {n.canonical_id for n in sub.nodes} == {"a"}
    assert sub.edges == []


def test_khop_ignores_unknown_seed_ids() -> None:
    """Seed ids with no node are dropped; a fully-unknown seed set is empty."""
    store = _line_graph()
    sub = store.khop(["ghost"], hops=2)
    assert sub.nodes == []
    assert sub.edges == []


def test_khop_multiple_seeds_union() -> None:
    """Multiple seeds expand independently and the subgraph is their union."""
    sub = _line_graph().khop(["a", "d"], hops=1)
    assert {n.canonical_id for n in sub.nodes} == {"a", "b", "e", "d", "c"}


def test_counts_reflect_stored_nodes_and_edges() -> None:
    """``node_count`` / ``edge_count`` report what was written."""
    store = _line_graph()
    assert store.node_count() == 5
    assert store.edge_count() == 4
