"""Contract test: real ``Neo4jGraphStore`` behaves like ``InMemoryGraphStore``.

Per TESTING §3, the contract layer proves each real adapter behaves like its fake
against a real service (here Neo4j via testcontainers). It gates the adapter, not
pipeline logic, so it is marked ``contract`` and excluded from the fast suite.
Skips cleanly when Docker is unavailable.

Asserts the ``GraphStore`` contract (ADR-0006, ARCHITECTURE §5c) on the real
adapter, mirrored against :class:`~graph_rag.fakes.InMemoryGraphStore`:

* ``upsert_entities`` writes idempotent multi-label ``:Entity:Type`` nodes with
  the right properties (queried via ``:Entity`` and via the per-type label);
* ``write_triples`` writes edges carrying the full ``EdgeProvenance`` properties;
* an open-fallback ``RELATED_TO`` edge preserves the original phrase in
  ``raw_predicate``;
* a dated fact stores the date as an edge property (there is NO ``:Date`` node);
* ``delete_document_edges`` removes only that document's edges (nodes kept), so a
  delete-then-rewrite keeps ``edge_count`` stable (graph idempotency);
* ``khop`` returns the correct connected subgraph at depth 1 and depth 2.

Graph nodes carry no ``vector`` (vectors live in ``ES-Entities``, ADR-0006), so
the entities here are built with ``vector=None`` — the property the real adapter
persists and the fake's stored object then agree at the seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graph_rag.fakes import InMemoryGraphStore
from graph_rag.models import CanonicalEntity, EdgeProvenance, Triple

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neo4j import Driver

    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore

pytestmark = pytest.mark.contract

_NEO4J_IMAGE = "neo4j:5.20"


@pytest.fixture(scope="module")
def driver() -> Iterator[Driver]:
    """A real Neo4j driver over a throwaway container (module-scoped).

    Skips the whole module if Docker / testcontainers is unavailable.
    """
    try:
        from testcontainers.neo4j import Neo4jContainer

        import graph_rag.adapters.neo4j_graph_store  # noqa: F401 - import hygiene guard
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"testcontainers/neo4j not importable: {exc}")

    try:
        container = Neo4jContainer(_NEO4J_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001 - Docker not available / cannot pull image.
        pytest.skip(f"Docker/Neo4j container unavailable: {exc}")

    try:
        neo4j_driver = container.get_driver()
        yield neo4j_driver
        neo4j_driver.close()
    finally:
        container.stop()


@pytest.fixture()
def store(driver: Driver) -> Neo4jGraphStore:
    """A fresh :class:`Neo4jGraphStore` over an empty graph for each test."""
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore

    # Wipe every node + edge so each test starts from an empty graph.
    driver.execute_query("MATCH (n) DETACH DELETE n")
    graph_store = Neo4jGraphStore(driver=driver)
    graph_store.init()
    return graph_store


def _entity(canonical_id: str, name: str, type_: str = "PERSON") -> CanonicalEntity:
    """Build a vector-less ``CanonicalEntity`` (graph nodes carry no vector)."""
    return CanonicalEntity(
        canonical_id=canonical_id,
        name=name,
        type=type_,  # type: ignore[arg-type]
        aliases=[],
        vector=None,
    )


def _triple(
    subject_id: str,
    predicate: str,
    object_id: str,
    *,
    doc: str = "doc:1",
    sentence_index: int = 0,
    source_sentence: str = "A sentence.",
    raw_predicate: str | None = None,
    confidence: float | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    date: str | None = None,
) -> Triple:
    """Build a ``Triple`` with fully-specified provenance for the contract asserts."""
    return Triple(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        provenance=EdgeProvenance(
            source_doc_id=doc,
            sentence_index=sentence_index,
            source_sentence=source_sentence,
            raw_predicate=raw_predicate,
            confidence=confidence,
            char_start=char_start,
            char_end=char_end,
        ),
        date=date,
    )


def test_upsert_entities_writes_multi_label_nodes(store: Neo4jGraphStore, driver: Driver) -> None:
    """Upsert writes ``:Entity:Type`` nodes with the right props; re-upsert overwrites."""
    alice = _entity("p:alice", "Alice", type_="PERSON")
    acme = CanonicalEntity(
        canonical_id="o:acme",
        name="Acme Corp",
        type="ORG",
        aliases=["Acme", "Acme Inc"],
        vector=None,
    )
    store.upsert_entities([alice, acme])

    fake = InMemoryGraphStore()
    fake.upsert_entities([alice, acme])

    # Node round-trips like the fake (vector is None on both — graph stores none).
    assert store.get_node("p:alice") == alice == fake.get_node("p:alice")
    assert store.get_node("o:acme") == acme == fake.get_node("o:acme")
    assert store.get_node("p:missing") is None
    assert store.node_count() == fake.node_count() == 2

    # Multi-label: the shared :Entity label plus the per-type label.
    person_labels = driver.execute_query(
        "MATCH (n:Entity {canonical_id: 'p:alice'}) RETURN labels(n) AS labels"
    ).records[0]["labels"]
    assert set(person_labels) == {"Entity", "Person"}

    # Queryable by the per-type label.
    orgs = driver.execute_query("MATCH (n:Organization) RETURN n.canonical_id AS cid").records
    assert [r["cid"] for r in orgs] == ["o:acme"]

    # Idempotent: re-upsert with new props overwrites in place (still 2 nodes).
    store.upsert_entities([CanonicalEntity(canonical_id="p:alice", name="Alice B.", type="PERSON")])
    assert store.node_count() == 2
    refreshed = store.get_node("p:alice")
    assert refreshed is not None and refreshed.name == "Alice B."


def test_write_triples_carries_full_provenance(store: Neo4jGraphStore) -> None:
    """A written edge carries every provenance property back through ``get_node_edges``."""
    store.upsert_entities([_entity("p:alice", "Alice"), _entity("o:acme", "Acme", type_="ORG")])
    triple = _triple(
        "p:alice",
        "WORKS_FOR",
        "o:acme",
        doc="doc:1",
        sentence_index=3,
        source_sentence="Alice works for Acme.",
        confidence=0.91,
        char_start=10,
        char_end=42,
    )
    store.write_triples([triple])

    fake = InMemoryGraphStore()
    fake.upsert_entities([_entity("p:alice", "Alice"), _entity("o:acme", "Acme", type_="ORG")])
    fake.write_triples([triple])

    assert store.edge_count() == fake.edge_count() == 1
    edges = store.get_node_edges("p:alice")
    assert len(edges) == 1
    edge = edges[0]
    assert edge == fake.get_node_edges("p:alice")[0]
    assert edge.predicate == "WORKS_FOR"
    assert edge.subject_id == "p:alice"
    assert edge.object_id == "o:acme"
    assert edge.provenance.source_doc_id == "doc:1"
    assert edge.provenance.sentence_index == 3
    assert edge.provenance.source_sentence == "Alice works for Acme."
    assert edge.provenance.confidence == pytest.approx(0.91)
    assert edge.provenance.char_start == 10
    assert edge.provenance.char_end == 42
    assert edge.provenance.raw_predicate is None
    assert edge.date is None


def test_related_to_fallback_preserves_raw_predicate(store: Neo4jGraphStore) -> None:
    """An open-fallback ``RELATED_TO`` edge keeps the original phrase in ``raw_predicate``."""
    store.upsert_entities([_entity("p:alice", "Alice"), _entity("p:bob", "Bob")])
    triple = _triple(
        "p:alice",
        "RELATED_TO",
        "p:bob",
        raw_predicate="was spotted near",
    )
    store.write_triples([triple])

    edges = store.get_node_edges("p:alice")
    assert len(edges) == 1
    assert edges[0].predicate == "RELATED_TO"
    assert edges[0].provenance.raw_predicate == "was spotted near"


def test_dated_fact_is_an_edge_property_not_a_node(store: Neo4jGraphStore, driver: Driver) -> None:
    """A DATE qualifier is stored on the edge; no standalone date node is created."""
    store.upsert_entities(
        [_entity("o:acme", "Acme", type_="ORG"), _entity("l:paris", "Paris", type_="LOCATION")]
    )
    triple = _triple("o:acme", "FOUNDED", "l:paris", date="1998-09-04")
    store.write_triples([triple])

    edge = store.get_node_edges("o:acme")[0]
    assert edge.date == "1998-09-04"

    # No :Date node exists, and the node count is exactly the two entities.
    date_nodes = driver.execute_query("MATCH (n:Date) RETURN count(n) AS c").records[0]["c"]
    assert date_nodes == 0
    assert store.node_count() == 2


def test_delete_document_edges_is_doc_scoped_and_idempotent(store: Neo4jGraphStore) -> None:
    """Delete removes only one doc's edges (nodes kept); rewrite keeps edge_count stable."""
    store.upsert_entities([_entity("p:a", "A"), _entity("p:b", "B"), _entity("p:c", "C")])
    doc1 = [
        _triple("p:a", "WORKS_FOR", "p:b", doc="doc:1"),
        _triple("p:b", "MEMBER_OF", "p:c", doc="doc:1"),
    ]
    doc2 = [_triple("p:a", "AFFILIATED_WITH", "p:c", doc="doc:2")]
    store.write_triples(doc1)
    store.write_triples(doc2)
    assert store.edge_count() == 3

    # Delete only doc:1's edges — doc:2's edge and all nodes remain.
    store.delete_document_edges("doc:1")
    assert store.edge_count() == 1
    assert store.node_count() == 3
    remaining = store.get_node_edges("p:a")
    assert [e.provenance.source_doc_id for e in remaining] == ["doc:2"]

    # Re-ingest doc:1 (delete-then-rewrite): edge_count returns to 3, not 5.
    store.delete_document_edges("doc:1")
    store.write_triples(doc1)
    assert store.edge_count() == 3


def test_khop_returns_connected_subgraph(store: Neo4jGraphStore) -> None:
    """k-hop expansion matches the fake at depth 1 and depth 2 (undirected BFS)."""
    # Chain: a -> b -> c -> d, plus an isolated node z.
    for cid in ("a", "b", "c", "d", "z"):
        store.upsert_entities([_entity(f"p:{cid}", cid.upper())])
    triples = [
        _triple("p:a", "WORKS_FOR", "p:b"),
        _triple("p:b", "WORKS_FOR", "p:c"),
        _triple("p:c", "WORKS_FOR", "p:d"),
    ]
    store.write_triples(triples)

    fake = InMemoryGraphStore()
    for cid in ("a", "b", "c", "d", "z"):
        fake.upsert_entities([_entity(f"p:{cid}", cid.upper())])
    fake.write_triples(triples)

    def _node_ids(sub: object) -> set[str]:
        return {n.canonical_id for n in sub.nodes}  # type: ignore[attr-defined]

    def _edge_keys(sub: object) -> set[tuple[str, str, str]]:
        return {(e.subject_id, e.predicate, e.object_id) for e in sub.edges}  # type: ignore[attr-defined]

    # Depth 1 from b: {a, b, c} (undirected: reaches a backwards and c forwards).
    real1 = store.khop(["p:b"], 1)
    fake1 = fake.khop(["p:b"], 1)
    assert _node_ids(real1) == _node_ids(fake1) == {"p:a", "p:b", "p:c"}
    assert _edge_keys(real1) == _edge_keys(fake1)

    # Depth 2 from a: {a, b, c} (a..c within two hops, d is three hops away).
    real2 = store.khop(["p:a"], 2)
    fake2 = fake.khop(["p:a"], 2)
    assert _node_ids(real2) == _node_ids(fake2) == {"p:a", "p:b", "p:c"}
    assert _edge_keys(real2) == _edge_keys(fake2)

    # Hop 0 = just the seed; an unknown seed is ignored.
    real0 = store.khop(["p:a", "p:missing"], 0)
    assert _node_ids(real0) == {"p:a"}
    assert real0.edges == []
