"""Neo4j adapter for the ``GraphStore`` port (V5, ADR-0006, ARCHITECTURE §5c).

Wraps the official ``neo4j`` Python driver (a light, pure-Python driver — no
torch) to persist the knowledge graph: canonical entities become multi-label
``:Entity:Type`` nodes keyed by ``canonical_id`` and per-document triples become
provenance-carrying relationships between them. V6 retrieval reads the connected
subgraph back via :meth:`khop`.

The real adapter is proved to behave identically to
:class:`~graph_rag.fakes.InMemoryGraphStore` at the seam by
``tests/contract/test_graph_store_contract.py``:

* :meth:`upsert_entities` is idempotent, keyed by ``canonical_id`` (``MERGE`` on
  the node, then overwrite ``name``/``type``/``aliases`` and (re)add the type
  label) — re-upserting overwrites, never duplicates;
* :meth:`write_triples` writes one relationship per triple, labelled by its
  ``predicate`` and carrying the full :class:`~graph_rag.models.EdgeProvenance`
  (plus the optional DATE qualifier) as edge properties;
* :meth:`delete_document_edges` removes only the edges of one ``source_doc_id``
  (nodes are shared and kept), so re-ingesting a document replaces its edges;
* :meth:`khop` returns the connected subgraph within ``hops`` of the seeds.

**Cypher-injection safety.** Neo4j cannot parametrize node labels, relationship
types or variable-length bounds — they must be literals in the query string. This
adapter never string-formats untrusted input: the node label is looked up from
the curated :data:`_TYPE_LABELS` table (validated against the closed
:data:`~graph_rag.models.CuratedType` set), the relationship type is validated
against :data:`~graph_rag.predicates.CLOSED_PREDICATES`, and the k-hop bound is
coerced to a non-negative ``int`` — all three come from closed, code-controlled
sets, never from document text.

**Edge idempotency.** :meth:`write_triples` uses ``CREATE`` (not ``MERGE``): the
KG-build checkpoint calls :meth:`delete_document_edges` for the document *before*
re-writing its triples, so the delete-then-create pair is the idempotency
mechanism (re-ingest → stable edge count) rather than per-edge MERGE. Nodes stay
idempotent via ``MERGE`` on ``canonical_id``.

**Node properties.** Only ``canonical_id``/``name``/``type``/``aliases`` are
stored on the node (ADR-0006). The entity ``vector`` lives in ``ES-Entities`` and
is intentionally NOT persisted here, so a :class:`~graph_rag.models.CanonicalEntity`
reconstructed from the graph has ``vector=None``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from neo4j import GraphDatabase

from graph_rag.logging import get_logger
from graph_rag.models import CanonicalEntity, EdgeProvenance, Subgraph, Triple
from graph_rag.predicates import CLOSED_PREDICATES

if TYPE_CHECKING:
    from neo4j import Driver

    from graph_rag.config import Settings

__all__ = ["Neo4jGraphStore"]

_logger = get_logger(__name__)

# CuratedType -> the node's second label (a shared ``:Entity`` label plus this
# per-type label, e.g. ``:Entity:Person``). The label is a code-controlled
# constant so string-formatting it into Cypher is safe; the entity ``type`` is
# validated against this table (which covers the whole closed CuratedType set)
# before use. ``DATE`` is included for completeness though dates are normally
# modeled as an edge qualifier, not a node (ADR-0006).
_TYPE_LABELS: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "LOCATION": "Location",
    "DATE": "Date",
    "EVENT": "Event",
    "NORP": "Norp",
    "PRODUCT": "Product",
}


class Neo4jGraphStore:
    """Neo4j-backed :class:`~graph_rag.ports.GraphStore` (V5-active, ADR-0006).

    Nodes are keyed by ``canonical_id`` (a uniqueness constraint enforces it, see
    :meth:`init`) so ``upsert_entities`` overwrites rather than duplicates. All
    reads/writes go through the injected driver's ``execute_query`` (auto-committed
    transactions); construction is cheap and does not open a connection until the
    first query.
    """

    def __init__(self, driver: Driver, database: str = "neo4j") -> None:
        """Build the store over an existing driver.

        Args:
            driver: A configured ``neo4j`` driver. ``GraphDatabase.driver(...)`` is
                lazy — the connection is opened on first use, not at construction.
            database: The target database name (Neo4j default is ``"neo4j"``).
        """
        self._driver = driver
        self._database = database

    @classmethod
    def from_settings(cls, settings: Settings) -> Neo4jGraphStore:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.neo4j_uri`` and the ``settings.neo4j_user`` /
        ``settings.neo4j_password`` credentials (a local-dev default, never a real
        secret — see :class:`~graph_rag.config.Settings`).
        """
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return cls(driver=driver)

    def close(self) -> None:
        """Close the underlying driver and its connection pool."""
        self._driver.close()

    def init(self) -> None:
        """Create the ``:Entity(canonical_id)`` uniqueness constraint if absent.

        Idempotent via ``IF NOT EXISTS`` — a no-op when the constraint already
        exists. The constraint also creates the backing index used by the
        ``MERGE``/``MATCH`` on ``canonical_id`` throughout this adapter.
        """
        self._run(
            "CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.canonical_id IS UNIQUE"
        )
        _logger.info("ensured :Entity(canonical_id) uniqueness constraint")

    # --- Writes -------------------------------------------------------------

    def upsert_entities(self, entities: list[CanonicalEntity]) -> None:
        """Create/merge multi-label ``:Entity:Type`` nodes, keyed by ``canonical_id``.

        Idempotent by ``canonical_id``: ``MERGE`` finds-or-creates the node, then
        ``SET`` overwrites ``name``/``type``/``aliases`` and (re)adds the type
        label — so re-upserting an entity updates it in place. Entities are grouped
        by type label so each group is one ``UNWIND`` batch (the label is a literal
        in the query, validated against :data:`_TYPE_LABELS`).
        """
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            label = _TYPE_LABELS.get(entity.type)
            if label is None:  # pragma: no cover - CuratedType keeps this unreachable
                raise ValueError(f"unknown entity type {entity.type!r} (not a CuratedType)")
            by_label[label].append(
                {
                    "canonical_id": entity.canonical_id,
                    "name": entity.name,
                    "type": entity.type,
                    "aliases": list(entity.aliases),
                }
            )
        for label, rows in by_label.items():
            self._run(
                "UNWIND $rows AS row "
                "MERGE (n:Entity {canonical_id: row.canonical_id}) "
                "SET n.name = row.name, n.type = row.type, n.aliases = row.aliases, "
                f"n:{label}",
                rows=rows,
            )
        _logger.debug("upserted %d entities across %d type label(s)", len(entities), len(by_label))

    def write_triples(self, triples: list[Triple]) -> None:
        """Write one provenance-carrying relationship per triple (via ``CREATE``).

        Relationships are grouped by ``predicate`` so each group is one ``UNWIND``
        batch (the relationship type is a literal in the query, validated against
        :data:`~graph_rag.predicates.CLOSED_PREDICATES`). Edge properties are set
        from :class:`~graph_rag.models.EdgeProvenance` plus the optional ``date``
        qualifier; ``None`` optionals are simply omitted (and read back as ``None``).
        Uses ``CREATE`` — idempotency is via :meth:`delete_document_edges` before a
        rewrite, not per-edge ``MERGE`` (see module docstring).

        Raises:
            ValueError: If a triple's ``predicate`` is not in the closed set.
        """
        by_predicate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for triple in triples:
            if triple.predicate not in CLOSED_PREDICATES:
                raise ValueError(
                    f"predicate {triple.predicate!r} is not in the closed predicate set; "
                    "map it via graph_rag.predicates.map_predicate before writing"
                )
            by_predicate[triple.predicate].append(
                {
                    "subject_id": triple.subject_id,
                    "object_id": triple.object_id,
                    "props": _edge_props(triple),
                }
            )
        for predicate, rows in by_predicate.items():
            self._run(
                "UNWIND $rows AS row "
                "MATCH (s:Entity {canonical_id: row.subject_id}) "
                "MATCH (o:Entity {canonical_id: row.object_id}) "
                f"CREATE (s)-[r:{predicate}]->(o) "
                "SET r += row.props",
                rows=rows,
            )
        _logger.debug(
            "wrote %d triples across %d predicate type(s)", len(triples), len(by_predicate)
        )

    def delete_document_edges(self, source_doc_id: str) -> None:
        """Remove every edge whose provenance ``source_doc_id`` matches (nodes kept).

        Called before re-writing a document's triples so re-ingest overwrites
        rather than duplicates (graph idempotency, TESTING.md gap #1). Nodes are
        shared across documents and left intact.
        """
        self._run(
            "MATCH ()-[r]->() WHERE r.source_doc_id = $doc DELETE r",
            doc=source_doc_id,
        )
        _logger.debug("deleted edges for source_doc_id=%s", source_doc_id)

    # --- Reads --------------------------------------------------------------

    def khop(self, seed_ids: list[str], hops: int) -> Subgraph:
        """Return the connected subgraph within ``hops`` of ``seed_ids`` (undirected).

        A variable-length undirected traversal ``(s)-[*0..hops]-(m)`` from every
        seed present in the graph collects the reachable node set (hop 0 = the
        seeds themselves), then the nodes and every edge whose endpoints are both
        in that set are returned. Seed IDs with no node are ignored. ``hops`` is
        coerced to a non-negative ``int`` and inlined as a literal bound (Neo4j
        cannot parametrize variable-length bounds).
        """
        depth = max(int(hops), 0)
        records = self._run(
            "MATCH (s:Entity) WHERE s.canonical_id IN $seeds "
            f"MATCH (s)-[*0..{depth}]-(m:Entity) "
            "RETURN collect(DISTINCT m.canonical_id) AS ids",
            seeds=list(seed_ids),
        )
        ids: list[str] = records[0]["ids"] if records else []
        nodes = self._nodes_by_ids(ids)
        edges = self._edges_within(ids)
        return Subgraph(nodes=nodes, edges=edges)

    def node_count(self) -> int:
        """Return the number of ``:Entity`` nodes currently stored."""
        records = self._run("MATCH (n:Entity) RETURN count(n) AS c")
        return int(records[0]["c"])

    def edge_count(self) -> int:
        """Return the number of edges currently stored."""
        records = self._run("MATCH ()-[r]->() RETURN count(r) AS c")
        return int(records[0]["c"])

    def get_node(self, canonical_id: str) -> CanonicalEntity | None:
        """Return the node for ``canonical_id``, or ``None`` if absent (test helper).

        The reconstructed entity carries ``vector=None`` — vectors live in
        ``ES-Entities``, not the graph (ADR-0006).
        """
        records = self._run(
            "MATCH (n:Entity {canonical_id: $cid}) "
            "RETURN n.canonical_id AS canonical_id, n.name AS name, "
            "n.type AS type, n.aliases AS aliases",
            cid=canonical_id,
        )
        if not records:
            return None
        return _to_entity(records[0])

    def get_node_edges(self, canonical_id: str) -> list[Triple]:
        """Return every edge incident to ``canonical_id`` (as subject or object)."""
        records = self._run(
            "MATCH (n:Entity {canonical_id: $cid})-[r]-() "
            "RETURN startNode(r).canonical_id AS subject_id, type(r) AS predicate, "
            "endNode(r).canonical_id AS object_id, properties(r) AS props",
            cid=canonical_id,
        )
        return [_to_triple(record) for record in records]

    # --- Internals ----------------------------------------------------------

    def _nodes_by_ids(self, ids: list[str]) -> list[CanonicalEntity]:
        """Return the :class:`CanonicalEntity` for each id in ``ids`` (missing skipped)."""
        if not ids:
            return []
        records = self._run(
            "MATCH (n:Entity) WHERE n.canonical_id IN $ids "
            "RETURN n.canonical_id AS canonical_id, n.name AS name, "
            "n.type AS type, n.aliases AS aliases",
            ids=ids,
        )
        return [_to_entity(record) for record in records]

    def _edges_within(self, ids: list[str]) -> list[Triple]:
        """Return every edge whose subject and object are both in ``ids``."""
        if not ids:
            return []
        records = self._run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE a.canonical_id IN $ids AND b.canonical_id IN $ids "
            "RETURN a.canonical_id AS subject_id, type(r) AS predicate, "
            "b.canonical_id AS object_id, properties(r) AS props",
            ids=ids,
        )
        return [_to_triple(record) for record in records]

    def _run(self, query: str, **params: Any) -> list[Any]:
        """Execute ``query`` with ``params`` in an auto-commit transaction; return records."""
        result = self._driver.execute_query(query, database_=self._database, **params)
        return list(result.records)


def _edge_props(triple: Triple) -> dict[str, Any]:
    """Build the edge property map for ``triple`` (omitting ``None`` optionals).

    Required provenance (``source_doc_id``/``sentence_index``/``source_sentence``)
    is always present; ``raw_predicate``/``confidence``/``char_start``/``char_end``
    and the ``date`` qualifier are stored only when set. Neo4j does not store
    ``null`` properties, so omitting them is equivalent to reading them back as
    ``None`` when the :class:`Triple` is reconstructed.
    """
    provenance = triple.provenance
    props: dict[str, Any] = {
        "source_doc_id": provenance.source_doc_id,
        "sentence_index": provenance.sentence_index,
        "source_sentence": provenance.source_sentence,
    }
    if provenance.raw_predicate is not None:
        props["raw_predicate"] = provenance.raw_predicate
    if provenance.confidence is not None:
        props["confidence"] = provenance.confidence
    if provenance.char_start is not None:
        props["char_start"] = provenance.char_start
    if provenance.char_end is not None:
        props["char_end"] = provenance.char_end
    if triple.date is not None:
        props["date"] = triple.date
    return props


def _to_entity(record: Any) -> CanonicalEntity:
    """Rebuild a :class:`CanonicalEntity` from a node row (``vector`` is not stored)."""
    return CanonicalEntity(
        canonical_id=record["canonical_id"],
        name=record["name"],
        type=record["type"],
        aliases=list(record["aliases"] or []),
        vector=None,
    )


def _to_triple(record: Any) -> Triple:
    """Rebuild a :class:`Triple` (with :class:`EdgeProvenance`) from an edge row."""
    props: dict[str, Any] = dict(record["props"])
    return Triple(
        subject_id=record["subject_id"],
        predicate=record["predicate"],
        object_id=record["object_id"],
        provenance=EdgeProvenance(
            source_doc_id=props["source_doc_id"],
            sentence_index=int(props["sentence_index"]),
            source_sentence=props["source_sentence"],
            raw_predicate=props.get("raw_predicate"),
            confidence=props.get("confidence"),
            char_start=(None if props.get("char_start") is None else int(props["char_start"])),
            char_end=(None if props.get("char_end") is None else int(props["char_end"])),
        ),
        date=props.get("date"),
    )
