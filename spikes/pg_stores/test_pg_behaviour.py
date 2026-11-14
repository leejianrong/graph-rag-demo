# ruff: noqa: E501
"""Beyond the reused contract tests: scope isolation (the shared checks) and a differential
run of random graph operations against the in-memory fake."""

from __future__ import annotations

import random
from collections.abc import Iterator

import psycopg
import pytest

from graph_rag.fakes import InMemoryGraphStore
from graph_rag.models import CanonicalEntity, EdgeProvenance, Triple
from graph_rag.scope import Scope
from spikes.pg_stores.pg_stores import (
    PgDocumentStore,
    PgEntityStore,
    PgGraphStore,
    PgObjectStore,
    ensure_schema,
)
from tests.isolation import (
    check_document_store,
    check_entity_store,
    check_graph_store,
    check_object_store,
)


@pytest.fixture()
def conn(pg_dsn: str) -> Iterator[psycopg.Connection]:
    c = psycopg.connect(pg_dsn, autocommit=True)
    ensure_schema(c, 384, reset=True)
    yield c
    c.close()


def test_scopes_are_isolated_in_every_store(conn: psycopg.Connection) -> None:
    check_object_store(PgObjectStore(conn))
    check_document_store(PgDocumentStore(conn), dims=384)
    check_entity_store(PgEntityStore(conn), dims=384)
    check_graph_store(PgGraphStore(conn))


def _snapshot(store: object, scope: Scope, ids: list[str]) -> tuple[object, ...]:
    assert isinstance(store, (InMemoryGraphStore, PgGraphStore))
    edges = sorted(
        (
            e.subject_id,
            e.predicate,
            e.object_id,
            e.provenance.source_doc_id,
            e.provenance.sentence_index,
            e.valid_to,
            e.superseded_by,
        )
        for i in ids
        for e in store.get_node_edges(scope, i)
    )
    sub = store.khop(scope, ids[:2], 2)
    return (
        store.node_count(scope),
        store.edge_count(scope),
        edges,
        sorted(n.canonical_id for n in sub.nodes),
        sorted((e.subject_id, e.object_id, e.provenance.source_doc_id) for e in sub.edges),
    )


@pytest.mark.parametrize("seed", range(5))
def test_random_graph_operations_match_the_fake(conn: psycopg.Connection, seed: int) -> None:
    rng = random.Random(seed)
    fake, real, scope = InMemoryGraphStore(), PgGraphStore(conn), Scope("t", "n")
    ids = [f"e{i}" for i in range(12)]

    def node(i: str) -> CanonicalEntity:
        return CanonicalEntity(canonical_id=i, name=i, type="ORG", aliases=[i.upper()])

    def triple() -> Triple:
        s, o = rng.sample(ids, 2)
        return Triple(
            subject_id=s,
            predicate=rng.choice(["PART_OF", "LOCATED_IN"]),
            object_id=o,
            provenance=EdgeProvenance(
                source_doc_id=f"d{rng.randrange(4)}",
                sentence_index=rng.randrange(3),
                source_sentence="s",
            ),
        )

    for store in (fake, real):
        store.upsert_entities(scope, [node(i) for i in ids])
    for _ in range(40):
        op = rng.choice(["write", "write", "delete_doc", "merge", "delete_nodes", "close"])
        batch = [triple() for _ in range(3)]
        keep, *drop = rng.sample(ids, 3)
        doc = f"d{rng.randrange(4)}"
        for store in (fake, real):
            if op == "write":
                store.write_triples(scope, batch)
            elif op == "delete_doc":
                store.delete_document_edges(scope, doc)
            elif op == "merge":
                store.merge_nodes(scope, keep, drop)
            elif op == "delete_nodes":
                store.delete_nodes(scope, drop[:1])
                store.upsert_entities(scope, [node(drop[0])])
            else:
                store.close_edge(scope, batch[0], "2026-01-01", "d9")
        assert _snapshot(real, scope, ids) == _snapshot(fake, scope, ids), (seed, op)
