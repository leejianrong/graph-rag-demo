# ruff: noqa: E501
"""GRA-54 spike: the four store ports over one Postgres with pgvector (ADR-0017).

Scratch code, not wired into the product and not part of any package: it needs ``psycopg``, which
the project does not depend on (``uv run --with "psycopg[binary]" ...``). Every table carries the
``scope`` key (``Scope.key``) and the ``tenant`` so ``scopes(tenant)`` works; every query filters on
``scope``, which is ADR-0015's "a field on every record" in SQL.

Deliberately simple: one connection, autocommit with explicit transactions, exact (non-indexed)
vector scans filtered by scope. It is a prototype of the *semantics*, not of performance tuning.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import psycopg

from graph_rag.merging import edge_key, rewire
from graph_rag.models import (
    CanonicalEntity,
    CuratedType,
    DocumentRecord,
    EntitySource,
    Subgraph,
    SupportingSentence,
    Triple,
)
from graph_rag.normalize import normalize_name
from graph_rag.scope import Scope
from graph_rag.timeutil import utc_now_iso

__all__ = ["PgObjectStore", "PgDocumentStore", "PgEntityStore", "PgGraphStore", "ensure_schema"]


def _vec(v: list[float] | None) -> str | None:
    """pgvector text input: ``[0.1,0.2]``."""
    return None if v is None else "[" + ",".join(repr(float(x)) for x in v) + "]"


def _parse_vec(text: str | None) -> list[float] | None:
    return None if text is None else [float(x) for x in text.strip("[]").split(",")]


def _scopes(conn: psycopg.Connection, table: str, tenant: str) -> list[Scope]:
    rows = conn.execute(f"SELECT DISTINCT scope FROM {table} WHERE tenant = %s", (tenant,))
    found = []
    for (key,) in rows:
        t, _, ns = key.partition("/")
        found.append(Scope(t, ns or None))
    return sorted(found, key=lambda s: (s.tenant, s.namespace or ""))


def ensure_schema(conn: psycopg.Connection, dims: int, *, reset: bool = False) -> None:
    """Create the extension and tables (``reset`` drops them first)."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    tables = ("objects", "sentences", "doc_links", "documents", "entities", "edges", "nodes")
    if reset:
        for t in tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            scope text, tenant text, bucket text, object_key text, data bytea,
            PRIMARY KEY (scope, bucket, object_key))""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            seq bigserial, scope text, tenant text, document_id text, object_key text, record jsonb,
            PRIMARY KEY (scope, document_id))""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_links (
            scope text, document_id text, canonical_id text, mentions int,
            PRIMARY KEY (scope, document_id, canonical_id))""")
    conn.execute("CREATE INDEX IF NOT EXISTS doc_links_cid ON doc_links (scope, canonical_id)")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS sentences (
            scope text, document_id text, pos int, sentence_index int, text text,
            char_start int, char_end int, vec vector({dims}),
            PRIMARY KEY (scope, document_id, pos))""")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS entities (
            seq bigserial, scope text, tenant text, canonical_id text, name text, type text,
            aliases jsonb, norm_names text[], vec vector({dims}), first_seen text,
            PRIMARY KEY (scope, canonical_id))""")
    conn.execute("CREATE INDEX IF NOT EXISTS entities_norm ON entities USING gin (norm_names)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            seq bigserial, scope text, tenant text, canonical_id text, name text, type text,
            aliases jsonb, PRIMARY KEY (scope, canonical_id))""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id bigserial PRIMARY KEY, scope text, tenant text, subject_id text, predicate text,
            object_id text, source_doc_id text, sentence_index int, triple jsonb,
            valid_to text, superseded_by text)""")
    conn.execute("CREATE INDEX IF NOT EXISTS edges_s ON edges (scope, subject_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS edges_o ON edges (scope, object_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS edges_d ON edges (scope, source_doc_id)")


class PgObjectStore:
    """:class:`~graph_rag.ports.ObjectStore` on a ``bytea`` column."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._c = conn

    def put(self, scope: Scope, bucket: str, object_key: str, data: bytes) -> None:
        self._c.execute(
            "INSERT INTO objects VALUES (%s,%s,%s,%s,%s) ON CONFLICT (scope, bucket, object_key) "
            "DO UPDATE SET data = EXCLUDED.data",
            (scope.key, scope.tenant, bucket, object_key, data),
        )

    def get(self, scope: Scope, bucket: str, object_key: str) -> bytes:
        row = self._c.execute(
            "SELECT data FROM objects WHERE scope=%s AND bucket=%s AND object_key=%s",
            (scope.key, bucket, object_key),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No object at {bucket}/{object_key}")
        return bytes(row[0])

    def delete(self, scope: Scope, bucket: str, object_key: str) -> None:
        self._c.execute(
            "DELETE FROM objects WHERE scope=%s AND bucket=%s AND object_key=%s",
            (scope.key, bucket, object_key),
        )

    def scopes(self, tenant: str) -> list[Scope]:
        return _scopes(self._c, "objects", tenant)

    def keys(self, scope: Scope) -> list[tuple[str, str]]:
        rows = self._c.execute(
            "SELECT bucket, object_key FROM objects WHERE scope=%s", (scope.key,)
        ).fetchall()
        return [(b, k) for b, k in rows]


class PgDocumentStore:
    """:class:`~graph_rag.ports.DocumentStore`: the record as JSONB, sentences as vector rows."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._c = conn

    def upsert(self, scope: Scope, record: DocumentRecord) -> None:
        with self._c.transaction():
            self._c.execute(
                "INSERT INTO documents (scope, tenant, document_id, object_key, record) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (scope, document_id) "
                "DO UPDATE SET record = EXCLUDED.record, object_key = EXCLUDED.object_key",
                (
                    scope.key,
                    scope.tenant,
                    record.document_id,
                    record.object_key,
                    record.model_dump_json(),
                ),
            )
            for table in ("doc_links", "sentences"):
                self._c.execute(
                    f"DELETE FROM {table} WHERE scope=%s AND document_id=%s",
                    (scope.key, record.document_id),
                )
            counts: dict[str, int] = {}
            for link in record.el_result:
                counts[link.canonical_id] = counts.get(link.canonical_id, 0) + 1
            for cid, n in counts.items():
                self._c.execute(
                    "INSERT INTO doc_links VALUES (%s,%s,%s,%s)",
                    (scope.key, record.document_id, cid, n),
                )
            for pos, vector in enumerate(record.sentence_vectors or []):
                if pos >= len(record.sentences):
                    continue
                s = record.sentences[pos]
                self._c.execute(
                    "INSERT INTO sentences VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    (
                        scope.key,
                        record.document_id,
                        pos,
                        s.index,
                        s.text,
                        s.char_start,
                        s.char_end,
                        _vec(vector),
                    ),
                )

    def get(self, scope: Scope, document_id: str) -> DocumentRecord | None:
        row = self._c.execute(
            "SELECT record FROM documents WHERE scope=%s AND document_id=%s",
            (scope.key, document_id),
        ).fetchone()
        return None if row is None else DocumentRecord.model_validate(row[0])

    def delete(self, scope: Scope, document_id: str) -> bool:
        with self._c.transaction():
            for table in ("doc_links", "sentences"):
                self._c.execute(
                    f"DELETE FROM {table} WHERE scope=%s AND document_id=%s",
                    (scope.key, document_id),
                )
            cur = self._c.execute(
                "DELETE FROM documents WHERE scope=%s AND document_id=%s", (scope.key, document_id)
            )
        return cur.rowcount > 0

    def all(self, scope: Scope) -> list[DocumentRecord]:
        rows = self._c.execute(
            "SELECT record FROM documents WHERE scope=%s ORDER BY seq", (scope.key,)
        ).fetchall()
        return [DocumentRecord.model_validate(r[0]) for r in rows]

    def scopes(self, tenant: str) -> list[Scope]:
        return _scopes(self._c, "documents", tenant)

    def entities_in_use(
        self, scope: Scope, canonical_ids: set[str], *, excluding: str | None = None
    ) -> set[str]:
        rows = self._c.execute(
            "SELECT DISTINCT canonical_id FROM doc_links WHERE scope=%s AND canonical_id = ANY(%s) "
            "AND document_id IS DISTINCT FROM %s",
            (scope.key, list(canonical_ids), excluding),
        ).fetchall()
        return {r[0] for r in rows}

    def entity_sources(self, scope: Scope, canonical_id: str) -> list[EntitySource]:
        rows = self._c.execute(
            "SELECT l.document_id, d.object_key, l.mentions FROM doc_links l "
            "JOIN documents d ON d.scope=l.scope AND d.document_id=l.document_id "
            "WHERE l.scope=%s AND l.canonical_id=%s ORDER BY l.mentions DESC, l.document_id "
            "LIMIT 1000",
            (scope.key, canonical_id),
        ).fetchall()
        return [EntitySource(document_id=a, name=b, mentions=c) for a, b, c in rows]

    def search_sentences(
        self, scope: Scope, *, vector: list[float], top_k: int
    ) -> list[SupportingSentence]:
        rows = self._c.execute(
            "SELECT document_id, text, char_start, char_end, sentence_index, "
            "1 - (vec <=> %s::vector) AS score FROM sentences WHERE scope=%s "
            "ORDER BY round((1 - (vec <=> %s::vector))::numeric, 6) DESC, document_id, sentence_index "
            "LIMIT %s",
            (_vec(vector), scope.key, _vec(vector), top_k),
        ).fetchall()
        return [
            SupportingSentence(
                document_id=a, text=b, char_start=c, char_end=d, sentence_index=e, score=f
            )
            for a, b, c, d, e, f in rows
        ]


def _entity_from(row: tuple[Any, ...]) -> CanonicalEntity:
    cid, name, type_, aliases, vec = row
    return CanonicalEntity(
        canonical_id=cid, name=name, type=type_, aliases=list(aliases), vector=_parse_vec(vec)
    )


_ENTITY_COLS = "canonical_id, name, type, aliases, vec::text"


class PgEntityStore:
    """:class:`~graph_rag.ports.EntityStore`: blocking by a GIN array, kNN by ``<=>``."""

    def __init__(self, conn: psycopg.Connection, clock: Callable[[], str] | None = None) -> None:
        self._c = conn
        self._clock = clock or utc_now_iso

    def upsert(
        self, scope: Scope, entity: CanonicalEntity, *, first_seen: str | None = None
    ) -> None:
        norm = sorted({normalize_name(entity.name), *map(normalize_name, entity.aliases)})
        self._c.execute(
            "INSERT INTO entities (scope, tenant, canonical_id, name, type, aliases, norm_names, "
            "vec, first_seen) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector,%s) "
            "ON CONFLICT (scope, canonical_id) DO UPDATE SET name=EXCLUDED.name, "
            "type=EXCLUDED.type, aliases=EXCLUDED.aliases, norm_names=EXCLUDED.norm_names, "
            "vec=EXCLUDED.vec, first_seen=COALESCE(%s, entities.first_seen)",
            (
                scope.key,
                scope.tenant,
                entity.canonical_id,
                entity.name,
                entity.type,
                json.dumps(entity.aliases),
                norm,
                _vec(entity.vector),
                first_seen or self._clock(),
                first_seen,
            ),
        )

    def delete(self, scope: Scope, canonical_id: str) -> None:
        self._c.execute(
            "DELETE FROM entities WHERE scope=%s AND canonical_id=%s", (scope.key, canonical_id)
        )

    def get(self, scope: Scope, canonical_id: str) -> CanonicalEntity | None:
        row = self._c.execute(
            f"SELECT {_ENTITY_COLS} FROM entities WHERE scope=%s AND canonical_id=%s",
            (scope.key, canonical_id),
        ).fetchone()
        return None if row is None else _entity_from(row)

    def block_candidates(
        self, scope: Scope, *, entity_type: CuratedType, normalized_name: str
    ) -> list[CanonicalEntity]:
        rows = self._c.execute(
            f"SELECT {_ENTITY_COLS} FROM entities WHERE scope=%s AND type=%s "
            "AND norm_names @> ARRAY[%s] ORDER BY seq",
            (scope.key, entity_type, normalized_name),
        ).fetchall()
        return [_entity_from(r) for r in rows]

    def knn(
        self,
        scope: Scope,
        *,
        vector: list[float],
        entity_type: CuratedType | None = None,
        top_k: int,
    ) -> list[tuple[CanonicalEntity, float]]:
        rows = self._c.execute(
            f"SELECT {_ENTITY_COLS}, 1 - (vec <=> %s::vector) AS score FROM entities "
            "WHERE scope=%s AND vec IS NOT NULL AND (%s::text IS NULL OR type=%s) "
            "ORDER BY round((1 - (vec <=> %s::vector))::numeric, 6) DESC, seq LIMIT %s",
            (_vec(vector), scope.key, entity_type, entity_type, _vec(vector), top_k),
        ).fetchall()
        return [(_entity_from(r[:5]), float(r[5])) for r in rows]

    def search(
        self,
        scope: Scope,
        *,
        query: str | None = None,
        entity_type: CuratedType | None = None,
        limit: int,
        after: tuple[str, str] | None = None,
    ) -> list[tuple[CanonicalEntity, str]]:
        key = normalize_name(query or "")
        seen, last = after if after is not None else (None, None)
        rows = self._c.execute(
            f"SELECT {_ENTITY_COLS}, first_seen FROM entities WHERE scope=%s "
            "AND (%s::text IS NULL OR type=%s) "
            "AND (%s = '' OR EXISTS "
            "(SELECT 1 FROM unnest(norm_names) n WHERE position(%s in n) > 0)) "
            "AND (%s::text IS NULL OR first_seen < %s OR (first_seen = %s AND canonical_id > %s)) "
            "ORDER BY first_seen DESC, canonical_id LIMIT %s",
            (scope.key, entity_type, entity_type, key, key, seen, seen, seen, last, limit),
        ).fetchall()
        return [(_entity_from(r[:5]), r[5]) for r in rows]

    def first_seen(self, scope: Scope, canonical_id: str) -> str | None:
        row = self._c.execute(
            "SELECT first_seen FROM entities WHERE scope=%s AND canonical_id=%s",
            (scope.key, canonical_id),
        ).fetchone()
        return None if row is None else row[0]

    def count(self, scope: Scope) -> int:
        row = self._c.execute(
            "SELECT count(*) FROM entities WHERE scope=%s", (scope.key,)
        ).fetchone()
        return int(row[0]) if row else 0

    def all(self, scope: Scope) -> list[CanonicalEntity]:
        rows = self._c.execute(
            f"SELECT {_ENTITY_COLS} FROM entities WHERE scope=%s ORDER BY seq", (scope.key,)
        ).fetchall()
        return [_entity_from(r) for r in rows]

    def scopes(self, tenant: str) -> list[Scope]:
        return _scopes(self._c, "entities", tenant)


def _triple_from(triple: dict[str, Any], valid_to: str | None, superseded_by: str | None) -> Triple:
    return Triple.model_validate({**triple, "valid_to": valid_to, "superseded_by": superseded_by})


class PgGraphStore:
    """:class:`~graph_rag.ports.GraphStore` as ``nodes`` and ``edges`` tables.

    ``khop`` is a depth-bounded recursive CTE over the undirected edges.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._c = conn

    def upsert_entities(self, scope: Scope, entities: list[CanonicalEntity]) -> None:
        for e in entities:
            self._c.execute(
                "INSERT INTO nodes (scope, tenant, canonical_id, name, type, aliases) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (scope, canonical_id) DO UPDATE "
                "SET name=EXCLUDED.name, type=EXCLUDED.type, aliases=EXCLUDED.aliases",
                (scope.key, scope.tenant, e.canonical_id, e.name, e.type, json.dumps(e.aliases)),
            )

    def _insert_edge(self, scope: Scope, t: Triple) -> None:
        self._c.execute(
            "INSERT INTO edges (scope, tenant, subject_id, predicate, object_id, source_doc_id, "
            "sentence_index, triple, valid_to, superseded_by) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                scope.key,
                scope.tenant,
                t.subject_id,
                t.predicate,
                t.object_id,
                t.provenance.source_doc_id,
                t.provenance.sentence_index,
                t.model_dump_json(),
                t.valid_to,
                t.superseded_by,
            ),
        )

    def write_triples(self, scope: Scope, triples: list[Triple]) -> None:
        with self._c.transaction():
            for t in triples:
                self._insert_edge(scope, t)

    def delete_document_edges(self, scope: Scope, source_doc_id: str) -> None:
        with self._c.transaction():
            self._c.execute(
                "DELETE FROM edges WHERE scope=%s AND source_doc_id=%s", (scope.key, source_doc_id)
            )
            self._c.execute(
                "UPDATE edges SET valid_to=NULL, superseded_by=NULL "
                "WHERE scope=%s AND superseded_by=%s",
                (scope.key, source_doc_id),
            )

    def close_edge(self, scope: Scope, edge: Triple, valid_to: str, superseded_by: str) -> None:
        self._c.execute(
            "UPDATE edges SET valid_to=%s, superseded_by=%s WHERE scope=%s AND subject_id=%s "
            "AND predicate=%s AND object_id=%s AND source_doc_id=%s AND sentence_index=%s",
            (valid_to, superseded_by, scope.key, *edge_key(edge)),
        )

    def delete_nodes(self, scope: Scope, canonical_ids: list[str]) -> None:
        with self._c.transaction():
            self._c.execute(
                "DELETE FROM edges WHERE scope=%s "
                "AND (subject_id = ANY(%s) OR object_id = ANY(%s))",
                (scope.key, canonical_ids, canonical_ids),
            )
            self._c.execute(
                "DELETE FROM nodes WHERE scope=%s AND canonical_id = ANY(%s)",
                (scope.key, canonical_ids),
            )

    def _edges(self, where: str, args: tuple[Any, ...]) -> list[Triple]:
        rows = self._c.execute(
            f"SELECT triple, valid_to, superseded_by FROM edges WHERE {where} ORDER BY id", args
        ).fetchall()
        return [_triple_from(*r) for r in rows]

    def merge_nodes(self, scope: Scope, keep_id: str, drop_ids: list[str]) -> None:
        drop = set(drop_ids) - {keep_id}
        if not drop:
            return
        d = list(drop)
        with self._c.transaction():
            touching = self._edges(
                "scope=%s AND (subject_id = ANY(%s) OR object_id = ANY(%s))", (scope.key, d, d)
            )
            present = self._edges(
                "scope=%s AND NOT (subject_id = ANY(%s) OR object_id = ANY(%s)) "
                "AND (subject_id=%s OR object_id=%s)",
                (scope.key, d, d, keep_id, keep_id),
            )
            moved = rewire(touching, keep_id, drop, present)
            self._c.execute(
                "DELETE FROM edges WHERE scope=%s "
                "AND (subject_id = ANY(%s) OR object_id = ANY(%s))",
                (scope.key, d, d),
            )
            for t in moved:
                self._insert_edge(scope, t)
            self._c.execute(
                "DELETE FROM nodes WHERE scope=%s AND canonical_id = ANY(%s)", (scope.key, d)
            )

    def split_nodes(
        self, scope: Scope, entities: list[CanonicalEntity], remove: list[Triple], add: list[Triple]
    ) -> None:
        with self._c.transaction():
            self.upsert_entities(scope, entities)
            stored = {edge_key(t) for t in self._edges("scope=%s", (scope.key,))}
            for t in add:
                if edge_key(t) not in stored:
                    stored.add(edge_key(t))
                    self._insert_edge(scope, t)
            for t in remove:
                self._c.execute(
                    "DELETE FROM edges WHERE scope=%s AND subject_id=%s AND predicate=%s "
                    "AND object_id=%s AND source_doc_id=%s AND sentence_index=%s",
                    (scope.key, *edge_key(t)),
                )

    def khop(self, scope: Scope, seed_ids: list[str], hops: int) -> Subgraph:
        reach = """
            WITH RECURSIVE reach(id, d) AS (
                SELECT canonical_id, 0 FROM nodes
                WHERE scope=%(s)s AND canonical_id = ANY(%(seeds)s)
              UNION
                SELECT n.canonical_id, r.d + 1 FROM reach r
                JOIN edges e ON e.scope=%(s)s AND (e.subject_id=r.id OR e.object_id=r.id)
                JOIN nodes n ON n.scope=%(s)s
                    AND n.canonical_id = CASE WHEN e.subject_id=r.id THEN e.object_id
                                              ELSE e.subject_id END
                WHERE r.d < %(h)s)
            SELECT DISTINCT id FROM reach"""
        params = {"s": scope.key, "seeds": seed_ids, "h": max(hops, 0)}
        ids = [r[0] for r in self._c.execute(reach, params).fetchall()]
        node_rows = self._c.execute(
            "SELECT canonical_id, name, type, aliases, NULL::text FROM nodes "
            "WHERE scope=%s AND canonical_id = ANY(%s) ORDER BY seq",
            (scope.key, ids),
        ).fetchall()
        edges = self._edges(
            "scope=%s AND subject_id = ANY(%s) AND object_id = ANY(%s)", (scope.key, ids, ids)
        )
        return Subgraph(nodes=[_entity_from(r) for r in node_rows], edges=edges)

    def node_count(self, scope: Scope) -> int:
        row = self._c.execute("SELECT count(*) FROM nodes WHERE scope=%s", (scope.key,)).fetchone()
        return int(row[0]) if row else 0

    def edge_count(self, scope: Scope) -> int:
        row = self._c.execute("SELECT count(*) FROM edges WHERE scope=%s", (scope.key,)).fetchone()
        return int(row[0]) if row else 0

    def scopes(self, tenant: str) -> list[Scope]:
        found = {*_scopes(self._c, "nodes", tenant), *_scopes(self._c, "edges", tenant)}
        return sorted(found, key=lambda s: (s.tenant, s.namespace or ""))

    def get_node(self, scope: Scope, canonical_id: str) -> CanonicalEntity | None:
        row = self._c.execute(
            "SELECT canonical_id, name, type, aliases, NULL::text FROM nodes "
            "WHERE scope=%s AND canonical_id=%s",
            (scope.key, canonical_id),
        ).fetchone()
        return None if row is None else _entity_from(row)

    def get_node_edges(self, scope: Scope, canonical_id: str) -> list[Triple]:
        return self._edges(
            "scope=%s AND (subject_id=%s OR object_id=%s)", (scope.key, canonical_id, canonical_id)
        )
