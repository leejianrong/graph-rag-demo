# ruff: noqa: E501
"""GRA-54 spike: Postgres at a size the demo worlds never reach (synthetic, random vectors).

Five tenants of 10,000 entities and 30,000 edges each (average degree 6, 384-dim random
vectors), in the prototype's own tables. Measures what the ADR needs and the bench cannot: k-hop
by recursive CTE, exact kNN with a scope filter, and kNN through an HNSW index with the scope
filter applied after the index scan. Random vectors, random graph: timings only, not quality.

    PG_DSN=... .venv/bin/python -m spikes.pg_stores.scale
"""

from __future__ import annotations

import os
import random
import statistics
import time

import psycopg

from graph_rag.scope import Scope
from spikes.pg_stores.pg_stores import PgGraphStore, ensure_schema

TENANTS, N, E, DIMS = 5, 10_000, 30_000, 384


def timed(fn, reps: int = 5) -> float:  # type: ignore[no-untyped-def]
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


def main() -> None:
    conn = psycopg.connect(os.environ["PG_DSN"], autocommit=True)
    ensure_schema(conn, DIMS, reset=True)
    rng = random.Random(0)
    t0 = time.perf_counter()
    for t in range(TENANTS):
        scope = f"t{t}"
        conn.execute(
            "INSERT INTO entities (scope, tenant, canonical_id, name, type, aliases, norm_names, "
            "vec, first_seen) SELECT %s, %s, 'e'||g, 'n'||g, 'ORG', '[]', ARRAY['n'||g], "
            "(SELECT ('['||string_agg(random()::text, ',')||']')::vector FROM generate_series(1,%s) "
            "WHERE g > 0), 'x' FROM generate_series(1,%s) g",
            (scope, scope, DIMS, N),
        )
        conn.execute(
            "INSERT INTO nodes (scope, tenant, canonical_id, name, type, aliases) "
            "SELECT %s, %s, 'e'||g, 'n'||g, 'ORG', '[]' FROM generate_series(1,%s) g",
            (scope, scope, N),
        )
        pairs = [(rng.randint(1, N), rng.randint(1, N)) for _ in range(E)]
        with (
            conn.cursor() as cur,
            cur.copy(
                "COPY edges (scope, tenant, subject_id, predicate, object_id, source_doc_id, "
                "sentence_index, triple) FROM STDIN"
            ) as cp,
        ):
            for a, b in pairs:
                triple = (
                    f'{{"subject_id":"e{a}","predicate":"PART_OF","object_id":"e{b}","provenance":'
                    '{"source_doc_id":"d","sentence_index":0,"source_sentence":"s"}}'
                )
                cp.write_row((scope, scope, f"e{a}", "PART_OF", f"e{b}", "d", 0, triple))
    conn.execute("ANALYZE")
    print(f"loaded {TENANTS * N} entities, {TENANTS * E} edges in {time.perf_counter() - t0:.0f}s")
    size = conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()
    print("database size", size[0] if size else "?")

    graph = PgGraphStore(conn)
    seeds = [f"e{i}" for i in (1, 2, 3, 4, 5)]
    for hops in (1, 2, 3):
        ms = timed(lambda h=hops: graph.khop(Scope("t0"), seeds, h), reps=3)  # type: ignore[misc]
        size = len(graph.khop(Scope("t0"), seeds, hops).nodes)
        print(f"khop {hops} hops from 5 seeds, one tenant: {ms:.0f} ms ({size} nodes)")

    q = "[" + ",".join(f"{rng.random():.4f}" for _ in range(DIMS)) + "]"
    knn = "SELECT canonical_id FROM entities WHERE scope=%s ORDER BY vec <=> %s::vector LIMIT 10"
    print(
        f"exact kNN k=10, scope filter, 10k of 50k rows: {timed(lambda: conn.execute(knn, ('t0', q)).fetchall()):.1f} ms"
    )

    t = time.perf_counter()
    conn.execute("CREATE INDEX entities_hnsw ON entities USING hnsw (vec vector_cosine_ops)")
    print(f"HNSW build over 50k x {DIMS}: {time.perf_counter() - t:.0f}s")
    conn.execute("ANALYZE entities")
    ann = timed(lambda: conn.execute(knn, ("t0", q)).fetchall())
    got = len(conn.execute(knn, ("t0", q)).fetchall())
    plan = conn.execute("EXPLAIN " + knn, ("t0", q)).fetchall()
    print(
        f"kNN k=10 with scope filter, index present: {ann:.1f} ms, {got} rows; plan: {plan[0][0][:60]}"
    )
    conn.execute("SET enable_seqscan = off")
    forced = conn.execute(knn, ("t0", q)).fetchall()
    print(f"  index forced, ef_search=40: {len(forced)} of 10 rows returned (post-filter by scope)")
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    conn.execute("SET hnsw.max_scan_tuples = 20000")
    forced = conn.execute(knn, ("t0", q)).fetchall()
    ms = timed(lambda: conn.execute(knn, ("t0", q)).fetchall())
    print(f"  index forced + iterative_scan: {len(forced)} of 10 rows, {ms:.1f} ms")
    conn.close()


if __name__ == "__main__":
    main()
