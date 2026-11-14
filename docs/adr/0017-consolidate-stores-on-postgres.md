# ADR-0017: Consolidate the stateful stores on Postgres with pgvector (prototyped, not migrated)

- Status: Accepted as direction (2026-11-14); nothing is migrated and no product code changed
- Related: ADR-0005 (index split), ADR-0006 (graph), ADR-0010 (ports and fakes), ADR-0015 (scopes), ADR-0016 (Valkey trigger); GRA-54, GRA-146
- Spike code: [`spikes/pg_stores/`](../../spikes/pg_stores/) (stores, reused contract tests, bench run, scale run)

## Context

Four stateful services carry the product: Elasticsearch (documents, entities, vectors), Neo4j (graph), MinIO
(objects) and Valkey (the ingest queue, since ADR-0016). Kafka is optional and out of this comparison. Steady-state
resident memory measured on the full stack (semula run 4, GRA-54 comment): neo4j about 800 MB, elasticsearch about
740 MB, app about 620 MB, worker 590 MB to 1.1 GB with the transformer NER model, minio 90 to 110 MB, valkey about
20 MB. The stores alone are roughly 1.65 GB of a roughly 3 GB stack, and each is a separate thing to back up,
upgrade, secure and bill for. For a SaaS with small tenants (a few thousand entities each) that floor, not the
tenant's data, sets the margin. The question: can one Postgres replace most of them, and what does it cost in
behaviour?

## Options

- **A. Postgres + pgvector for documents, entities, vectors and graph.** The graph is plain tables with a
  recursive CTE for k-hop (Apache AGE is a variant, see below). Queue stays on Valkey (ADR-0016; Postgres
  `SKIP LOCKED` is possible but would redo a working, tested component for about 20 MB). Objects either stay on
  S3-compatible storage or move into a `bytea` column.
- **B. Status quo.** ES + Neo4j + MinIO + Valkey.
- **C. Middle path.** Postgres (documents, entities, vectors, objects) + Neo4j (graph) + Valkey.

## Numbers

**Memory** (measured for the existing services; Postgres measured here as `docker stats` of `pgvector/pgvector:0.8.1-pg17`
with `shared_buffers=128MB`):

| | stores' resident memory | notes |
|---|---|---|
| B status quo | about 1,650 MB (neo4j 800, es 740, minio about 100, valkey 20) | |
| C | about 1,070 MB (pg about 150 to 210, neo4j 800, minio about 100, valkey 20) | Neo4j still dominates |
| A | about 230 MB (pg about 150 to 210, valkey 20; objects in pg) | saves about 1.4 GB, about half the stack |

Postgres: 68 MB idle, 146 to 212 MB after the benchmark world and after loading the synthetic scale set (50,000
entities with 384-dim vectors and 150,000 edges, 165 MB on disk). Not measured: a Postgres under concurrent
load, and the real stack's ES and Neo4j on this host (the spike was limited to one container).

**Managed prices** (looked up 2026-11-14; third-party summaries disagree in places, so confirm on the vendor page):

| service | price | source |
|---|---|---|
| Supabase Pro (Postgres + pgvector) | $25/month base, $10 compute credit; Small (2 GB) instance $15, so about $30 to $36/month for a small SaaS with 20 GB | [Supabase pricing](https://supabase.com/pricing), [guide](https://www.jetadmin.io/blog/supabase-pricing-2026-guide-to-plans-limits-and-real-world-costs/) |
| Neon Launch (Postgres + pgvector) | $0.106 per CU-hour (a CU is about 4 GB), $0.35 per GB-month, scales to zero | [Neon pricing](https://neon.com/pricing), [FAQ](https://neon.com/faqs/postgres-services-no-minimum-charge) |
| Neo4j AuraDB Professional | $0.09 per GB-hour, 1 GB minimum (about $65 per GB-month) | [Neo4j pricing](https://neo4j.com/pricing) |
| Elastic Cloud Hosted, Standard | from about $99/month | [Elastic pricing](https://www.elastic.co/pricing/cloud-hosted) |
| Hetzner Cloud CX22 / CX32 (self-host) | about EUR 4.5 / EUR 8 per month, sources differ after the 2026 increases | [price roundup](https://cloudpricecheck-com.pages.dev/hetzner/cloud-servers-pricing) |

Rough monthly cost to host one small deployment: **A** is one managed Postgres, about $30 to $40 (Supabase) or
well under that on Neon when idle; **B** is Aura Professional at 1 to 2 GB ($65 to $130) plus Elastic Standard
($99) plus an object store and a queue, so about $200 to $250 before the app; **C** adds Aura to A, about $100 to
$170. Self-hosted, A fits the app, worker and Postgres in a 8 GB VM; B needs about 3 GB resident before any load,
and ES wants more headroom than that. These are list prices, not a quote and not measured bills.

**Query latency on this project's multi-hop retrieval** (`spikes/pg_stores/run_bench.py`, `small` world: 150
entities, 300 edges, 117 documents, 120 questions, FakeEmbedder, offline pipeline, graph-rag graph view, same
questions through the in-memory stores and the Postgres stores):

| | in-memory | postgres |
|---|---|---|
| 1-hop path@5 / recall@5 | 0.900 / 0.900 | 0.900 / 0.900 |
| 2-hop path@5 / recall@5 | 0.067 / 0.333 | 0.033 / 0.317 |
| 3-hop, 4-hop path@5 | 0.000, 0.000 | 0.000, 0.000 |
| all questions path@5 / recall@5 | 0.242 / 0.360 | 0.233 / 0.359 |
| query p50 / p95 (ms) | 17 to 32 / 19 to 67 (runs vary) | 18 / 40 |
| ingest, 117 documents | 0.2 to 0.4 s | 3 to 5 s |

Top-5 sentences are identical on 111 of 120 questions and the top-3 ranked nodes on 108 of 120. The one question
printed (Q004, 2 hops, gold "Lomo Industries") gets the same ranked nodes and the same top sentences from both.
The differences are tie-breaks: pgvector stores float32, so cosines that are equal or within 1e-16 in the fake's
float64 order differently (scores differ at the 8th decimal), which changes which entities seed the subgraph. The
2-hop path@5 gap is one question in 30. Rounding the score to 6 places in the `ORDER BY` removed some but not all
of it. The in-memory fake is a brute-force Python scan, so its latency is not a speed target; the point is that
Postgres at this size costs nothing visible per query.

**Scale** (`spikes/pg_stores/scale.py`; 5 tenants, 10,000 entities and 30,000 edges each, random 384-dim vectors,
timings only, one tenant queried): k-hop by recursive CTE from 5 seeds is 5 ms at 1 hop, 10 ms at 2 hops (114
nodes) and 54 ms at 3 hops (621 nodes); exact kNN k=10 with the scope filter over 10,000 of 50,000 rows is 8.8 ms.
With an HNSW index over all rows (40 s to build) the planner uses it and a scoped query **silently returned 8 of
10 rows**, because the scope filter is applied after the index scan; with `hnsw.iterative_scan = relaxed_order` it
returned 10 of 10 in 3.1 ms. Neo4j's and ES's numbers at the same size were not measured.

## Comparison

- **Ops burden.** One Postgres (one backup, one upgrade path, one credential set) against three services plus
  queue. Postgres is the most widely managed of them. B and C keep the JVM tuning and the ES index lifecycle.
- **Managed availability.** pgvector is offered by most managed Postgres (Supabase, Neon and others above).
  Neo4j and Elastic exist as managed products but at the prices above and with minimums. AGE is not offered by the
  common managed services I looked at (not checked per vendor, so treat as unverified), which is why the graph here
  uses CTEs.
- **Multi-tenant isolation (ADR-0015).** `Scope` maps to a `scope` and `tenant` column on every table, the same
  "a field on every record" model, and the prototype passes the shared two-scope checks (`tests/isolation.py`). The
  MinIO reserved-prefix rule (`_scopes/`) has no Postgres equivalent because scope is a column, so a key cannot
  spell another scope. Postgres also offers row-level security as a second guard against a forgotten filter, which
  none of ES, Neo4j Community or MinIO gives; not built or tested here. The cost is the same one the ADR names:
  every query must filter, and the HNSW result above shows a filter can also cost recall.
- **Query quality.** Same on the bench world at this size (above); not shown for real bge vectors, larger graphs or
  the LLM-built graph. Dense/entity kNN is exact and equal to the fake until an ANN index is added; BM25 is not used
  by the product path today.
- **Graph depth.** A CTE is fine to 3 hops on 10,000-node scopes. Neo4j's advantage (index-free adjacency, Cypher
  paths, algorithms) shows at deeper traversals and much larger graphs, which this product does not run.

## Decision

Adopt **A** as the target for the small-tenant and dev profile: documents, entities, vectors, graph and objects in
one Postgres with pgvector, queue stays on Valkey. The ports (ADR-0010) make it a second backend, selected by
configuration, not a rewrite; ES, Neo4j and MinIO stay selectable and stay the reference until a Postgres backend
passes the same contract suite in CI. Objects go in `bytea` for the small profile (documents are text) and stay on
S3-compatible storage where tenants store large files. Apache AGE is not chosen: it is not on common managed
Postgres and the CTE already meets the port. C is rejected as the default because Neo4j alone is about 800 MB and
$65 or more per month, and it is the larger of the two savings; it remains a valid configuration for a tenant whose
graph outgrows the CTE.

## Consequences

- Follow-up cards (not done here): promote the spike to `graph_rag/adapters/pg_*` with `psycopg` as an optional
  extra; run its contract tests in CI with the pinned `pgvector/pgvector` image; an `HNSW` plus iterative-scan
  setting or a per-scope exact scan, chosen by tenant size; connection pool and thread safety (the spike uses one
  connection); a migration tool, since the product has none for schema changes; export/delete of a tenant by SQL.
- The bench gate's floors were set against the in-memory fake. A Postgres backend gives tie-break differences
  (above), so a gate run on it needs its own floors or a deterministic tie rule shared by every backend.
- Dropping Elasticsearch drops its text analysis. The product path uses no BM25 today; if it does later,
  `tsvector` must be evaluated. Not tested.
- Two dev profiles result (light: Postgres; full: ES + Neo4j) and both need CI contract coverage.

## What would change the decision

- A real-embedder (bge) or LLM-graph bench run where Postgres differs from the fake by more than tie noise.
- Tenants whose graphs reach hundreds of thousands of edges per scope or queries needing more than 3 hops, where the
  CTE's 54 ms at 3 hops (here, 621 nodes) grows past an interactive budget.
- Vector recall loss under scope filters that iterative scan does not fix at real tenant counts.
- A managed-service price change that removes the gap (Aura's free tier, say), or a requirement for BM25 or ES
  features the product starts to use.

## What the prototype proved and did not

Proved: `ObjectStore`, `DocumentStore`, `EntityStore` and `GraphStore` can be implemented on Postgres with
pgvector. 30 of the existing contract tests ran unchanged against it (the ES entity, entity-search, document and
graph modules, plus the MinIO round-trip tests), and 5 seeded random-operation runs matched the in-memory fake on
merges, deletes, closes and k-hop. The shared two-scope isolation checks pass. The offline pipeline ran end to end
on it and answered 120 bench questions with the fake's scores, within tie noise.

Did not prove: the real ES or Neo4j results on this host (one container only), real embedder vectors, the
LLM-built graph, concurrency, a connection pool, ANN recall on real data, row-level security, Apache AGE (not
tried), `SKIP LOCKED` as a queue (not tried), managed-service behaviour (everything ran in a local container), or
any migration of existing data. Six contract tests tied to a backend (Neo4j labels, raw ES writes, the MinIO
reserved-key rule) did not run. The scale numbers use random vectors and a random graph and say nothing about
quality.
