# ADR-0015: Tenants are a field on every record, scoped by a `Scope` argument

- Status: Accepted (2026-11-07)
- Related: ADR-0001 (document id), ADR-0004 (entity linking), ADR-0005 (index split), ADR-0014 (agent API); GRA-19, GRA-20, GRA-21, GRA-45

## Context

Every store is global today. `document_id` is `sha256(bucket/key)` and the entity `canonical_id` is
`sha256(el:type:name)`, so two users who each mention "Acme Corp" share one entity, and nothing in a port call says
whose data it is. ADR-0014 already says the API key names the tenant and that an optional `namespace` partitions
memory inside a tenant. This ADR decides how the stores honour that. It is written for a possible hosted service
(board epics GRA-E4 and GRA-E9, with validation still open in GRA-E11), and it also serves one self-hosted instance
shared by several people or projects.

## Decision

**A `tenant` field on every record, not a separate home per tenant.** Elasticsearch keeps its two indexes, each
document carrying `tenant` and `scope` keywords (`scope` is `tenant` or `tenant/namespace`) and every query, kNN
included, filters on `scope`. Neo4j keeps one database, each `Entity` node carrying `tenant` and an indexed `scope`
property. MinIO keeps one bucket: a non-default scope's objects live under `_scopes/<tenant>/root/<key>` (or
`_scopes/<tenant>/ns/<namespace>/<key>`), and no caller may name a key that starts with `_scopes/`, so a default-scope
key cannot spell another scope's location. A record with no `scope` field was written before scopes existed and
belongs to the default scope, which is why nothing is migrated. (Refined while building GRA-21: a single `scope` key
to filter on, and the reserved prefix, were not in the first draft.)

**Ports take a `Scope`, as a required first argument.** `Scope(tenant, namespace=None)` is a small immutable value
object. Every method of `ObjectStore`, `DocumentStore`, `EntityStore` and `GraphStore` (24 today) takes one, with no
default, so a call without a scope fails type checking. A bound view such as `store.for_tenant(t)` was rejected: it
makes a missing tenant a runtime matter instead of a static one. The namespace rides in the same object so that
GRA-45 adds API plumbing and tests, not a second pass over the ports and not a second change to ids.

**The default scope keeps today's ids.** For any other scope the document id hashes `scope|bucket/key` and the
entity id hashes `el:scope:type:name`. For the default scope (the default tenant, no namespace) both functions hash
the existing strings, so stored data, the bench gate floors and every committed fixture stay valid and nothing is
migrated or re-ingested. A test pins the ids a default-scope ingest produces today. Entity linking blocks and ranks
candidates inside the scope only, so the same name in two scopes stays two entities.

**With `REQUIRE_AUTH` off, everything runs as the default tenant.** With it on (GRA-23) the tenant comes from the API key and no
body names one. It is off by default so existing local setups keep working; a hosted service must turn it on. The namespace travels in the request body and is the first scope
field a caller can set.

Operators export and delete a tenant with `graph-rag-admin` (GRA-59). The four stores each list the scopes a tenant holds (`scopes(tenant)`), then the existing per-scope calls do the work, and a delete ends by checking that no store still lists the tenant. The default tenant is refused: records from before tenancy carry no tenant field.

Keys (GRA-22) are `grk_<key_id>_<token>`; only the SHA-256 is stored, in a JSON file (`API_KEYS_PATH`) that `graph-rag-keys` manages. A file suits one service instance; several instances need a shared store, which is not built.

## Consequences

- Isolation holds only if every query carries the filter. The guards are the required argument (mypy lists every
  missed call site) and contract tests that run two scopes against the fakes and the real adapters. The MinIO
  contract tests do not run in CI today, so the object-store change is covered by the fake and unit tests in CI
  and by a local container run.
- The default-scope rule is one special case in two functions.
- Neo4j needs an index on `Entity.tenant` so counts and listings do not scan; the demo graphs are small and this is
  not measured.
- Delivery is one PR per card in this order: GRA-19 (this ADR), GRA-20 (ids), GRA-21 (ports and adapters), GRA-45
  (namespace). GRA-24 later widens the cross-tenant suite; each earlier PR carries its own two-scope tests.

## Alternatives considered

- **An index pair and a database per tenant**: the strongest isolation, since a forgotten filter cannot leak. It
  costs more as tenants multiply: shards and mappings per sign-up in Elasticsearch, a bucket per tenant in MinIO, and
  in Neo4j one database each. The compose file runs `neo4j:5.20`, Community Edition, which as I read Neo4j's
  documentation allows a single user database; more need Enterprise. This was not tested here.
- **A bare `tenant: str` argument now, namespace later**: smaller first diff, but a second pass over the same methods
  and call sites, and ids hashed twice.
- **Hash the scope into every id, default included**: a cleaner formula, but every stored document and entity must
  be re-ingested and every bench number re-baselined.

## Not decided here

- The infrastructure spike (GRA-54, consolidating the four services). GRA-19 asks for the choice to be informed by
  it; it has not been done, so this choice rests on the Neo4j edition constraint and the assumption of many small
  tenants. It is also the cheaper choice to reverse, since ids and filters change and no infrastructure does.
- Key format and storage (GRA-22), auth on the endpoints (GRA-23), usage metering (GRA-58), and tenant export and
  deletion (GRA-59).
