# ADR-0014: The agent-facing API is four verbs with one error shape

- Status: Accepted (2026-11-05)
- Related: ADR-0001 (thin ingest), ADR-0007 (query), GRA-35; contract in `docs/api/agent-api.openapi.yaml`

## Context

The product direction is memory for agents and developers, reached over HTTP, MCP and a CLI. The endpoints
that exist were built for a demo: `POST /ingest` takes a multipart file, `POST /query` returns the whole retrieval
result, `DELETE /documents/{id}` forgets by an id only the ingest response reveals, and errors are FastAPI's
`{"detail": ...}`. Three clients on three shapes would drift. One contract, written first, keeps them together.

## Decision

**Four verbs, each one tool name.** `remember`, `recall`, `forget`, `lookup`. The OpenAPI `operationId`, the MCP
tool name and the CLI subcommand are the same word, so an agent's prompt, a script and a request log all agree.

**One JSON request body per call, under `/v1`.** All four are `POST /v1/<verb>` with a JSON body, including
`lookup` and `forget`. A resource-style `DELETE /v1/sources/{id}` would put a source URL in a path (escaping) and
split the surface into two styles; an MCP tool call is a name plus one JSON object anyway. `/health` stays where it
is, unversioned. Additive changes (a new optional field) do not bump the version; `/v2` is for breaking ones.

**Two ids for a source, one rule.** The caller names a source with `source_id` (a URL, a path, a note id). The
server derives `document_id` from it deterministically (ADR-0001). `forget` takes either, so a caller who kept
only the source id is not stuck. `remember` without a `source_id` derives one from the text, so identical text is
stored once.

**Writes are safe to repeat.** `remember` with the same source and the same text does nothing and says
`status: unchanged` (unless its last job failed: then it is queued again and says `accepted`); with new text it replaces the source (until updates are built it ingests alongside). `forget` of
a source that is gone is `not_found`, which a caller can treat as done.

**Writes are asynchronous, reads are not.** `remember` returns `202` once the text is stored and queued. The
pipeline runs afterwards, so a `recall` immediately after may not see it. This is stated in the contract; a job-status
call (GRA-32) is the way to wait.

**Recall returns its evidence.** The answer comes with the subgraph, the ranked entities, the ids of the entities the question matched (`matched_entity_ids`), and the supporting
sentences, each with its source document and character offsets, so an agent can cite and a person can check. The
default path makes no LLM call (`synthesize: false`).

**One error shape.** Every failure is `{"error": {"code", "message", "details?"}}`.

| `code` | HTTP | When |
|---|---|---|
| `invalid_request` | 400 | Malformed body, a missing or conflicting field, a value out of range (a token budget of 0) |
| `unauthorized` | 401 | The API key is missing or wrong |
| `not_found` | 404 | No source or entity matches |
| `payload_too_large` | 413 | The text is over the size limit |
| `unavailable` | 503 | A part the call needs (the retriever, the eraser, a store) is not running or not configured. `remember` answers it with `details.reason` `llm_not_configured` when no LLM provider key or endpoint is set (GRA-154); nothing is stored or queued. The original `POST /ingest` route refuses the same case with a `503` whose body is `{"detail": ...}`, not this error shape |
| `internal` | 500 | Anything else; the message does not leak internals |

Clients branch on `code`; `message` is one sentence for a person that says what to change. The code set is closed:
adding one is a contract change.

**Tenancy comes from the key.** A request carries an API key as a bearer token; the key names the tenant, so no
body has a tenant id. `namespace` (optional) partitions memory inside a tenant.

**Planned fields are in the contract now.** A field is
specified and marked `x-status: planned` with the card that builds it, and ignored by the server until then. `namespace`, `token_budget` and `as_of` were planned here and are built (GRA-45, ADR-0015; GRA-38; GRA-44). Specifying
them first fixes their names before three clients each invent one.

## Mapping onto what exists

| Verb | Wraps | `/v1` shape |
|---|---|---|
| `remember` | `POST /ingest` (multipart file, `source_url`, `?force`) | Text is the upload and `source_id` the `source_url`; `source_id` is derived from the text when omitted. Served (GRA-132). |
| `recall` | `POST /query` | Same retrieval and body. `token_budget` (with `truncated` and `tokens_used`) is served (GRA-38) and `namespace` (GRA-45); `as_of` (GRA-44) keeps the edges that held on that day. Served (GRA-132). |
| `forget` | `DELETE /documents/{doc_id}` | Takes `source_id` or `document_id`; the facade derives the id from the source. Served (GRA-132). |
| `lookup` | none | Built over `GraphStore.get_node` / `get_node_edges` and name blocking in `EntityStore`; takes a name (or alias) or an entity id, and an optional `namespace`; no match is `404`. Served (GRA-37). |

### Entity browsing (GRA-142)

Two read-only routes beside the verbs feed the graph explorer; they are not agent tools (no MCP or CLI mapping).
Both take the optional `namespace` query parameter, run in the key's tenant, and answer in the same error shape.

- `GET /v1/entities?q=&type=&limit=&cursor=&namespace=`: entities whose name or any alias contains `q` (case and
  punctuation ignored; no `q` lists all), optionally of one `type`, newest first (`first_seen` descending, then
  `canonical_id`). `limit` is 1 to 100 (default 20). `next_cursor` is an opaque keyset cursor to pass back as
  `cursor`, and `null` on the last page; a cursor this API did not issue is `invalid_request`.
- `GET /v1/entities/{entity_id}?neighbours_limit=&sources_limit=&namespace=`: the entity (`canonical_id` is its
  stable id), `aliases`, `first_seen`, `neighbours` (outgoing facts, then incoming, each direction cut at
  `neighbours_limit`, 1 to 100, default 25, with `outgoing_total` and `incoming_total`) and `sources` (documents that
  mention it, most mentions first, cut at `sources_limit`, with `sources_total`). A source is `{document_id, name,
  mentions}`: the source id and a count, never document text. An id the scope does not hold, including another
  tenant's, is `404`. `merged_from` lists the entities merged into this one (GRA-144, below); it is `[]` when none were.

`first_seen` is stamped by the entity store on an id's first write and kept through later ones. An entity stored
before it was kept has `first_seen: null` and sorts last.

`GET /v1/me` is not a verb. It answers who the key is: the tenant, the `namespace` the request names, and the key's last four characters (never the key), so a UI can sign in with a key and show the tenant (GRA-47). A bad or missing key is `401`.

### Addendum: key management (GRA-143)

`GET /v1/keys`, `POST /v1/keys` and `DELETE /v1/keys/{key_id}` (`graph_rag/keys_api.py`, in `docs/openapi.json`, not in
the hand-written contract, like `/v1/me`) list, create and revoke the calling tenant's API keys. Shapes:
`{"keys": [{key_id, label, created_at, key_suffix, status: "active"|"revoked", revoked_at, current}]}`;
create takes `{"label"}` (at most 64 characters, whitespace collapsed, no control characters) and answers `201` with
that shape plus `secret`, with `Cache-Control: no-store`; revoke answers `200` with the key, and revoking twice is
harmless. Choices: any live key of the tenant may manage keys (no admin flag; per-key scopes are out of scope), bounded
by a cap on active keys, a create rate limit and a refusal (`400`, `details.reason` `own_key`) to revoke the key in
use. Another tenant's key is `404`, the same as an unknown id. With auth off the routes answer `400`
(`details.reason` `auth_disabled`) and never act on the default tenant. There is no `last_used`: it would cost a write
per request. The `key_id` is random and never derived from the secret; the last four characters of the secret are
stored with the record (keys made before this have none, so `key_suffix` is `null`).

### Addendum: merging entities (GRA-144)

`POST /v1/entities/{keep_id}/merge` with `{"merge_ids": [...], "namespace"?}` folds entities into `keep_id`
(`graph_rag/curation_api.py`, in `docs/openapi.json`, not in the hand-written contract). It is a curation route, not an
agent tool. `merge_ids` holds 1 to 10 distinct ids and never `keep_id` (`400 invalid_request`). It answers `200`:
`{"merge_id", "merged": [ids folded by this call], "already_merged": [ids an earlier call folded in], "entity": <the
survivor, EntitySummary>}`.

- **Errors.** An unknown id, the kept id included, or an id of another tenant or namespace is `404 not_found`, and nothing
  is written. A new error code, `409 conflict` (`details.reason` `type_mismatch`, `details.entity_ids`), refuses entities
  whose type differs from the kept one: GRA-30's linker only unifies entities of one type (blocking filters on type), so
  this does the same. A request may mix new ids and ids already merged into `keep_id`; an id already merged into a
  different entity is `404` with `details.merged_into`.
- **Idempotent.** Repeating a merge, or sending ids an earlier merge already folded in, is a `200` no-op that returns the
  earlier `merge_id` with those ids in `already_merged`.
- **What follows what**, as in GRA-30 (the same `GraphStore.merge_nodes`): the survivor keeps its name, vector and
  `first_seen` and gains each dropped name and alias (deduplicated through `normalize_name`); the dropped edges move onto
  it with provenance and validity untouched (a loop is dropped, a repeat is not duplicated); every document's
  `el_result` link to a dropped id names the survivor; the dropped entities leave the entity store and the graph.
- **After the merge.** `GET /v1/entities/{dropped_id}` is `404` with `details.merged_into` (a hint, not a redirect: the
  simpler choice, and one status for "not here"). A later document that names a dropped entity links to the survivor,
  because the dropped name is now one of its aliases and linking blocks on type and name-or-alias
  (`tests/unit/test_entity_merge.py`). If a split later removes the alias, that stops being true; that is the split's
  call. Merging B into C after A into B leaves two records; C lists only B in `merged_from`, B's record lists A.
- **The record** (`MergeRecordStore`, per scope, in Elasticsearch next to the other stores; the port has `put`, `get`,
  `for_entity`, `containing`, `all`, `scopes`, `delete`). Shape, `MergeRecord`:
  `{merge_id, keep_id, status: "pending"|"done"|"split", created_at, completed_at, keep_aliases_before: [str],
  dropped: [DroppedEntity]}` with `DroppedEntity = {canonical_id, name, type, aliases, first_seen, vector, edges:
  [Triple as stored], document_links: [{document_id, link: EntityLink as stored}]}`. A split restores from it alone:
  upsert each dropped entity (vector included) in both stores, write its `edges`, set each `document_links` link back to
  the dropped id (match on `document_id`, `mention_text` and `canonical_id == keep_id`), take the dropped names out of the
  survivor's aliases (keeping those in `keep_aliases_before` and any added since), and set `status` to `split` (GRA-145,
  below; `EntityStore.upsert` takes an optional `first_seen` for that).
  `GET /v1/entities/{id}` returns the finished records as `merged_from: [{canonical_id, name, type, aliases, merge_id,
  merged_at}]` (no vector, no edges). Records go with the tenant in `graph-rag-admin` export and delete.
- **Consistency.** There is no transaction across the entity store, the graph and the documents, so the record is written
  first as `pending` and the steps run in an order that is safe to repeat: survivor aliases, graph node and
  `merge_nodes`, document links, deleting the dropped entities, and last marking the record `done`. A crash in between
  leaves a `pending` record, some steps applied, and (until the delete step) the dropped entity still visible in
  entity search with its edges already on the survivor. The next merge request that names the survivor or any of the
  dropped ids finishes every `pending` record it finds before doing anything else, so repeating the same request heals
  it; the response then lists the ids under `already_merged`. Nothing heals it by itself, and a merge that is never
  retried stays half-done. Merges are serialised by a lock in the API process only; two API processes merging the same
  entity at once is not guarded.

Today's endpoints keep working. The `/v1` facade is a thin layer that renames and reshapes; it adds no pipeline
logic. Errors at the facade are converted from the existing ones (`404` and `503` already carry the meaning).

### Addendum: splitting a merged entity (GRA-145)

`POST /v1/entities/{keep_id}/split` with `{"merge_id"?, "entity_ids"?, "namespace"?}` undoes a recorded merge
(`graph_rag/curation_api.py`, `graph_rag/splitting.py`; in `docs/openapi.json`, not in the hand-written contract). It
restores only what a merge record lists, so it cannot split an entity that was never merged. `merge_id` names the record
(from a merge response or from `merged_from`); left out, it is the only record of `keep_id` that still has entities to
split out (none: `409`; several: `400 invalid_request` with `details.merge_ids`). `entity_ids` (1 to 10, distinct) picks
some of the record's dropped entities; left out, all that are still merged. It answers `200`:
`{"merge_id", "split": [EntitySummary put back by this call], "already_split": [ids an earlier call put back], "entity":
<the survivor now, EntitySummary>}`. `first_seen` of a restored entity is the one it had before the merge.

- **Errors.** Unknown or other-tenant or other-namespace `keep_id`, or an unknown `merge_id`: `404 not_found`, nothing
  written. `409 conflict` with `details.reason`: `no_merge_record` (the entity has no merge to split, the record is for a
  different survivor, or `entity_ids` holds an id the merge did not fold in; `details.entity_ids` for the last),
  `merge_pending` (the merge never finished; repeat the merge request first), `survivor_merged` (`keep_id` was itself
  merged into another entity, `details.merged_into` and `details.merge_id`; split it out of that one first),
  `entity_exists` (a restored id has been taken by another entity since), `neighbour_merged` (an edge to restore ends at
  an entity that has been merged away since; split that one out first). The checks run before anything is written.
- **Idempotent.** Splitting ids a record has already split is a `200` no-op with those ids in `already_split`, as a
  repeated merge is (a `409` would make a retry after a lost response look like a failure).
- **What goes back.** Entities: each dropped entity is put back in the entity store (vector, aliases, recorded
  `first_seen` through `EntityStore.upsert(..., first_seen=)`) and in the graph, and its names that only it brought leave
  the survivor's aliases. A name stays if the survivor had it before the merge (`keep_aliases_before`), if it is the
  survivor's own name, or if another still-merged entity brought it too. A name the survivor also earned from a later
  document cannot be told apart from the merged one and goes too. Edges: each edge the record lists for a restored entity
  returns to its original endpoints, and the copy the merge put on the survivor is removed
  (`GraphStore.split_nodes`). Where the other end is another dropped entity that is staying merged, the edge points at the
  survivor. An edge that became a loop on the survivor comes back as the edge it was. A repeat the merge did not duplicate
  (the survivor had an identical edge) is restored as the dropped entity's own edge while the survivor keeps its copy,
  because the merge recorded the survivor's colliding edges (`shadowed_edges`). Links: each recorded link goes back to
  the dropped id, matched on the document and on every field but the id among the links that now name the survivor; two
  links identical but for the id cannot be told apart, so the first goes. A document deleted or re-ingested since is
  skipped.
- **What stays.** Edges, links and aliases created after the merge stay with the survivor: only what the record lists
  moves. A later document that linked to the survivor through a dropped name keeps that link. From the split on, a
  document naming the restored entity links to it by its own name, since the survivor no longer has it as an alias
  (`tests/unit/test_entity_split.py`).
- **Entity detail.** `merged_from` lists only the records' entities that are still merged. A restored id answers
  `GET /v1/entities/{id}` `200` and no longer gives `merged_into`.
- **Order and chains.** A record is split by the survivor it merged into. A merged into B and B into C: A cannot be split
  out of B while B is dropped (`survivor_merged`); split B out of C, then A out of B. The other order needs no rule. The
  same ordering holds for a neighbour (`neighbour_merged`). A restored entity that is merged elsewhere again gets a new
  record; the old one keeps it listed in `split_ids`, so repeating the old split is a no-op.
- **The record.** `MergeRecord` gained `split_ids` (dropped ids restored), `splitting_ids` (the ids of the split in
  flight), `split_at`, `shadowed_edges`, and the status `splitting`. Status is `done` while some dropped entities are
  still merged, `splitting` during a split and `split` once every dropped entity is back. Records written before this
  change have no `shadowed_edges`, so splitting one removes the survivor's identical edge if there was one.
- **Consistency.** As for the merge, with no shared transaction: the record is marked `splitting` first, then the graph
  (`split_nodes`: nodes, then the edges to add, then the edges to remove), the documents' links, the restored entities,
  the survivor's aliases, and last the record is marked split. Every step is safe to run again. A crash leaves a
  `splitting` record and some steps applied: until the restored entity is written it is missing from entity search while
  its edges and links are already back, and after that and before the alias step both it and the survivor answer to its
  name. Repeating the split request, or any merge that names the survivor or a dropped id of the record, finishes
  it first. Nothing heals it by itself. The lock that serialises merges and splits lives in one process only.
- **Not in scope.** Splitting an entity with no merge record, or splitting by moving arbitrary edges.

## Consequences

- The contract is written by hand and not generated from the code, because it also describes what is not built.
  `tests/unit/test_agent_api_contract.py` keeps it honest: the file is valid OpenAPI and every `$ref` resolves
  (the validator does not check that); every operation marked `existing` maps onto an endpoint the app serves;
  real `recall` and `forget` responses fit their schemas; and a field marked `planned` that the code starts to accept
  fails the test until its status is changed.
- The MCP server (GRA-36, GRA-37), the CLI (GRA-39) and the SDK (GRA-40) are built against this file, not against the
  Python models.
- `openapi-spec-validator` joins the dev extra.
- Rate limits are `429` with code `rate_limited` and a `Retry-After` header (GRA-137).
- Not decided here: pagination of `lookup` neighbours beyond `limit`, and streaming recall.

## Alternatives considered

- **Resource-style REST** (`POST /memories`, `GET /entities/{id}`, `DELETE /memories/{id}`): familiar, but two
  styles on one small surface, source URLs in paths, and no tool names to share with MCP.
- **Generate the contract from FastAPI**: always true, but it cannot describe planned work, and it would tie the
  public names to internal model names.
- **Extend today's endpoints in place**: fastest, but it leaves multipart ingest and `detail` errors in front of
  every agent.
