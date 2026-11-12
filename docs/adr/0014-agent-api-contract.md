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
`status: unchanged`; with new text it replaces the source (until updates are built it ingests alongside). `forget` of
a source that is gone is `not_found`, which a caller can treat as done.

**Writes are asynchronous, reads are not.** `remember` returns `202` once the text is stored and queued. The
pipeline runs afterwards, so a `recall` immediately after may not see it. This is stated in the contract; a job-status
call (GRA-32) is the way to wait.

**Recall returns its evidence.** The answer comes with the subgraph, the ranked entities and the supporting
sentences, each with its source document and character offsets, so an agent can cite and a person can check. The
default path makes no LLM call (`synthesize: false`).

**One error shape.** Every failure is `{"error": {"code", "message", "details?"}}`.

| `code` | HTTP | When |
|---|---|---|
| `invalid_request` | 400 | Malformed body, a missing or conflicting field, a value out of range (a token budget of 0) |
| `unauthorized` | 401 | The API key is missing or wrong |
| `not_found` | 404 | No source or entity matches |
| `payload_too_large` | 413 | The text is over the size limit |
| `unavailable` | 503 | A part the call needs (the retriever, the eraser, a store) is not running or not configured |
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
  tenant's, is `404`. Entities merged later (GRA-144) will add a `merged_from` list; it is not there yet.

`first_seen` is stamped by the entity store on an id's first write and kept through later ones. An entity stored
before it was kept has `first_seen: null` and sorts last.

`GET /v1/me` is not a verb. It answers who the key is: the tenant, the `namespace` the request names, and the key's last four characters (never the key), so a UI can sign in with a key and show the tenant (GRA-47). A bad or missing key is `401`.

Today's endpoints keep working. The `/v1` facade is a thin layer that renames and reshapes; it adds no pipeline
logic. Errors at the facade are converted from the existing ones (`404` and `503` already carry the meaning).

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
