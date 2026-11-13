# Threat model and security checklist

Written 2026-11-08 for GRA-60, from the code as it stands after the tenancy work (ADR-0015, GRA-19 to GRA-24, GRA-59).
It covers the service a hosted deployment would run: an HTTP API with API keys in front of MinIO, Elasticsearch, Neo4j
and the trigger queue (Valkey, or Kafka), with LLM calls during ingest and optional synthesis. It is a review of this repo, not a penetration test, and
nobody outside the project has reviewed it. Each row says what protects against the threat today, and what does not.

**Status key:** Mitigated (a control exists and a test shows it), Partial (a control exists with a stated gap), Open (nothing
yet). Open and Partial rows name the card that tracks them.

## What is protected, and from whom

| Asset | Who may see or change it | Attackers considered |
|---|---|---|
| A tenant's documents, entities, facts and raw files | The holder of that tenant's API key | Another tenant; someone with no key; someone who can reach the backing stores or Kafka |
| API keys | The tenant that was shown the key once | Someone who reads the key file or the logs |
| The LLM provider key and the stores' credentials | The operator | Anyone who reads the repo, the image or the compose file |
| Availability | Everyone | A tenant, or anyone with a key, sending too much |

Out of scope: the hosting platform, TLS termination, the LLM provider's own handling of prompts, and denial of service at the
network level.

## 1. Authentication

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| A caller with no key reads or writes memory | Mitigated when `REQUIRE_AUTH=true` | 401 on every endpoint but `/health` (`tests/unit/test_auth_endpoints.py`). With auth off, the service refuses to start on any address but loopback unless `ALLOW_OPEN_BIND=true` (`graph_rag/bind.py`, `tests/unit/test_bind.py`, GRA-136) | Auth is still **off by default**; the check catches a forgotten flag only if the bind is reachable. The local compose stack sets `ALLOW_OPEN_BIND` on purpose, so copying it to a host defeats the check |
| A stolen or leaked key is used | Partial | Keys can be revoked and the revoke applies on the next request (`graph-rag-keys revoke`) | No expiry, no per-key scopes, no way to tell a tenant their key was used from a new place |
| A leaked key mints keys, or locks the owner out (GRA-143) | Partial | `/v1/keys` lets any live key of a tenant list, create and revoke that tenant's keys, so a leaked key can already reach all of that tenant's data; what it must not do is create keys without limit or cut the owner off. Active keys are capped (`MAX_KEYS_PER_TENANT`, 10), creation is rate limited per tenant (`KEY_CREATE_LIMIT_PER_MINUTE`, 5), a request cannot revoke the key it used, and the secret is returned once with `Cache-Control: no-store` and never logged (`tests/unit/test_key_management.py`). The key file is locked (`flock`) so the API and `graph-rag-keys` do not lose each other's writes, and it is read on every request, so a revoke applies at once | A leaked key can mint up to the cap and revoke the owner's other keys, so the owner must use the CLI to recover. No audit log and no `last_used`. Limits live in the process |
| A leaked key merges entities (GRA-144) | Partial | `POST /v1/entities/{id}/merge` lets any live key of a tenant fold that tenant's entities together, which rewrites its graph, its documents' links and its entity names. It cannot reach another tenant's entities (scope-first stores; an id from another tenant is a `404`), the list is capped at 10 ids per call, and every merge is stored first as a record holding each dropped entity, its edges and its document links (`graph_rag/merge_records.py`), so the damage can be reversed from the record. Records are exported and deleted with the tenant (`graph-rag-admin`). Same-type only (`409` otherwise) | A wrong merge is undone with the split below. A merge is not atomic across Elasticsearch, Neo4j and the documents; a crash leaves a `pending` record that the next merge touching those ids finishes. No audit log beyond the record, and the lock that serialises merges lives in one process |
| A leaked key splits merged entities (GRA-145) | Partial | `POST /v1/entities/{id}/split` lets any live key of a tenant put entities back that one of its recorded merges folded in. It can only restore what a merge record lists (no merge record, no split: `409`), only inside the tenant (an id from another tenant is a `404`), at most 10 ids per call, and it checks before writing that a restored id is free and that its neighbours still exist. It undoes the effect of a merge, so a key that can merge can also undo its own damage, and one that can split can undo a legitimate merge | There is no audit log beyond the record, which keeps `split_ids` and `split_at` but not who split. A split is not atomic across Elasticsearch, Neo4j and the documents; a crash leaves a `splitting` record that a repeat of the split (or a later merge touching it) finishes, and the lock lives in one process |
| Guessing a key | Mitigated | 256 random bits in the token; only the SHA-256 is stored; the comparison is constant time and a miss costs the same as a wrong token (`graph_rag/auth.py`) | |
| Brute-forcing keys or flooding with bad ones | Partial | A key is 256 random bits, so guessing is not feasible. Failed requests are counted per client address (`FAILED_KEY_LIMIT_PER_MINUTE`, default 20); past that, bad keys get 429 instead of 401. Good keys are checked first and never throttled by it | Counts live in the process, so several instances each allow the full amount. Behind a proxy every client shares one address, so the throttle is global to bad keys. It limits noise, not the cost of a hash check |
| Reading the key file | Partial | Mode 0600, only hashes inside | A hash alone does not open the API, but the file also names every tenant. One file per instance; several instances need a shared store. No card yet; GRA-31 and GRA-57 (staging) will force it |
| A key for the `default` tenant opens data it should not | Mitigated (GRA-134) | `graph-rag-keys create default` is refused, and a `default` key already in a key file no longer verifies (`tests/unit/test_auth.py`) | |
| Key shown in logs or shell history | Partial | The CLI and MCP server read the key only from `GRAPH_RAG_API_KEY`, never a flag; `graph-rag-keys create` prints it once | The key is in the environment of the process and in `claude mcp add --env`, which writes it to the client's config |
| The web UI's files are public (GRA-53) | Accepted | `/` serves static files with no auth: they hold no data and no key. The API key lives in browser memory only and every API route still checks it (`tests/unit/test_ui_static.py`). UI responses carry a Content-Security-Policy (own scripts, styles and fonts only (IBM Plex is bundled); `frame-ancestors 'none'`), `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`; API responses do not | The UI makes no request to any outside host. The headers are not set by a reverse proxy in front, which may add its own |
| `/docs` and `/openapi.json` are public | Open, low | | They describe the API only; no data. Left open, no card |

## 2. Tenant isolation

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| Tenant B reads, queries or deletes tenant A's memory through the API | Mitigated | The tenant comes only from the key, never the body; every store call takes a scope and filters on it (ADR-0015). B's attempts to recall, query, forget by source id, forget by document id and delete all find nothing or a 404, on the fakes and on real MinIO, Elasticsearch and Neo4j (GRA-24) | |
| A tenant spells another's location through a key or id it controls | Mitigated | Object keys starting with `_scopes/` are rejected on `/ingest` and `/v1/remember`; tenant and namespace cannot contain `/` or `\|`; ids hash the scope | |
| Same text from two tenants collides | Mitigated | Document and entity ids include the scope (`tests/unit/test_scope_ids.py`) | |
| A bug in one store adapter ignores the scope | Partial | One shared isolation check runs on every adapter (`tests/isolation.py`) | A new adapter or a new query has to be added to it by hand. Cost of the Neo4j scope filter on large graphs is unmeasured |
| Shared LLM response cache leaks across tenants | Mitigated | Cached responses are kept per tenant: the default tenant keeps the cache root and any other tenant gets its own directory (named by a hash of the tenant). Ingest and synthesis run inside `acting_for(tenant)`, which the client reads. `graph-rag-admin delete` removes the tenant's directory. `tests/unit/test_llm_cache_tenancy.py` shows identical text in two tenants does not share an entry | Identical text in two tenants now costs two provider calls. The cache is per tenant, not per namespace. A new call site that reaches the LLM outside `acting_for` would use the default tenant's cache, so add it there. Entries are still plain files on disk |
| Someone with access to the stores or the trigger queue bypasses the API | Partial (GRA-138) | `docker-compose.hosted.yml` publishes no store port, puts the stores on an internal network only `app` joins, and turns Elasticsearch security on; checked live with the four stores up: nothing answered from the host, Elasticsearch gave 401 without or with a wrong password and 200 with the right one, and a store had no route to the internet. **Triggers:** the default queue, Valkey, requires `VALKEY_PASSWORD` (the service refuses to start with auth on and none set) and sits on the internal network. **Only the `kafka` backend has no login** (`--profile kafka`), and it is safe only while `app` and its workers are its clients. A forged trigger names a tenant and an object key, and processing reads that tenant's own stored object, so it can only re-run documents the tenant already stored (cost, not injection; `tests/unit/test_hosted_stack.py`) | The local `docker-compose.yml` is still open on the host. Anyone who can reach the hosted Valkey, Kafka or MinIO can still write there directly. MinIO and Neo4j traffic is plain, and Elasticsearch auth is over HTTP, inside the network. A shared or external Kafka needs SASL or signed triggers first. No audit log |
| Data of a departed tenant remains | Mitigated | `graph-rag-admin delete` removes it from all four stores and checks none still lists it (GRA-59) | A trigger already queued in Kafka can recreate records; no audit log |

## 3. Ingested text and the LLM (prompt injection)

Documents are untrusted: a tenant, or whoever wrote a document a tenant ingests, controls the text that goes into prompts.

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| A document tells the extraction model to emit false or harmful facts | Partial | The model only returns triples; ones naming an id outside this document's linked entities, or a sentence index out of range, are dropped; predicates map to a closed set (`graph_rag/stages/kg_build.py`) | A document can still make the model state a false fact about its own entities. That fact is stored with its source sentence, so it is traceable and `forget` removes it, but it is believed. Tested against a model that obeys the injection (`tests/unit/test_prompt_injection.py`): ids outside the document, a sentence that does not exist and an instruction in the predicate are all handled as data. Whether a real model obeys is not measured |
| An injected document reaches another tenant's data | Mitigated | The prompt contains only the document's own text and entities; retrieval for synthesis is scoped to the caller's tenant, so another tenant's text never enters the prompt. Tests ingest a victim's and an attacker's document and check neither prompt holds the other's text, and that edges aimed at the victim's entity ids are dropped | |
| An injected document steers the synthesized answer | Partial | Synthesis is off by default (`synthesize` false, no LLM call); when on, the prompt is limited to retrieved evidence | Evidence from a malicious document within the same tenant can say "ignore the question". The grounded-answer instruction is not enforced by a check. Accepted for now and pinned by a test: the injected sentence reaches the synthesis prompt unfiltered and the model's reply comes back as `prose`; the retrieved evidence, answer and ranking do not depend on it. Default recall makes no model call |
| The model call leaks the key or runs tools | Mitigated | The model has no tools and sees no secrets; its output is parsed as data | |
| A very large document drives LLM cost | Partial | `/v1/remember` caps text at `max_remember_bytes` (1 MB) and `/ingest` at `max_upload_bytes` (10 MB, 413 above it, nothing stored or queued; `tests/unit/test_ingest_cap.py`) | A cap bounds one request, not a tenant's total: many in-limit uploads still cost. Card: GRA-137 and GRA-58 |

The red-team set is `tests/unit/test_prompt_injection.py`, in the fast suite and so in CI (GRA-140). It uses a stand-in model
that does exactly what an injected document asks, so it checks what the pipeline does with hostile output and what it sends the
model, not whether a real model resists. Accepted behaviour it pins:

- A false fact about the document's own entities is stored and believed, with its source sentence, and removed by forgetting the document.
- Document sentences reach the synthesis prompt as evidence with no filter, and the reply is returned as `prose` unchecked.
- A document can make the linker merge entities in its own tenant: if its coreference cluster names two forms that already belong to two entities of one type, they become one (GRA-30). Merges are not undone by forgetting the document, and there is no split yet (GRA-51).
- A document's text can appear in an edge's raw predicate and in the sentences recall returns, so a client that shows them must treat them as untrusted text.

Not covered: a real model's behaviour, and a model that returns malformed or oversized output beyond what the schema check rejects.

## 4. Secrets

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| Secrets committed to the repo | Mitigated | gitleaks runs in CI (`secret-scan`) on every PR | It scans for patterns; a weak default password is not a pattern |
| Default credentials reach a deployment | Mitigated for the hosted stack (GRA-138) | `docker-compose.hosted.yml` has no defaults and stops when one is unset; `app` with `REQUIRE_AUTH` on exits if a MinIO key, the Neo4j password or the (unset) Elasticsearch password is still the local-dev value, unless `ALLOW_DEFAULT_CREDENTIALS=true` (`tests/unit/test_hosted_stack.py`) | A weak but non-default password passes; nothing checks strength. The local compose and `Settings` still default to `minioadmin` and `devpassword` |
| The LLM provider key leaks | Partial | Read from the environment (`Settings`); I found no log line that prints it and no key baked into the Dockerfile | Whoever runs a container can read it. With `REQUIRE_TENANT_LLM_KEY=true` (GRA-25, off by default) each named tenant's calls use a key it stored itself, encrypted with Fernet under `TENANT_LLM_KEY_SECRET` in a 0600 file, and the deployment key is never used for them; `tests/unit/test_tenant_llm_keys.py` shows the key absent from logs, responses and command output. The master key sits in the environment next to the other secrets and decrypts every tenant's key; the file is per instance, like the API key file; a key is plaintext in process memory while a call runs |
| Questions and document names in logs | Mitigated | At info level and above nothing logs question text, answers, source names or object keys; a test drives ingest, query, recall and a failed document and checks none appears ([LOG-POLICY.md](LOG-POLICY.md)) | A traceback from `logger.exception` can quote a library's input; debug level is not covered; the web server's access log holds document ids in paths |

## 5. Availability

| Threat | Status | Gap, and the card |
|---|---|---|
| One tenant exhausts the service | Partial | Each key has a request budget (`RATE_LIMIT_PER_MINUTE`, default 600, a burst of that many then a refill), so one key cannot starve another's requests; `/ingest` is capped per request | A request budget is not a cost budget: one `/ingest` can still cost an LLM bill, and ingest runs in the API process unless `INGEST_MODE=publish` moves it to workers (GRA-31). Counts are metered (GRA-58), but there is no quota. Per instance, as above. Not applied with `REQUIRE_AUTH` off |
| Slow or huge recall | Partial | `token_budget` bounds what is returned, not the work of finding it; I found no query timeout in the API or retriever |

## Follow-ups filed

These cards were created from this review. Each is small; GRA-134 and GRA-135 are the ones to do before any hosted use.

1. GRA-134 (done): refuse API keys for the `default` tenant.
2. GRA-135 (done): cap `/ingest` upload size.
3. GRA-136 (done): refuse to start on a non-local bind with auth off.
4. ~~GRA-137: rate limit requests and failed keys.~~ Done.
5. GRA-138 (done): lock down the backends (no host-published store ports, no default credentials, authenticated triggers).
6. ~~GRA-139: scope the LLM response cache by tenant, or turn it off when hosted.~~ Done.
7. ~~GRA-140: prompt-injection tests through ingested documents.~~ Done.
8. ~~GRA-141: a log policy, with question text and source names redacted.~~ Done.

Existing cards that also close rows above: GRA-58 (metering, done behind `METER_USAGE`; counts, not limits), GRA-31 (workers separate from the API), GRA-25 (tenant LLM keys, done behind `REQUIRE_TENANT_LLM_KEY`).

## Checklist before a hosted deployment

- [ ] `REQUIRE_AUTH=true` and `ALLOW_OPEN_BIND` unset (the service will not start otherwise), and the key file on a mounted volume with mode 0600 (one instance only)
- [ ] No data was written with auth off, or it has been moved
- [ ] Run `docker-compose.hosted.yml` (or equivalent): the stores are on an internal network, none uses a default credential, and the trigger queue has no client but the service
- [ ] TLS terminates in front of the API
- [ ] Upload size is capped (`MAX_UPLOAD_BYTES`, default 10 MB) and request rates are limited (`RATE_LIMIT_PER_MINUTE`, per instance; behind several instances set it to the share each should allow)
- [ ] The LLM cache directory is on storage only the service reads (entries are per tenant but are plain files)
- [ ] Logs are handled per [LOG-POLICY.md](LOG-POLICY.md): info level carries no content; do not run at debug
- [ ] `graph-rag-admin delete` has been tried on a staging tenant
