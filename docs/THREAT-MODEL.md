# Threat model and security checklist

Written 2026-11-08 for GRA-60, from the code as it stands after the tenancy work (ADR-0015, GRA-19 to GRA-24, GRA-59).
It covers the service a hosted deployment would run: an HTTP API with API keys in front of MinIO, Elasticsearch, Neo4j
and Kafka, with LLM calls during ingest and optional synthesis. It is a review of this repo, not a penetration test, and
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
| Guessing a key | Mitigated | 256 random bits in the token; only the SHA-256 is stored; the comparison is constant time and a miss costs the same as a wrong token (`graph_rag/auth.py`) | |
| Brute-forcing keys or flooding with bad ones | Open | None | No rate limit or lockout on failed keys. Card: GRA-137 |
| Reading the key file | Partial | Mode 0600, only hashes inside | A hash alone does not open the API, but the file also names every tenant. One file per instance; several instances need a shared store. No card yet; GRA-31 and GRA-57 (staging) will force it |
| A key for the `default` tenant opens data it should not | Mitigated (GRA-134) | `graph-rag-keys create default` is refused, and a `default` key already in a key file no longer verifies (`tests/unit/test_auth.py`) | |
| Key shown in logs or shell history | Partial | The CLI and MCP server read the key only from `GRAPH_RAG_API_KEY`, never a flag; `graph-rag-keys create` prints it once | The key is in the environment of the process and in `claude mcp add --env`, which writes it to the client's config |
| `/docs` and `/openapi.json` are public | Open, low | | They describe the API only; no data. Left open, no card |

## 2. Tenant isolation

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| Tenant B reads, queries or deletes tenant A's memory through the API | Mitigated | The tenant comes only from the key, never the body; every store call takes a scope and filters on it (ADR-0015). B's attempts to recall, query, forget by source id, forget by document id and delete all find nothing or a 404, on the fakes and on real MinIO, Elasticsearch and Neo4j (GRA-24) | |
| A tenant spells another's location through a key or id it controls | Mitigated | Object keys starting with `_scopes/` are rejected on `/ingest` and `/v1/remember`; tenant and namespace cannot contain `/` or `\|`; ids hash the scope | |
| Same text from two tenants collides | Mitigated | Document and entity ids include the scope (`tests/unit/test_scope_ids.py`) | |
| A bug in one store adapter ignores the scope | Partial | One shared isolation check runs on every adapter (`tests/isolation.py`) | A new adapter or a new query has to be added to it by hand. Cost of the Neo4j scope filter on large graphs is unmeasured |
| Shared LLM response cache leaks across tenants | **Open (found in this review)** | | The cache is keyed by a hash of the model, prompt and parameters and shared by all tenants (`.cache/llm`). Identical text from two tenants reuses one answer, so a tenant could learn that another has processed the same text (by speed or cost; I did not try it), and a poisoned entry would reach both. The cached responses, which quote document sentences, are also kept on disk. Card: GRA-139 |
| Someone with access to the stores or Kafka bypasses the API | Open | | Compose publishes Elasticsearch (9200, security off), MinIO, Kafka and Neo4j on the host with default credentials (`minioadmin`, `devpassword`). A Kafka message carries its own tenant, so anyone who can publish can write into any tenant. Card: GRA-138 |
| Data of a departed tenant remains | Mitigated | `graph-rag-admin delete` removes it from all four stores and checks none still lists it (GRA-59) | A trigger already queued in Kafka can recreate records; no audit log |

## 3. Ingested text and the LLM (prompt injection)

Documents are untrusted: a tenant, or whoever wrote a document a tenant ingests, controls the text that goes into prompts.

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| A document tells the extraction model to emit false or harmful facts | Partial | The model only returns triples; ones naming an id outside this document's linked entities, or a sentence index out of range, are dropped; predicates map to a closed set (`graph_rag/stages/kg_build.py`) | A document can still make the model state a false fact about its own entities. That fact is stored with its source sentence, so it is traceable, but it is believed. No red-team test exists |
| An injected document reaches another tenant's data | Mitigated | The prompt contains only the document's own text and entities; retrieval for synthesis is scoped to the caller's tenant, so another tenant's text never enters the prompt | |
| An injected document steers the synthesized answer | Partial | Synthesis is off by default (`synthesize` false, no LLM call); when on, the prompt is limited to retrieved evidence | Evidence from a malicious document within the same tenant can say "ignore the question". The grounded-answer instruction is not enforced by a check |
| The model call leaks the key or runs tools | Mitigated | The model has no tools and sees no secrets; its output is parsed as data | |
| A very large document drives LLM cost | Partial | `/v1/remember` caps text at `max_remember_bytes` (1 MB) and `/ingest` at `max_upload_bytes` (10 MB, 413 above it, nothing stored or queued; `tests/unit/test_ingest_cap.py`) | A cap bounds one request, not a tenant's total: many in-limit uploads still cost. Card: GRA-137 and GRA-58 |

Tracked work: a red-team fixture set (documents containing instructions) run through ingest and recall, with the bench as the
place to keep it. Card: GRA-140.

## 4. Secrets

| Threat | Status | What exists | Gap, and the card |
|---|---|---|---|
| Secrets committed to the repo | Mitigated | gitleaks runs in CI (`secret-scan`) on every PR | It scans for patterns; a weak default password is not a pattern |
| Default credentials reach a deployment | Open | `minioadmin`, `devpassword` and no Elasticsearch security are the compose and `Settings` defaults, marked local-dev | Nothing stops them being used in production. Card: GRA-138 |
| The LLM provider key leaks | Partial | Read from the environment (`Settings`); I found no log line that prints it and no key baked into the Dockerfile | Whoever runs a container can read it; tenants bringing their own key (GRA-25) would add per-tenant secrets that need storing |
| Questions and document names in logs | Open | `/query` logs the question text and synthesis logs it when on; ingest logs document ids and source names. (`/v1/recall` without synthesis logs no question.) | Questions can hold personal data. Card: GRA-141 |

## 5. Availability

| Threat | Status | Gap, and the card |
|---|---|---|
| One tenant exhausts the service | Open | No rate limit, quota or metering (GRA-58); `/ingest` is capped per request only; ingest runs in the API process (GRA-31) |
| Slow or huge recall | Partial | `token_budget` bounds what is returned, not the work of finding it; I found no query timeout in the API or retriever |

## Follow-ups filed

These cards were created from this review. Each is small; GRA-134 and GRA-135 are the ones to do before any hosted use.

1. GRA-134 (done): refuse API keys for the `default` tenant.
2. GRA-135 (done): cap `/ingest` upload size.
3. GRA-136 (done): refuse to start on a non-local bind with auth off.
4. GRA-137: rate limit requests and failed keys.
5. GRA-138: lock down the backends (no host-published store ports, no default credentials, authenticated Kafka triggers).
6. GRA-139: scope the LLM response cache by tenant, or turn it off when hosted.
7. GRA-140: prompt-injection tests through ingested documents.
8. GRA-141: a log policy, with question text and source names redacted.

Existing cards that also close rows above: GRA-58 (metering), GRA-31 (workers separate from the API), GRA-25 (tenant LLM keys).

## Checklist before a hosted deployment

- [ ] `REQUIRE_AUTH=true` and `ALLOW_OPEN_BIND` unset (the service will not start otherwise), and the key file on a mounted volume with mode 0600 (one instance only)
- [ ] No data was written with auth off, or it has been moved
- [ ] The stores are not reachable from outside the service's network, and none uses a default credential
- [ ] TLS terminates in front of the API
- [ ] Upload size is capped (`MAX_UPLOAD_BYTES`, default 10 MB) and request rates are limited (GRA-137, not built)
- [ ] The LLM cache is scoped or off
- [ ] Logs are handled as holding user questions
- [ ] `graph-rag-admin delete` has been tried on a staging tenant
