# Memory and knowledge-graph layers for agents: a comparison (GRA-67)

Accessed 2026-11-14. Every external claim below links its source and was read on that date. Where I could not read a number from the source, it says "unverified". Vendor benchmark numbers are the vendor's own and use different models, judges and settings, so they cannot be ranked against each other. Pricing pages change; re-check before relying on a price.

## Summary

- The field splits in two. **Memory layers for chat agents** (Zep, Mem0, Supermemory, Letta) are scored on LoCoMo and LongMemEval. **Graph RAG over documents** (LightRAG, Microsoft GraphRAG, HippoRAG) is scored on win rates or on HotpotQA, 2WikiMultihopQA and MuSiQue. Cognee sits between the two. This project is built as document graph RAG and exposed as an agent memory API, so it overlaps both camps and is mature in neither.
- Every competitor has more users, more real-stack mileage and a managed option. This project has none of those.
- What this project has that I did not find elsewhere: a $0 deterministic query path, an offline mode with no key, a synthetic data generator with programmatic gold, and a bench that reports its own negative results.
- Published benchmark claims are contested. Zep and Mem0 disagree on each other's LoCoMo score, and the one independent multi-hop comparison I found has LightRAG far below a plain dense retriever.

## Comparison table

| | Zep / Graphiti | Mem0 | Cognee | LightRAG | This project |
|---|---|---|---|---|---|
| What it is | Zep: managed temporal context graph for agents. Graphiti: the open-source graph framework inside it ([repo](https://github.com/getzep/graphiti)) | Memory layer for agents: library, self-hosted server, cloud ([repo](https://github.com/mem0ai/mem0)) | Memory layer that builds a graph from documents, code and conversations; `remember`, `recall`, `improve`, `forget` ([repo](https://github.com/topoteretes/cognee)) | Graph RAG framework with dual-level retrieval; SDK, REST server, web UI ([repo](https://github.com/HKUDS/LightRAG)) | Document-to-graph pipeline with `/v1` `remember`, `recall`, `forget`, `lookup`, MCP server and CLI ([AGENTS.md](../../AGENTS.md)) |
| Licence | Graphiti Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 |
| Stars (2026-11-14) | Graphiti 32.8k | 68.9k | 33.3k | 40.7k | Not applicable |
| Self-host | Graphiti only: you supply the graph DB (Neo4j 5.26, FalkorDB, Neptune; Kuzu deprecated), the LLM and the embedder, and build the service yourself | Yes: library, or `docker compose` server | Yes: pip, Docker, Compose with UI and MCP profiles. The repo says the Postgres graph store is a demo and the production version is a licensed product | Yes: pip or Docker Compose. Default stores are file-based and marked "not for production"; Postgres recommended. Ships with auth off | Yes: Docker Compose (Neo4j, Elasticsearch, MinIO, Kafka, app, worker), or one process over in-memory stores for the quickstart |
| Managed cloud | Yes (Zep). Enterprise adds BYOK and BYOC | Yes (app.mem0.ai) | Yes (Cognee Cloud) | None described | No |
| Typical LLM need | Required at ingestion; OpenAI by default, structured-output support recommended | Required; default gpt-5-mini plus OpenAI embeddings | Not for ingestion or retrieval (local GLiNER and embeddings, described as demo grade); needed for answers and improvement stages | Required on every chunk for extraction; query model should be stronger | Optional. Offline path uses heuristics; real stack uses an LLM for coref and graph build |
| Query-time LLM | Not stated on the pages read | Not stated on the pages read | Needed for generated answers | Yes (query and keyword roles) | None on the default `/query` path |
| Time handling | Temporal graph with validity windows | Not stated on the pages read | Bi-temporal memory is listed as Enterprise only | Not stated | `valid_to` supersession for four single-valued predicates and `as_of` queries (`graph_rag/supersession.py`) |
| Graph availability | Core | Graph memory is on the Custom plan only, per the pricing page | Core | Core | Core |
| Published price | Free: 10k credits/month. Flex $125/month for 50k credits. Flex Plus $375/month for 200k. Overage $25 per 10k credits. Enterprise custom ([pricing](https://www.getzep.com/pricing)) | Free: 10k adds and 1k retrievals for the first 30 days, then 50k adds and 10k retrievals with your own LLM key. Custom plan: contact sales ([pricing](https://mem0.ai/pricing)) | Free (1M tokens, 1 workspace). Standard $1.00 per 1M tokens processed plus $5 per extra workspace. Enterprise BYOC by contract ([pricing](https://www.cognee.ai/pricing)) | No hosted price | None yet; see GRA-68 |

Notes on the table. Zep credits: 1 credit per episode up to 350 bytes, 1 more per additional 350 bytes; retrieval and storage are unmetered. The Mem0 pricing page's meta description mentions "Pro" plans that the body does not list, so the body is what I recorded. The Mem0 "unverified" items are the ones marked "not stated": I read the repo and pricing page, not the full docs.

## Other projects worth knowing

| Project | What it is | Source |
|---|---|---|
| Supermemory | Memory and context engine, MIT, runs locally or hosted, MCP server. Claims #1 on LongMemEval, LoCoMo and ConvoMem with 95% Recall@15 on LongMemEval; the page gives no competitor numbers behind the ranking. Hosted price not on the repo page (unverified) | [repo](https://github.com/supermemoryai/supermemory) |
| Letta (formerly MemGPT) | Platform for stateful agents, Apache-2.0, self-host or Letta Cloud. Publishes no benchmark claims on its repo page. Mem0's blog lists a Letta LoCoMo score of 74.0 with a filesystem-only setup; that is a competitor's report | [repo](https://github.com/letta-ai/letta) |
| Microsoft GraphRAG | MIT, 36.6k stars. The README says the project is largely in maintenance mode and that indexing "can be an expensive operation". It is not an officially supported Microsoft offering | [repo](https://github.com/microsoft/graphrag) |
| HippoRAG 2 | MIT research framework (Personalized PageRank over an LLM-built graph). Closest academic match to this project's retrieval idea | [repo](https://github.com/OSU-NLP-Group/HippoRAG), [paper](https://arxiv.org/abs/2502.14802) |

## Published benchmark claims

All are self-reported unless stated. None has been reproduced here.

| Claim | Source | What to know |
|---|---|---|
| Zep: 94.8% on DMR (gpt-4-turbo) vs 93.4% for MemGPT; 98.2% with gpt-4o-mini | [Zep paper, Table 1](https://arxiv.org/abs/2501.13956) | In the same table a full-conversation baseline scores 94.4% and 98.0%, so DMR barely separates memory from "put it all in the prompt". |
| Zep: LongMemEval 71.2% with gpt-4o vs 60.2% full context; 63.8% vs 55.4% with gpt-4o-mini; context 1.6k tokens vs 115k; latency 2.58 s vs 28.9 s (gpt-4o) | Zep paper, Table 2 | The abstract's "up to 18.5%" is the best case, the gpt-4o row. Zep lost to full context on single-session-assistant questions (Table 3). |
| Zep LoCoMo: corrected to 75.14% (J score), ahead of Mem0 Graph's 68.44% | [Zep blog, 2025-05-06](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) | Zep says Mem0's run of Zep had a wrong user model, wrong timestamp handling and sequential searches. Mem0's co-founder disputed Zep's number in a GitHub issue (58.44% when excluding the adversarial category). I read that only through a search summary, so treat the 58.44% as unverified. |
| Mem0 paper: 26% relative improvement over OpenAI memory on LLM-as-judge; about 91% lower p95 latency and over 90% lower token cost vs full context; graph memory about 2% higher | [Mem0 paper abstract](https://arxiv.org/abs/2504.19413) | Absolute LoCoMo scores not read (abstract only). |
| Mem0, April 2026 algorithm: LoCoMo 92.5, LongMemEval 94.4, BEAM 1M 64.1, BEAM 10M 48.6, about 7k tokens per retrieval | [Mem0 repo README](https://github.com/mem0ai/mem0), [Mem0 blog](https://mem0.ai/blog/locomo-benchmark) | The README says the scores come from the managed platform, which has "proprietary optimizations not available in the open-source SDK". Evaluation code is public (`mem0ai/memory-benchmarks`). |
| Cognee: BEAM 0.79 at 100K tokens, 0.67 at 10M | [Cognee repo](https://github.com/topoteretes/cognee) | 20 questions from one conversation; the 10M figure is labelled exploratory and was selected and scored on the same questions. Cognee's own page tells readers to read the method before comparing. |
| LightRAG: 60.0% to 84.8% win rate against NaiveRAG; against GraphRAG 52% to 55% on three datasets and 49.6% on the mixed one | [LightRAG repo](https://github.com/HKUDS/LightRAG) | Win rate judged by an LLM; the README gives no latency or token numbers. |
| Independent-ish: on 2WikiMultihopQA, QA F1 of NV-Embed-v2 dense retrieval 61.5, GraphRAG 58.6, LightRAG 11.6; HippoRAG 2 71.0. MuSiQue 45.7, 38.5, 1.6, 48.6. HotpotQA 75.3, 68.6, 2.4, 75.5 | [HippoRAG 2 paper, Table 2](https://arxiv.org/abs/2502.14802) | Run by the HippoRAG authors with Llama-3.3-70B-Instruct and NV-Embed-v2 for every method, and they reimplemented the baselines. LightRAG's collapse may reflect their setup. It is the only same-conditions multi-hop table I found, and a strong dense retriever beats both graph baselines in it. |
| Supermemory: 95% Recall@15 on LongMemEval | [repo](https://github.com/supermemoryai/supermemory) | Recall, not answer accuracy, which scores systematically higher (Mem0's own blog says so). |
| Letta, Microsoft GraphRAG, Graphiti repo pages | repo pages | No benchmark results on the pages read. |

Two cautions the sources themselves support. First, LoCoMo and LongMemEval numbers vary with the dataset revision, answer model, top-k, reranking and judge ([Mem0 blog](https://mem0.ai/blog/locomo-benchmark)), so 92.5 against Zep's 80 to 83 is not a ranking. Second, a plain full-context baseline is competitive on short histories (DMR above).

## Where this project is weaker

Read against [BENCH.md](../BENCH.md), these are the facts, not worries.

1. **No retrieval win on the hard text.** On `hard-small` (145 documents, 300 questions) whole-path@5 is 0.24 for BM25, 0.24 for dense and 0.20 for graph-rag's graph view with the full stack. The graph helps at 2 hops only inside overlapping intervals, and no system recovers a path above 2 hops. The sweep's 2-hop lead (about 18% against 1%) appears only on the plain `small` world and shrinks with graph density.
2. **Extraction has gaps.** With spaCy `en_core_web_sm` doing NER and Haiku building the graph, strict triple F1 was 0.70 on `hard-tiny` and 0.789 on `hard-small`; the two worlds differ by nine points, so neither is a stable figure. The 2026-11 semula runs also showed kg-build missing roles and acquisitions and attaching an event's place to the person; GRA-162 fixed those and the committed results were regenerated afterwards (GRA-166). I have not re-measured on a third world.
3. **The numbers come from an in-repo bench on synthetic worlds.** Competitors report LoCoMo, LongMemEval and BEAM. This project scores the LongMemEval format on a six-question fixture that "says nothing about the real dataset", and the real `longmemeval_s` run and the LLM-judged accuracy are not built. There is no number here that a buyer could put next to Mem0's 94.4.
4. **The real stack has not been run at scale.** The Neo4j, Elasticsearch, Kafka and MinIO path has contract tests but no large-corpus run and no measured cost or latency per document. That is why GRA-55 is open.
5. **Four stateful stores.** Neo4j, Elasticsearch, MinIO and Kafka, about 3 GB of memory at steady state. Mem0 and LightRAG run as a library over one or two stores; Cognee runs locally with embedded defaults. Operating cost and failure modes are higher here.
6. **No SDK, no managed service, no users.** There is a REST API, a CLI and an MCP server. There is no client library and no hosted option, and the UI is not built.
7. **Answer quality is untested against the field.** The default path returns evidence and the top entity, not an answer. Zep and Mem0 numbers are end-to-end answer accuracy.
8. **Small closed predicate set.** Supersession covers four predicates I picked by reading the schema, not by measurement.

## Where it is differentiated

1. **$0 deterministic query.** `/query` makes no LLM call by default and gives the same answer for the same graph. LightRAG calls an LLM at query time by design and Cognee's generated answers need one; the Zep and Mem0 pages I read do not say either way, so I make no claim about them. Cognee's retrieval needs no LLM, so this is not unique.
2. **Runs with no key.** `graph-rag-offline` serves the whole API over in-memory stores. Competitors need at least an embedding key or a local model download.
3. **Evidence with provenance.** A response carries the connected path and the source sentences behind each edge, with `valid_to` history, so an answer can be checked. Zep's validity windows are comparable; Mem0's pages read do not show it.
4. **A generator and a bench that can embarrass the product.** The synthetic generator derives gold from its own world graph, never from the pipeline, and the bench reports negative results (items 1 and 2 above) plus a quality gate that fails CI on a drop. I found no competitor publishing its own losses to BM25.
5. **Tenant scoping at the store seam.** Every store call takes a `Scope`, and with `REQUIRE_AUTH` on the API key names the tenant. Competitors have this in their managed tiers (Mem0's SSO and audit logs are Custom-plan only); self-hosted LightRAG ships with auth off.

None of these is a moat. The first two are cheap to copy; the fourth is a habit, not a product feature.

## What this means for GRA-68

The comparison favours a narrow claim: a memory layer where retrieval costs nothing and every answer can be checked, for developers who want to run it themselves. It does not support a claim of better accuracy. The pricing hypothesis is in [positioning-and-pricing.md](positioning-and-pricing.md).

## Sources not read in full

The Zep and Mem0 docs sites beyond the pricing pages, Letta Cloud pricing, Supermemory pricing, the Mem0 paper's tables and the Mem0 GitHub issue on Zep's LoCoMo score. Each is marked unverified where it matters above.
