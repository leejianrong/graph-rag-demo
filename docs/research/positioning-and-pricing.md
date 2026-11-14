# Positioning and a pricing hypothesis (GRA-68)

Written 2026-11-14 for internal decisions. The cost figures are an **estimate built from five prompts' worth of recordings and list prices**. GRA-55 (measured cost per 1,000 documents) is parked, and this page does not replace it. Where GRA-55 reports a number, that number wins and the tables below get rewritten. The comparison with other products is in [memory-layer-comparison.md](memory-layer-comparison.md).

## Position

**Audience.** Developers who build agents and want a memory and knowledge store they run themselves and can inspect. Close second: a small team with a pile of documents whose useful questions span several of them ("how is X connected to Y"), which is the original use case in [FRAME.md](../FRAME.md).

**Difference.** Retrieval costs nothing: the default `/query` path calls no LLM and returns the same answer for the same graph. Each answer comes with the connected path and the source sentence behind every edge, so a person can check it. The whole API also runs in one process with no key, which makes it cheap to try and cheap to test. The generator and the bench let us measure our own claims, and they have already shown that we do not beat BM25 or a dense retriever on the harder text ([BENCH.md](../BENCH.md)).

**What we do not claim.** Better answer accuracy than Zep, Mem0 or any dense retriever, a managed service, or scale. The comparison page lists the gaps.

## Cost model

Query cost on the default path is $0 in LLM spend. The variable cost is ingestion, where the real stack calls an LLM twice per document (coreference, then graph build). NER, entity linking and embeddings run locally.

### Tokens per document, from recordings

Source: semula recordings of four documents from a scratch run (run 6, not committed) (the prompts after GRA-162): 8 prompts, one coref and one graph-build per document. I counted characters of every request message and response and converted at **4 characters per token**. That ratio is a rule of thumb, not a tokenizer count; the prompts are English prose plus JSON schema, which is usually a little denser than prose, so the estimate probably under-counts tokens by 10 to 30%. Anthropic's pricing page also says models from 4.7 onward use a tokenizer that produces about 30% more tokens, so treat the token counts as plus or minus 30%.

| Stage | Prompts | Request chars (mean) | Response chars (mean) | Of which document text |
|---|---|---|---|---|
| Coref | 4 | 2,102 | 291 | 120 to 159 chars |
| Graph build | 4 | 4,020 | 427 | same document |
| Per document | | 6,122 (about 1,530 tokens) | 718 (about 180 tokens) | about 141 chars |

What this shows: for a two-sentence document, **about 95% of the input is fixed prompt**: the instructions, the JSON schema (about 1,700 characters in graph build) and the entity list. The fixed part is about 1,460 tokens per document across both calls. Run 5 (before GRA-162) was cheaper in the graph-build prompt (about 2,980 characters) because the role, acquisition and event rules were shorter, so prompt rules cost tokens on every document. Four documents is a tiny sample, the recorded documents are short and fact-dense, and the counts are characters, not provider-reported usage.

### Formula

For a document of D tokens:

```
input tokens   = F + 2D        F is about 1,460 (fixed prompt, both calls), D is read twice
output tokens  = 180 + 0.5 (D - 35)   assumption: output grows with document length, at a lower density than the recorded fact-dense text
cost per doc   = (input x price_in + output x price_out) / 1,000,000
cost per 1k    = 1,000 x cost per doc
```

The output slope (0.5 output tokens per extra document token) and the absence of growth in the entity list inside the graph-build prompt are **assumptions, not measurements**. Long documents with many entities will push both up.

### List prices used

All per million tokens, standard (non-batch) input/output, read 2026-11-14 from [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing): Claude Haiku 5.5 $0.10 / $0.50 (prompts up to 100,000 tokens), Claude Haiku 4.5 $1 / $5, Claude Sonnet 5.5 $2 / $10. The Batch API takes 50% off both. I do not know which Haiku model produced the recorded answers (the recording server logs the model as `semula`), so I bracket with two Haiku generations. A web search summary lists GPT-5 mini at $0.25 / $2.00 and GPT-5 nano at $0.05 / $0.40; I could not read OpenAI's own page (HTTP 403), so those are unverified and not used below.

### Result: LLM cost per 1,000 documents

Three document sizes: 35 tokens (the recorded shape), 700 tokens (about 2,800 characters, a news article), 3,000 tokens (about 12,000 characters, a long report).

| Document | Input / output tokens per doc | Haiku 5.5 | Haiku 4.5 | Sonnet 5.5 |
|---|---|---|---|---|
| 35 tokens | 1,530 / 180 | $0.24 | $2.43 | $4.86 |
| 700 tokens | 2,860 / 512 | $0.54 | $5.42 | $10.85 |
| 3,000 tokens | 7,460 / 1,662 | $1.58 | $15.77 | $31.55 |

Batch halves each cell. A 30% allowance for retries and token-count error multiplies each by 1.3 (700-token row: $0.70, $7.05, $14.10). **Working range for a news-sized document: about $0.3 to $14 per 1,000 documents, centred on $0.5 to $11 before retries.** The spread is mostly model choice (a factor of 20 between Haiku 5.5 and Haiku 4.5), not my assumptions. Larger models for the graph-build step would add more.

### Infrastructure

The stack's steady-state memory is about 3 GB: Neo4j about 800 MB, Elasticsearch about 740 MB, app about 620 MB, worker 590 to 1,100 MB. That needs an 8 GB machine once the operating system and page cache are counted; a 4 GB machine has no headroom. A 4 vCPU, 8 GB shared-CPU server is listed at EUR 30.49 to 62.99 per month excluding VAT depending on region ([Hetzner, read 2026-11-14](https://www.hetzner.com/cloud/regular-performance)); I use EUR 30 to 63, roughly $33 to $70 at an assumed $1.10 per euro. That is **one dedicated stack**: infra per customer is the whole figure unless tenants share a stack. Tenant scopes exist (ADR-0015), so sharing is possible, but I have not measured how many tenants one stack carries or how disk grows per 1,000 documents. Storage, Kafka and MinIO disk and backups are not included.

### What GRA-55 must measure to replace this

1. Tokens in and out per stage per document from the provider's usage fields, on a real corpus of at least 500 documents across the three size bands, with the named model and date.
2. Retry and structured-output failure rate, and the growth of the graph-build prompt with entity count.
3. Wall-clock time per document and the worker's peak memory under N documents.
4. Disk used by Neo4j, Elasticsearch and MinIO per 1,000 documents, and steady memory at 10k and 100k documents.
5. The resulting cost per 1,000 documents at each size, with Haiku and a larger model.

## Pricing hypothesis

**Metric: per document ingested, with queries unmetered.** The reason is cost shape. Queries cost $0 on the default path, so charging per query would tax the thing we do cheaply and encourage customers to ration the thing that makes the store useful. Ingestion is where the LLM bill is, so it is the metric that tracks our cost. It is also the metric competitors use for the same step: Zep bills credits per episode (1 per 350 bytes), with retrieval, storage and users unmetered ([Zep pricing](https://www.getzep.com/pricing)); Cognee bills $1.00 per million tokens processed ([Cognee pricing](https://www.cognee.ai/pricing)). Per seat does not fit an API used by agents. Flat fees hide the cost driver: one customer ingesting 3,000-token reports costs about 30 times what one ingesting 35-token notes does.

**Unit.** A document unit is 4,000 characters or part of it, so a long report counts as several. This follows Zep's 350-byte unit in shape and keeps the price proportional to tokens.

**Shape, as a hypothesis to test with customers, not a quote:**

| Tier | Price | Includes |
|---|---|---|
| Self-host | $0 | Apache-2.0 code, no limits, you pay your own LLM and servers |
| Hosted | $49 per month plus $10 per extra 1,000 document units | 5,000 units, unlimited queries, MCP, API, CLI, tenant isolation |
| Hosted, your own LLM key | $49 per month plus $3 per extra 1,000 units | Same; the customer's key pays the provider |

**Check against the cost model (700-token document, one unit is roughly one document):**

| Case | LLM cost per 1k | Margin at $10 per 1k |
|---|---|---|
| Haiku 5.5 | $0.54 | about 95% |
| Haiku 4.5 | $5.42 | about 46% |
| Sonnet 5.5 | $10.85 | negative |

So the overage price covers Haiku-class models and not a larger one, which is why the model is ours to choose on the hosted tier and why the bring-your-own-key row exists. The $49 base only works on a **shared** stack: 5,000 units of 700-token documents cost $2.70 (Haiku 5.5) to $27 (Haiku 4.5) in LLM spend, and a dedicated EUR 30 to 63 server on top gives $36 to $97 against $49 of revenue. At 20 tenants per stack (my guess, not measured) infra is under $4 per tenant. For comparison, Zep's Flex plan is $125 per month for 50,000 credits, about 6,000 700-token documents at 8 credits each, so $49 for 5,000 units sits well under it; Cognee at $1 per million tokens would charge about $0.70 per 1,000 of these documents, which is close to our cost on Haiku 5.5 and below it on Haiku 4.5. Cognee's price is the one we cannot undercut.

**Self-host remains free.** The code is Apache-2.0 and so are Mem0, Zep's Graphiti and Cognee, so hosting convenience is what is being sold.

## One risk

The product's retrieval does not beat BM25 or a dense retriever on its own bench: whole-path@5 of 0.20 to 0.24 on `hard-small`, with no 3-hop or 4-hop path recovered by any system. A per-document price charges for the step (LLM graph building) whose retrieval benefit is not yet shown, against competitors who publish LoCoMo and LongMemEval numbers. If the real stack on a real corpus does not show a gain, the honest position is the $0-query, checkable-evidence story only, and the price should fall to the level of Cognee's $1 per million tokens or below. Decide the price after GRA-55 and a real-stack retrieval run, not before.
