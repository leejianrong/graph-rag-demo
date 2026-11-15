# Where graph-rag loses multi-hop paths, and a chain search that recovers some (GRA-170)

Done on 2026-11-15 on `hard-small` (one world, 145 Haiku-written documents, 300 questions, 75 per hop count), with the recorded LLM stages replayed from the committed cache, spaCy `en_core_web_sm` and bge embeddings. The card's goal was to beat BM25 and dense. **On `hard-small` overall it does not, within the noise; from two hops up it does, and on three other generated worlds it does by a wider margin. It loses one-hop questions to plain search.**

## 1. The trace

`kgbench trace` (new) follows every question's gold path through the stages and names the first stage where it is lost ([trace-hard-small.md](trace-hard-small.md) has the full tables). It reproduces the published whole-path@5: 61 of 300 for the graph view, 72 for dense, 71 for BM25.

The 239 questions the graph view does not solve, by the first stage that fails them:

| stage | questions | share | by hop count (1, 2, 3, 4) |
|---|---|---|---|
| 1 extraction or graph build: a path fact has no edge from its sentence | 110 | 46% | 13, 30, 30, 37 |
| 4 traversal: the correct edges are not in the retrieved subgraph | 62 | 26% | 2, 7, 26, 27 |
| 5 ranking: the whole path is in the subgraph but not in the top 5 | 62 | 26% | 6, 30, 17, 9 |
| 2 linking: an edge joins the wrong entities | 2 | 1% | 0, 1, 0, 1 |
| 3 seeding: the start entity is not among the seeds | 3 | 1% | 1, 0, 1, 1 |

- **Extraction is the largest single cause on this recording**, and it grows with hop count because a path needs every one of its facts: 189 of 300 questions have an edge for every fact. Looked into afterwards (GRA-179, [hard-small-trf.md](hard-small-trf.md)): of the 45 path facts with no right edge, 35 are the NER missing an endpoint, because this cache used `en_core_web_sm`. With the product-default `en_core_web_trf` and a re-recorded builder, extraction loses none of the 300 questions, and ranking and traversal are what is left. The numbers below were measured on the `sm` recording.
- **Together the two retrieval stages lose 124**, as many as extraction. Of the 187 questions whose edges are all right, 123 have the whole path inside the retrieved subgraph and only 61 are solved.
- **Linking is not the problem here, by this measure.** It counts an edge as right when its endpoints carry the gold entities' names or aliases, so a merge of two same-named entities is invisible to it. It does not contradict the entity merges found on 2Wiki ([real-2wiki-medium-50.md](real-2wiki-medium-50.md)); it does not test for them.
- **Seeding is fine**: the start entity is among the seeds for 278 of 300.

The subgraph is large. It averages 59 of the graph's 263 edges, so "the path is in the subgraph" is a weak statement, and the ranking stage is picking 2 to 4 edges from about 59.

## 2. What did not work

Four orderings of the subgraph's edges, all tried on `hard-small` and all scored on whole-path@5 over the same 300 questions:

| ordering | solved of 300 |
|---|---|
| current graph view (edges by the rank of their endpoints) | 61 |
| each edge's sentence by cosine to the question | 63 |
| greedy chain from the seeds, best cosine at each step | 60 |
| **beam search over chains, scored by cosine to the chain's sentences joined** | **66** |

A later hop's sentence names an entity the question never mentions, so it scores low on its own, and ranking edges one at a time cannot find it. Scoring the whole chain can. `KHOP_DEPTH=3` (a deeper subgraph) with the chain search gave 0.26 overall against 0.27 at depth 2 and the same 0.07 at three hops, so the 2-hop expansion is not what stops the longer questions.

## 3. The change

`PATH_RETRIEVAL` (default off, so nothing changes unless it is set): `graph_rag/query/chains.py` runs a beam search from the seed entities over the subgraph, growing a chain one edge at a time up to `PATH_DEPTH` (4) edges, keeping `PATH_BEAM` (8) chains per length, and scores each chain by the cosine between the question and the chain's sentences joined. The response's `subgraph.edges` is then listed with the best chains' edges first. That list's order was unspecified, so the response shape is unchanged. No LLM call and no new store; it embeds the chains' sentences with the embedder the retriever already has.

The bench gets a `graph-rag path` row, added only when the retriever has the flag on: the two nearest supporting sentences, then the sentences behind the edges in the retriever's order, then the rest. The two nearest are what keep one-hop questions that plain search already gets.

Cost: about 20 ms per query on CPU on a 1,000-entity graph (60 queries, 15.5 s against 14.3 s for the graph row). It scales with the beam and the node degree.

## 4. Results

Whole-path@5, 95% Wilson intervals on pooled counts (the held-out rates are means over seeds, so their counts are rounded).

**`hard-small`, where it was designed** (LLM and spaCy pipeline, 300 questions; [path-retrieval-hard-small.md](path-retrieval-hard-small.md)). With the flag off the same command reproduces the published 0.24, 0.24, 0.24 and 0.20.

| hops | BM25 | dense | graph view (flag off) | graph-rag path |
|---|---|---|---|---|
| all | 24% (19–29) | 24% (20–29) | 20% (16–25) | **27%** (22–32) |
| 1 | 89% | 89% (80–94) | 71% | 81% (71–89) |
| 2 | 3% (1–9) | 7% (3–15) | 9% | **19%** (11–29) |
| 3 | 3% | 0% (0–5) | 1% | **7%** (3–15) |
| 4 | 0% | 0% | 0% | 0% |

**Three other worlds, held out** (`small` preset, seeds 0, 1, 2, template text, the offline pipeline, 300 questions; [path-retrieval-small-worlds.md](path-retrieval-small-worlds.md)):

| hops | BM25 | dense | graph view | graph-rag path |
|---|---|---|---|---|
| all | 25% (20–30) | 25% (20–30) | 28% | **35%** (30–41) |
| 1 | 95% | 99% (93–100) | 96% | 89% (80–94) |
| 2 | 4% (1–11) | 1% (0–7) | 16% | **40%** (30–51) |
| 3 | 0% | 0% | 0% | 9% |
| 4 | 0% | 0% | 0% | 1% |

One `medium` world (1,000 entities, 817 documents, seed 0, 60 questions, CPU): path 33% against 25% for BM25 and dense, 27% against 0% at two hops, with 15 questions per hop count, so an indication only.

- **Overall on `hard-small` it is not a win.** 27% (22–32) against 24% (20–29) overlaps. At two hops 19% (11–29) against 7% (3–15) overlaps a little; I would not call either decisive.
- **On the held-out worlds the two-hop gain is large and clear** (40% against 1% and 4%), and the overall gain is borderline (35% against 25%, intervals touch at 30%). Those worlds use template text and the offline graph, which links co-occurring entities, so chains are easy to find; this says less about real text.
- **It costs one-hop questions**: 81% against 89% on `hard-small`, 89% against 99% on the small worlds. A likely cause, which I did not check: the chain's sentences push the one nearest sentence that plain search gets right out of the top 5 for some of those questions. Keeping more nearest sentences might help one hop and cost the others; I did not try it, because every extra setting tried on `hard-small` makes its numbers less honest.
- **Nothing helps at four hops** (0 to 1%), and three hops reach 7 to 9%.

## 5. Caveats

- **The method was designed on `hard-small`.** I tried the four orderings above and `KHOP_DEPTH=3` there, and nothing else: the beam is 8 and the nearest-two split was fixed before the first run of the chain search. The held-out worlds were not looked at until the design was fixed, but they share the generator, so they are a different draw from the same family, not a different kind of data.
- **No real-text check.** The 2Wiki run (GRA-15, 50 questions) replays from a committed cache and would show whether this helps on real text. I did not run it with the flag on.
- **One embedder (bge-small), one recording model for the LLM stages, one world for the real-text-like case.**
- **The quality gate is unchanged**, because the flag is off by default; its floors would only move if the gate ran with `PATH_RETRIEVAL` on.
- A first `medium` run on three seeds and 200 questions ran for 49 minutes on the shared GPU without finishing and was stopped. I did not find the cause; the one-seed run above took six minutes on CPU.

## 6. Reproduce

```bash
# the trace (CPU is enough; replays the committed LLM answers)
uv run --extra embed kgbench trace --dir datagen/examples/hard-small --pipeline llm+spacy --embedder bge --k 5 \
  --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm
# the comparison with the new row (set PATH_RETRIEVAL=false for the published numbers)
PATH_RETRIEVAL=true uv run --extra embed kgbench hard --dir datagen/examples/hard-small --questions 300 \
  --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm \
  --texts hard --pipelines llm+spacy
PATH_RETRIEVAL=true uv run --extra embed kgbench compare --preset small --seeds 0 1 2 --questions 100 --embedder bge --k 5
```

Set `OPENAI_API_BASE` to an unreachable address and `OPENAI_API_KEY` to anything so a cache miss fails instead of calling a provider, and `CUDA_VISIBLE_DEVICES=` to keep it off a shared GPU.
