# hard-small with the product-default NER (GRA-179)

Done on 2026-11-15. The same world, documents and 300 questions as [hard-small-pipelines.md](hard-small-pipelines.md), but with `en_core_web_trf` (the NER the product uses by default) in place of `en_core_web_sm`, and the coreference and graph-build stages re-recorded by a Haiku agent through semula (290 answers, committed, replayed here with an unreachable endpoint so any provider call would fail). Everything else is as before: bge-small-en-v1.5 embeddings, one world, the Haiku-written text.

## Why this was run

[path-retrieval.md](path-retrieval.md) found that extraction was the largest single cause of lost paths on `hard-small`: 110 of the 239 unsolved questions had a path fact with no edge. Looking into it ([trace-hard-small.md](trace-hard-small.md), the last section) put 35 of the 45 facts without a right edge on the NER missing an endpoint, 7 on the builder emitting nothing, 1 on a wrong sentence index and 2 on a wrong pair. Counting detected endpoints with no LLM at all:

| NER | path-fact endpoints found | facts with both found |
|---|---|---|
| `en_core_web_sm` (what the recorded cache used) | 85% | 74% |
| `en_core_web_trf` (the product default) | 100% | 100% |
| the rule-based detector | 100% | 100% |

(Detection counts any predicted span overlapping the gold mention, so it is lenient; 215 path facts, 430 endpoints.) The published `hard-small` pipelines used `sm` only because that was the cache that had been recorded. So the "extraction" cause was mostly a smaller model than the product default. This run re-records the LLM stages with `trf` to see what is left.

## Extraction

| | `sm` + Haiku (published) | `trf` + Haiku (this run) |
|---|---|---|
| mention F1 | 0.856 | 0.900 |
| entity F1 | 0.728 | 0.737 |
| entity-pair F1 | 0.796 | 0.881 |
| triple F1 (strict) | 0.789 | 0.876 |
| linking F1 | 0.859 | 0.870 |

## Where paths are lost now ([trace-hard-small-trf.md](trace-hard-small-trf.md))

| first stage that fails the question | `sm` (published trace) | `trf` |
|---|---|---|
| 1 extraction or graph build: a path fact has no edge | 110 | **0** |
| 2 linking: an edge joins the wrong entities | 2 | 5 |
| 3 seeding | 3 | 4 |
| 4 traversal: the correct edges are not in the subgraph | 62 | 87 |
| 5 ranking: the whole path is in the subgraph but not in the top 5 | 62 | 133 |
| solved (graph view) | 61 | 71 |

Every one of the 300 questions has an edge for every fact of its path. The two retrieval stages are now almost everything that is left: 220 of the 229 unsolved questions. Of the 221 distinct path facts, 210 have the right edge and 11 have edges from their sentence that join other entities. (The `sm` and `trf` columns count the first stage to fail, so a question the `sm` run lost to extraction moves to a later stage here; the stage counts are not additive across the two.)

## Retrieval

Whole-path@5 over the 300 questions, 95% Wilson intervals. "Graph view" is the flag-off replay of the committed cache; "path" is `PATH_RETRIEVAL=true` ([hard-small-trf-path.md](hard-small-trf-path.md)).

| hops | BM25 | dense | graph view | graph-rag path | (`sm`: graph view, path) |
|---|---|---|---|---|---|
| all | 24% | 24% (20–29) | 24% (20–29) | **31%** (26–36) | 20%, 27% |
| 1 | 89% | 89% (80–94) | 87% (77–93) | 85% (76–92) | 71%, 81% |
| 2 | 3% (1–9) | 7% (3–15) | 7% (3–15) | **28%** (19–39) | 9%, 19% |
| 3 | 3% | 0% (0–5) | 1% | 9% (5–18) | 1%, 7% |
| 4 | 0% | 0% | 0% | 0% | 0%, 0% |

- **With `trf` the graph view no longer loses to plain search.** It ties dense overall (24%) and is within noise at one hop (87% against 89%). With `sm` it lost 18 points at one hop.
- **The chain search (`PATH_RETRIEVAL`) is now clearer.** 31% (26–36) against 24% (20–29) overlaps only a little, and the two-hop gain is large: 28% (19–39) against 7% (3–15). The one-hop cost shrinks to 85% against 89%, inside the intervals. Nothing helps at four hops.
- **The flag-on run's own `graph-rag graph` row says 0.26**, not 0.24, because the chain-first edge order changes that row's tie-break. The 0.24 figure is the flag-off replay; I use it as the graph view.
- This is the world the chain search was designed on; the held-out check in [path-retrieval.md](path-retrieval.md) did not use `trf` or this recording.

## Caveats

- One world, one embedder, one recording model. The answers are a Haiku agent's, written without seeing the other prompts, and are not the API model's.
- The agent reported its own doubtful answers: a number of coref merges made on a surname, initial or short form the text does not state outright (for example `Godi` with `G. Gireme`, `Luse` with `Luse Group`, `Dune` with `Dune Partners`); some short forms kept separate that may be one entity (`VG` and `Velu Group`, `Pidu` and `Pidu Works`); one wrong merge, `Duva` into the `Duva Z1` cluster in the document "Duva Z1 is part of Duva Q36"; three triples left out because an endpoint was not a canonical entity in the prompt (`VaLo`, `Rito Summit`, `VC`); a self-loop `Gaze Industries AFFILIATED_WITH Gaze Industries` at confidence 0.5. I did not read every answer. The agent used about 865k tokens for the 290 prompts, a subscription cost, not an API bill.
- The synthetic worlds' invented, capitalised names suit `trf` and a rule-based detector alike (both found every endpoint), so this says little about real text. On the 2Wiki run `trf` was already in use.

## Reproduce

```bash
# replay (no provider call is possible: the endpoint is unreachable)
export OPENAI_API_BASE=http://127.0.0.1:1/v1 OPENAI_API_KEY=unused CUDA_VISIBLE_DEVICES=
PATH_RETRIEVAL=false uv run --extra embed kgbench hard --dir datagen/examples/hard-small --questions 300 \
  --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small-trf --spacy en_core_web_trf \
  --texts hard --pipelines llm+spacy
uv run --extra embed kgbench trace --dir datagen/examples/hard-small --pipeline llm+spacy --embedder bge --k 5 \
  --llm-cache bench/kgbench/data/llm-cache-hard-small-trf --spacy en_core_web_trf
```

Use `PATH_RETRIEVAL=true` for the path row. The cache replays only with `en_core_web_trf` 3.8.0, bge-small-en-v1.5, model string `openai/claude-haiku` and the prompts and entity-linking rule of the day.
