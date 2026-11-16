# The temporal bench: who finds the right version of a fact? (GRA-190, bench half)

Done on 2026-11-16. The generator can now make a world that changes ([generator-evolution.md](generator-evolution.md)); this scores systems on it. `kgbench temporal` builds an evolving world (`heavy`: churn, retraction, births, closures, renames), writes its dated timeline documents in date order, derives 150 questions of five temporal types (30 each: `as_of`, `latest`, `when`, `before`, `duration`) and runs BM25, dense search and graph-rag on them. A system that can use a date is given the question's year (`as_of`; graph-rag passes the end of that year, because documents are dated mid-year); BM25 and dense search cannot and work from the words. bge-small-en-v1.5 embeddings, the offline graph-rag pipeline, `small` world, seed 0, 95% Wilson intervals.

## Two measures

Whole-path@5 is the bench's usual one: every gold fact is expressed by one of the top 5 sentences. It cannot tell a right answer from a stale one, because the sentence for the old employer and the sentence for the new one can both be in the top 5. So there is a second measure: **the first sentence about the fact states the right version.** Among the retrieved sentences about the same subject and relation, the first must state the gold fact (for `as_of` the version that held that year, for `latest` the current one) and not another version. A reader who takes the top evidence gets the wrong answer when this fails.

## Hard world (aliases, shared names, look-alikes; 180 entities, 433 facts, 375 dated documents)

**Whole path in the top 5**

| type | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| as_of | 30 | 87% (26/30, 70%–95%) | 90% (27/30, 74%–97%) | 70% (21/30, 52%–83%) | 47% (14/30, 30%–64%) |
| latest | 30 | 67% (20/30, 49%–81%) | 57% (17/30, 39%–73%) | 53% (16/30, 36%–70%) | 27% (8/30, 14%–44%) |
| when | 30 | 80% (24/30, 63%–90%) | 97% (29/30, 83%–99%) | 77% (23/30, 59%–88%) | 57% (17/30, 39%–73%) |
| before | 30 | 63% (19/30, 46%–78%) | 63% (19/30, 46%–78%) | 50% (15/30, 33%–67%) | 27% (8/30, 14%–44%) |
| duration | 30 | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 87% (26/30, 70%–95%) | 50% (15/30, 33%–67%) |
| all | 150 | 79% (119/150, 72%–85%) | 81% (122/150, 74%–87%) | 67% (101/150, 59%–74%) | 41% (62/150, 34%–49%) |

**The first sentence about the fact states the right version** (not an older or later one)

| type | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| as_of | 30 | 33% (10/30, 19%–51%) | 50% (15/30, 33%–67%) | 47% (14/30, 30%–64%) | 27% (8/30, 14%–44%) |
| latest | 30 | 13% (4/30, 5%–30%) | 30% (9/30, 17%–48%) | 27% (8/30, 14%–44%) | 3% (1/30, 1%–17%) |
| when | 30 | 77% (23/30, 59%–88%) | 97% (29/30, 83%–99%) | 77% (23/30, 59%–88%) | 50% (15/30, 33%–67%) |
| before | 30 | 90% (27/30, 74%–97%) | 90% (27/30, 74%–97%) | 90% (27/30, 74%–97%) | 57% (17/30, 39%–73%) |
| duration | 30 | 90% (27/30, 74%–97%) | 100% (30/30, 89%–100%) | 87% (26/30, 70%–95%) | 33% (10/30, 19%–51%) |
| all | 150 | 61% (91/150, 53%–68%) | 73% (110/150, 66%–80%) | 65% (98/150, 57%–72%) | 34% (51/150, 27%–42%) |


## Clean world (150 entities, 405 facts, 347 dated documents)

**Whole path in the top 5**

| type | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| as_of | 30 | 83% (25/30, 66%–93%) | 83% (25/30, 66%–93%) | 80% (24/30, 63%–90%) | 70% (21/30, 52%–83%) |
| latest | 30 | 87% (26/30, 70%–95%) | 77% (23/30, 59%–88%) | 77% (23/30, 59%–88%) | 23% (7/30, 12%–41%) |
| when | 30 | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 67% (20/30, 49%–81%) |
| before | 30 | 73% (22/30, 56%–86%) | 83% (25/30, 66%–93%) | 83% (25/30, 66%–93%) | 23% (7/30, 12%–41%) |
| duration | 30 | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 60% (18/30, 42%–75%) |
| all | 150 | 89% (133/150, 83%–93%) | 89% (133/150, 83%–93%) | 88% (132/150, 82%–92%) | 49% (73/150, 41%–57%) |

**The first sentence about the fact states the right version** (not an older or later one)

| type | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| as_of | 30 | 47% (14/30, 30%–64%) | 57% (17/30, 39%–73%) | 70% (21/30, 52%–83%) | 57% (17/30, 39%–73%) |
| latest | 30 | 43% (13/30, 27%–61%) | 33% (10/30, 19%–51%) | 33% (10/30, 19%–51%) | 0% (0/30, 0%–11%) |
| when | 30 | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 63% (19/30, 46%–78%) |
| before | 30 | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 70% (21/30, 52%–83%) |
| duration | 30 | 93% (28/30, 79%–98%) | 100% (30/30, 89%–100%) | 100% (30/30, 89%–100%) | 33% (10/30, 19%–51%) |
| all | 150 | 77% (115/150, 69%–83%) | 78% (117/150, 71%–84%) | 81% (121/150, 74%–86%) | 45% (67/150, 37%–53%) |


## What it says

- **Retrieving the sentence is mostly solved; choosing the version is not.** On the hard world BM25 and dense put the gold sentence in the top 5 for 79% and 81% of questions, but for `latest` the first sentence about the fact is the current one only 13% (BM25) and 30% (dense) of the time, and for `as_of` 33% and 50%. Every system lists an older version first more often than not on those two types. The documents that state a fact when it begins and again when it ends give a word search two near-identical sentences to choose between.
- **`when`, `before` and `duration` are easy by comparison.** The gold sentence names both entities and the year, and there is rarely a rival sentence ahead of it (dense 97-100%). `before` and `when` are weak tests of recency: the gold may be any of two facts, so a version check passes more easily.
- **graph-rag is no better, and its graph view is worse.** The offline pipeline builds a co-occurrence graph whose only relation is `RELATED_TO`, so it has no single-valued relation for supersession to end, no `valid_to` ever gets set, and the `as_of` filter can only drop facts that begin after the year. Handing it the year therefore does little. The graph view also pulls sentences by the entities' rank, not by the question's words, which loses here (41% against 81%).
- **This is the floor, not the real stack.** What the real pipeline does with dated text (read `since 2011` and `until 2019`, supersede on a newer document) needs an LLM-built graph. That recording was not done here; see below.

## Gate

Thirteen new floors in `bench/kgbench/gate.json`: graph-rag whole-path@5 per temporal type, overall, and the first-version share, on the gate's `small` world with `heavy` evolution (clean, 100 questions, fake embedder, no provider). They guard the plumbing: the dated documents, `document_date` reaching ingest (a test fails without it), `as_of` reaching the retriever. They are floors for an offline pipeline that cannot use time, so they will not catch a regression in real temporal behaviour.

## Caveats

- One world, one seed, 30 questions per type: the intervals are wide (about 15 points either way).
- The offline pipeline only. No LLM-written text and no LLM-built graph, so nothing here says how the real stack handles time. A recorded run (a Haiku agent through semula, as for `hard-small`) is the next step; it costs a prompt or two per document.
- BM25 and dense search are given no recency signal; a trivial one (prefer the sentence from the later document) would change the `latest` numbers. That is a baseline the product should be compared with.
- Documents state every date as an explicit year. No relative dates, no implicit change (GRA-188).

## Reproduce

```bash
export LITELLM_MODE=PRODUCTION UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=
uv run --extra embed kgbench temporal --difficulty hard --embedder bge --questions 150
uv run --extra embed kgbench temporal --difficulty clean --embedder bge --questions 150
uv run kgbench gate            # the floors
```
