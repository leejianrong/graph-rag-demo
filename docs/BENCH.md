# The evaluation bench

2WikiMultihopQA only tells you how the pipeline does on someone else's questions.
To measure retrieval against ground truth we control, the repo is gaining its own
bench in [`bench/`](../bench/) (command `kgbench`, design in
[ADR-0011](adr/0011-in-repo-evaluation-bench.md)). Instead of grading against
an extraction of our own, it scores the pipeline on a synthetic world graph made by the
standalone [generator](#synthetic-data-generator), so the gold answer never depends on the
pipeline under test. Extraction scoring is built; try it with no Docker or key:

```bash
make extract PRESET=small        # kgbench extract: P/R/F1 for mentions, entities and triples, per predicate
make linking PRESET=small        # kgbench linking: B-cubed precision/recall per difficulty setting
make order PRESET=small          # kgbench order: ingest one corpus in shuffled orders, report the spread
```

`kgbench sweep` changes one world knob at a time and scores every system at each value
([docs/results/sweeps/sweep.md](results/sweeps/sweep.md), with a chart and the realised stats of every world). It
reads 2-hop paths, the only hop count where the systems differ: BM25 and dense solve about 1% of them, offline graph-rag
18% on the base world. graph-rag's lead shrinks steadily as the graph gets denser (average degree 2 to 10: +0.28 to
+0.03), and as homonyms (0.5: +0.02) and churn (0.3: +0.03) rise; it is flat across degree skew. The lead over size is
not monotone (+0.15, +0.12, +0.09, +0.20 for 100 to 2,000 entities), which at about 120 questions per point is noise.
A first run scored every question and was mostly noise; it is not published. `make sweep KNOB=avg_degree` runs one
knob (about 10 minutes with bge; all six take over an hour).

`kgbench order` ingests the same corpus as generated and in shuffled orders, each into a fresh pipeline, and reports how much each score moves. The offline pipeline shows none ([docs/results/order-medium.md](results/order-medium.md)); merging by name cannot depend on order, so the question stays open for the real stack (`--real`).

It reports mentions (exact span), entities (distinct gold entities recovered per canonical made) and
triples, strictly and by entity pair only, so you can see a relation found but not named. Entity
linking is scored with B-cubed (merged homonyms cost precision, split aliases cost recall), on a clean,
an alias-heavy and an ambiguous version of the world. Retrieval scoring (recall@k and
whole-path@k per hop count, `kgbench.retrieval`) is built, and a BM25 baseline (`kgbench.baselines`) sits behind the same interface. On the small set it
finds every 1-hop path in the top 5 and almost no 3-hop or 4-hop path: the headroom a graph is meant to
close. A dense baseline (`DenseRetriever`) uses the project's own embedder port: the fake in tests and
bge-small for published numbers (`pip install` the `embed` extra and run `make embed-model`). graph-rag itself is wrapped as a retriever too, in two views: its
passages, and the sentences behind its retrieved subgraph's edges. The first comparison, on `medium`
worlds over three seeds, is in [docs/results/retrieval-medium.md](results/retrieval-medium.md):

```bash
make compare                     # kgbench compare: path@k and recall@k per hop count, every system
```

It took about 8 minutes on a 4 vCPU cloud CPU, and about half an hour on a busy laptop, with bge (`EMBEDDER=fake` is quick and only for checking). The committed numbers came from the cloud run and match an earlier laptop run cell for cell. What it
shows so far: every system finds 1-hop answers (whole-path@5 of 0.94 to 0.99). At 2 hops only the
graph view recovers any whole path (0.11, against 0.00 to 0.01 for BM25 and dense), and at 3 and 4
hops no system recovers one in the top 5. The offline graph-rag does not yet beat the baselines, and
its graph view costs about 4 points at 1 hop. That is the offline heuristic graph, a floor; whether
the real stack's LLM-built graph closes the gap is the open question.

A quality drop fails the build. `make gate` runs the small world offline (no provider, about 15 seconds)
and compares graph-rag's retrieval recall and path recovery per hop count, extraction F1 and linking F1
with the floors stored in [`bench/kgbench/gate.json`](../bench/kgbench/gate.json). The fast CI job and the
pre-push hook run it as a test. After a change that is meant to move the numbers, `uv run kgbench gate
--update` rewrites the floors, so raising or lowering one shows up in the diff.

`kgbench judge` measures an optional LLM judge for open-ended answers: false positive and negative rates on 69 labelled cases, replayed offline from committed Haiku answers ([docs/results/judge.md](results/judge.md)). It is off unless asked for.

`kgbench hard` scores the same world as template text and as the harder Haiku-written text in `datagen/examples/hard-small` ([docs/results/hard-small.md](results/hard-small.md); one world, 300 questions, 75 per hop count, bge embeddings, offline graph-rag). What it shows:

- **Extraction gets worse with the text.** Mention F1 falls from 0.91 to 0.76 and entity F1 from 0.74 to 0.57. Linking F1 barely moves (0.84 to 0.83), and entity-pair F1 is unchanged (0.68 to 0.69).
- **Strict triple F1 is 0.000, by construction.** The offline graph builder links co-occurring entities with `RELATED_TO` only (627 of 627 predicted triples), so no relation name can match; entity-pair F1 is the offline measure. The report leaves the row out. Relations need the LLM builder.
- **Retrieval: harder text costs the 1-hop questions a little** (BM25 and dense 0.95/0.96 to 0.89, graph-rag graph 0.77 to 0.75). Past 1 hop nobody recovers more than a few percent of whole paths, and with 75 questions per row the 2-hop differences (0.03 to 0.08) are noise.
- **graph-rag has no 2-hop lead on this world.** The sweep above found one (about 18% against 1%) on the plain `small` world. This world adds homonyms (0.15), aliases and distractors, so ambiguity is the likely cause, but I did not isolate it.
- **Not covered:** one world, template and LLM text only, no real-stack run, and Haiku rarely used pronouns, so coreference is barely tested.
