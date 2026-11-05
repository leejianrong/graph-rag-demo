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

**The real LLM stages, through semula.** `build_offline_components(llm_stages=True)` swaps the offline coref and graph-build heuristics for the real LLM stages (NER, linking and the stores stay in memory), and `kgbench hard --llm-cache DIR [--llm-base-url URL]` scores them. Point the base URL at a semula container and have a Haiku agent answer the parked prompts; every answer is cached, so a re-run needs no endpoint. [docs/results/hard-tiny.md](results/hard-tiny.md) does this on the `hard-tiny` world (26 documents, Haiku as the coref and graph-build model):

- **Relations appear.** Strict triple F1 is 0.92 on template text and 0.96 on the harder text, where the offline builder cannot score at all. Entity-pair F1 rises from 0.77 and 0.75 to 0.91 and 0.96.
- **Mentions and entities do not move.** NER is the same offline heuristic, so mention F1 (0.99 template, 0.75 hard) is unchanged and entity F1 gains only 3 points.
- **Retrieval: a small, unproven gain on the hard text.** graph-rag graph path@5 / recall@5 goes from 0.20 / 0.45 to 0.25 / 0.53 over all 60 questions. At 15 questions per hop count and one world, that is one sample, not a result.
- **Cost:** 104 recorded Haiku answers for 48 documents, no API spend; I did not measure the time per document.
- **Not covered:** `hard-small` (145 documents, about 290 sequential prompts), a different model, the real stores and real spaCy NER.

**Real NER changes the picture** (`build_offline_components(spacy_ner=True)`, `kgbench hard --spacy en_core_web_sm`; [docs/results/hard-tiny.md](results/hard-tiny.md), with charts). On `hard-tiny`, spaCy's small model against the offline heuristic NER:

- **Heuristic NER is nearly an oracle on this data.** The generator's names are invented, capitalised words, which a capitalisation heuristic finds almost perfectly. So the LLM-stage numbers above (triple F1 0.92 and 0.96) were flattered by the NER stage beneath them.
- **spaCy loses mentions on template text, gains on the harder text.** Mention F1 0.985 to 0.907 on template sentences (it misses some invented names) and 0.753 to 0.804 on the harder prose.
- **Relations cost most.** With Haiku building the graph and spaCy finding the entities, strict triple F1 on the harder text falls from 0.96 to 0.70 and entity-pair F1 from 0.96 to 0.67. Graph building cannot link entities NER never found.
- **graph-rag retrieval with spaCy** is no better than without: on the hard text graph-rag's whole-path@5 is 0.22 with spaCy and the LLM builder, against 0.25 with the heuristic NER and the LLM builder, and BM25 stays at 0.30 (60 questions, 15 per hop count: differences of a few points are noise).
- **Not covered:** the transformer model (`en_core_web_trf`, the product default; a much larger download), one world, and the `small` world (145 documents, about 290 plus further sequential prompts per pipeline).

**The fuller stack on `hard-small`** (`kgbench hard --dir datagen/examples/hard-small --texts hard --pipelines offline spacy llm+spacy`; [docs/results/hard-small-pipelines.md](results/hard-small-pipelines.md), with charts). 145 Haiku-written documents, 300 questions (75 per hop count), bge embeddings; Haiku as the coref and graph-build model through semula (288 recorded answers, committed, replayable offline) and spaCy `en_core_web_sm` as NER. One world.

- **Extraction improves down the stack.** Mention F1 0.756 (heuristic NER) to 0.856 (spaCy); entity-pair F1 0.690 to 0.712 (spaCy, offline builder) to 0.806 (Haiku builder). With the Haiku builder, strict triple F1 is 0.796, where the offline builder cannot score. Linking F1 is flat (0.83 to 0.84).
- **On the smaller `hard-tiny` world the same stack scored 0.70 triple F1.** The two worlds disagree by nine points, so treat either as one sample, not a stable figure.
- **Retrieval: graph-rag does not win.** Whole-path@5 over all 300 questions: BM25 0.24, dense 0.24, graph-rag (passages) 0.24, graph-rag (graph view) 0.21 with the full stack. At 1 hop BM25 and dense (0.89) beat the graph view (0.75). At 2 hops the graph view has the highest score (0.09, 7 of 75; 95% interval 4.6% to 18.0%) but dense has 0.07 (5 of 75) and BM25 0.03 (2 of 75): the intervals overlap heavily, so that is not a lead.
- **No hop count above 2 is solved** by any system (at most 0.03).
- **Not covered:** the transformer NER model, a second world or seed, the template text at this scale with these pipelines (not run, to save about 290 prompts), and the real stores.

**Are the questions answerable from the text? A critique after the Hugging Face cookbook** (`kgbench critique --dir datagen/examples/hard-tiny`; [docs/results/critique-hard-tiny.md](results/critique-hard-tiny.md), chart and every rating beside it). Haiku, through semula, rates each of the 60 `hard-tiny` questions 1–5 on groundedness (answerable from the documents holding its supporting facts), relevance and standalone, rationale first, with the cookbook's prompts (relevance reworded for this domain). The cookbook keeps a question when all three are at least 4. 180 ratings, committed and replayable offline. One world, all `entity` questions, one rater model.

- **Groundedness falls with hops:** 100% at 1 hop (15/15, interval 80–100%), 73% at 2 and 3, 53% at 4 (8/15, 30–75%); 75% overall (45/60).
- **The failures are name collisions.** The world has seven surface forms shared by two entities (for example two "Dide Holdings", both with the alias "Dide"). Questions whose path touches such an entity: 23 of 37 grounded (62%, 46–76%). Questions that don't: 22 of 23 (96%, 79–99%). I read all 15 low-groundedness rationales: every one concerns a name the text cannot resolve between entities, and one (Q39) is also a real ambiguity in the path. This is the generator's deliberate hard linking case, so these questions are hard by design, but they are not answerable from the text alone and the dataset does not flag them.
- **Standalone** passes 60/60, as expected for templated wording.
- **Relevance does not discriminate here.** Ratings sit at 2–3 for nearly everything (9/60 reach 4), and the 3- and 4-hop questions all fail it. The rater is judging invented worlds and long chained wording, not whether the question is answerable. Under the cookbook rule, 8 of 60 questions are kept (13%); I would not read that as the dataset being 87% bad, and the "groundedness and standalone" column (75%) is the figure that measures the text.
- **Caveats:** the rater is the same model family that wrote the articles and the judge cases, only the groundedness failures were read, and 15 questions per hop count give wide intervals. The ratings have not been checked against a human (GRA-128).
- **Follow-up:** the generator could record which questions touch a shared name, so a score can be reported with and without them.

**Can the questions be answered without the chain? A shortcut check** (`uv run --extra embed kgbench shortcut --dir datagen/examples/hard-small --embedder bge --texts hard template`; [docs/results/shortcut-hard-small.md](results/shortcut-hard-small.md), with a chart per system and text). MuSiQue is built so a single-hop model fails; this asks the same of ours without an LLM. 300 questions, 75 per hop count, top 5 sentences, bge for the dense system, one world. Intervals are 95% Wilson.

- **The final fact alone rarely wins.** A top 5 that holds the last hop's sentence but not the whole path happens for at most 5% of questions at any hop count (4 of 75, template text, dense, 4 hops) and for none at 1 hop. That sentence names the answer, so this is the cleanest shortcut, and it is small.
- **A guess from the schema gets about 4.5%.** That is the expected accuracy of picking uniformly among the entities that can fill the answer slot of the final relation (no documents).
- **The answer's name shows up in the top 5 more often than luck, but that is an upper bound, not a solver.** On the hard text at 2, 3 and 4 hops it appears for 17%, 19% and 13% of questions with BM25 and 28%, 16% and 15% with dense, against about 7% for five random sentences. (At 4 hops BM25's interval, 7–23%, still contains chance.) A reader would also have to pick that name out of five sentences, which this does not test.
- **So the low whole-path scores are not inflated by shortcuts we can measure**: whole-path@5 is 0–7% at 2 or more hops, and the cheap routes to the answer are mostly closed. The same holds on template text.
- **Not done:** the MuSiQue test proper, a single-hop *reader* (an LLM given the top 5 and asked for the answer), and a second world. All 300 questions are `entity`; set, count and compare questions need their own shortcut definition.

**Questions now say when their path touches a shared name** (`shared_name` in the gold record, [ADR-0013](adr/0013-question-gold-schema.md); the committed `hard-tiny` and `hard-small` question files carry it and nothing else in them changed). The critique report splits on it ([docs/results/critique-hard-tiny.md](results/critique-hard-tiny.md)), and the split reproduces what I found by hand: 23 of 37 grounded when it is set, 22 of 23 when it is not. It is set for 37 of the 60 `hard-tiny` questions and 177 of the 300 `hard-small` ones (59%), so on the `hard-small` world most questions pass through an entity that shares a name. That is how the generator builds its linking case, but it means a score over all questions is mostly a score over those. The retrieval and extraction reports are not yet split on the flag.

**Do the documents say what the world says? A faithfulness check** (`kgbench faithfulness --dir datagen/examples/hard-small`; [docs/results/faithfulness-hard-small.md](results/faithfulness-hard-small.md), with a chart, and every verdict beside it). Haiku, through semula, reads each of the 145 `hard-small` articles against its facts and answers STATED, WRONG or MISSING per fact (quoting the article's words), then lists clauses that add detail about an entity beyond the facts. 175 replies, committed and replayable offline: 145 documents plus 15 articles with a fact's sentence removed and 15 with a fact reversed in the list, to see whether the judge notices.

- **Facts:** 326 of 328 listed facts judged STATED (99.4%, interval 98–100%), 2 MISSING, 0 WRONG.
- **Unlisted claims:** 4 of 145 documents (2.8%, 1–7%). The model's own list: "a product that buyers often compare with similar equipment", "which has drawn attention from several buyers", "The same person has also worked for Luse Group and Suvu Doveku", "Life there moves with the tides and the seasons, which shapes how people plan their weeks". I read all four documents. Three are what the judge said. The fourth (D00050) is not: the quoted clause, "The same person has also worked for Luse Group and Suvu Doveku", is a sentence that states two listed facts (F130, F131), so that report is a false positive; the same document does contain the unlisted "at another point", which is the one that was reported earlier. So by my reading 4 of 145 documents have an unlisted claim, found by the judge in 3 of them and one by an earlier read; treat 4 as a floor.
- **The judge catches removed facts but not reversed ones.** A removed fact was called MISSING in 14 of 15 (70–99%). A reversed fact was called WRONG in **0 of 15** (0–20%): it quoted "Dako produces Vipu-X55" as stating "Vipu X55 produces Dako Works". So "0 WRONG" says nothing about relation direction.
- **A run before the final prompt was discarded.** The first prompt gave 99.4% stated, no unlisted claims at all (although I knew of three) and the same 0 of 15 on reversals. The final prompt (quote the words, check direction, go clause by clause) found unlisted claims; it did not fix reversals. I kept only the second run; the first replies are not in the repo.
- **What covers direction instead is code:** of the 290 sentences that state exactly one fact, none names the object before the subject, and 6 do not name the subject in the sentence (five say "the firm also produces X"; one, "Gaze Industries is affiliated with Gaze Industries.", is a pair of entities that share a name). Passive or reversed wording would fail this, but it checks order, not the relation word, and says nothing about sentences that state several facts.
- **Caveats:** one model family wrote the articles and judged them; the judge failed one of its two controls; one world.
