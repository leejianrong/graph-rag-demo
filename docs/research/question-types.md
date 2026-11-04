# Question types and gold schemas for KG / GraphRAG / agent-memory evaluation

Method: I fetched and grepped the full text of most papers via arXiv PDFs. Anything marked "unverified" came from memory or from search snippets only.

## 1. Catalogue of benchmarks

| Benchmark | Question types | Grading |
|---|---|---|
| 2WikiMultihopQA ([arXiv 2011.01060](https://arxiv.org/abs/2011.01060)) | Four types: comparison, inference, compositional, bridge-comparison. Questions are generated from templates and logical rules over Wikidata and text. Each question ships "evidence" as a set of triples giving the reasoning path. | Answer EM/F1, supporting-fact F1, and evidence-triple generation. This is closest to our gold-path design. |
| MuSiQue ([2108.00573](https://arxiv.org/abs/2108.00573)) | About 25K questions of 2-4 hops, composed from single-hop questions. Filtered to resist disconnected reasoning (a single-hop model drops about 30 F1). Adds paired unanswerable contrast questions (MuSiQue-Full, about 50K). | Answer F1, support F1, and answerable-vs-unanswerable contrast scoring. |
| HotpotQA | Bridge and comparison questions. I did not fetch this one. | Unverified here (EM/F1 plus supporting-fact F1, from memory). |
| MultiHop-RAG ([2401.15391](https://arxiv.org/abs/2401.15391)) | Inference, Comparison, Temporal, and Null queries, built on news. Null means unanswerable and the system should return a null response. LLM-generated from bridge entities. | Short-answer match against ground truth, plus retrieval metrics. |
| CRAG ([2406.04744](https://arxiv.org/abs/2406.04744)) | 4,409 questions in 5 domains and 8 categories. The categories include conditions, comparison, aggregation, multi-hop, set queries, post-processing-heavy, and false-premise (12% of questions). Entity popularity and temporal dynamism are varied. | Each answer is labelled perfect, acceptable, missing, or incorrect, scored 1, 0.5, 0, -1. "Truthfulness" is the mean score. Auto-eval first tries exact match, then falls back to an LLM judge (two judges). The paper reports F1 of 94.7% and 98.9% against human grading. |
| FRAMES ([2409.12941](https://arxiv.org/abs/2409.12941)) | Multi-source multi-hop questions that need integrating several Wikipedia pages. Baseline accuracy is about 0.40 without retrieval and 0.66 with a multi-step pipeline. | Question-type breakdown and grading method not verified. |
| RGB ([2309.01431](https://arxiv.org/abs/2309.01431)) | Four testbeds: noise robustness (varied noise ratio), negative rejection (no answer in the documents), information integration, and counterfactual robustness (documents contain errors). | Exact-match accuracy. Rejection rate. Error detection rate. Error correction rate. |
| FreshQA ([2310.03214](https://arxiv.org/abs/2310.03214)) | 600 questions: never-changing, slow-changing, fast-changing, and false-premise. | STRICT mode (all claims must be correct and current) and RELAXED mode (only the main answer counts). For false-premise questions the model must point out the false premise to get credit. Comparing the two modes is described as a way to measure hallucination. |
| CREPE and (QA)² ([CREPE 2211.17257](https://arxiv.org/abs/2211.17257), [(QA)² 2212.10003](https://arxiv.org/abs/2212.10003)) | CREPE has about 8.4K Reddit questions, about a quarter with a false presupposition, with the presupposition and its correction annotated. (QA)² has 602 search queries. Both rest on secondary snippets, so details are partly unverified. | CREPE: detect the presupposition and judge whether it is true. (QA)²: human acceptability (best model 59%). |
| SQuAD 2.0 ([1806.03822](https://arxiv.org/abs/1806.03822)) | 53,775 new unanswerable questions, written adversarially by crowdworkers with a plausible but wrong answer in the paragraph. | EM and F1 with a no-answer option. Scores are reported separately for HasAns and NoAns. A model scoring 86% F1 on SQuAD 1.1 got 66% on 2.0. |
| AbstentionBench ([2506.09038](https://arxiv.org/abs/2506.09038)) | 20 datasets with over 35K unanswerable questions: unknown answers, underspecification, false premises, subjective questions, outdated information. | Abstention recall, via an LLM judge. Reasoning fine-tuning lowered abstention by about 24% on average. |
| Microsoft GraphRAG ([2404.16130](https://arxiv.org/abs/2404.16130)) | Global sensemaking questions. An LLM is given a corpus description and produces K personas, then N tasks per persona, then M questions per persona-task pair. The questions must need whole-corpus understanding and must not need low-level facts. | There is no gold answer, so it uses head-to-head LLM-judge comparison on comprehensiveness, diversity, and empowerment, with "directness" as a control criterion. As a check it also uses claim-based measures: Claimify extracts claims (about 31 per answer), then counts them and clusters them. Reported win rates were 72-83% (comprehensiveness) and 62-82% (diversity). |
| GraphRAG-Bench ([2506.05690](https://arxiv.org/abs/2506.05690)) | Four levels: fact retrieval, complex reasoning, contextual summarise, and creative generation. | Evaluates the whole pipeline: graph construction, retrieval, and generation. Metrics are accuracy plus a faithfulness measure. |
| STaRK ([2404.13207](https://arxiv.org/abs/2404.13207)) | Retrieval over a semi-structured KB (text plus relations). Queries mix relational and textual constraints, with 9-14K synthesized queries per dataset. Many have multiple gold nodes. | Set retrieval: Hit@k, MRR, Recall@20 (metric names from the paper's text, not read in full). |
| LongMemEval ([2410.10813](https://arxiv.org/abs/2410.10813)) | 500 questions over 5 abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. The abstention items are made by modifying existing questions into false-premise ones. Histories run about 115K tokens (S) or 500 sessions and about 1.5M tokens (M). | A GPT-4o judge with per-question-type prompts. The paper reports over 97% agreement with human experts. |
| LoCoMo ([2402.17753](https://arxiv.org/abs/2402.17753)) | Up to 35 sessions per conversation. Question categories: single-hop, multi-hop, temporal, commonsense/world knowledge, and adversarial. Also event summarisation. | Not verified in detail. |
| GrailQA ([2011.07743](https://arxiv.org/abs/2011.07743)) and ComplexWebQuestions | GrailQA has up to 4 relations plus optional counting, superlative, or comparative functions. It has i.i.d., compositional, and zero-shot splits. I could not verify ComplexWebQuestions or KGQAn in this spike. | GrailQA: exact-match answer sets, F1 over answer sets. Metric names for GrailQA are from memory (unverified). |

## 2 and 3. Taxonomy, with programmatic feasibility

"Prog." means exact gold can be derived from the ground-truth graph with no LLM.

| Type | Example | Gold form | Prog.? | How graded |
|---|---|---|---|---|
| Single/multi-hop entity | "Where was the founder of Acme born?" | One entity ID plus the gold path of fact IDs | Yes | EM on ID or alias, plus path/support-fact F1 |
| Set/list | "Which cities host a branch of Acme?" | Set of entity IDs | Yes | Set precision/recall/F1 |
| Counting/aggregation | "How many employees did Acme have in 2019?" | Integer, plus the supporting set | Yes | Exact number, and the set if the system returns it |
| Comparison | "Who was born earlier, A or B?" | One of the two entities, plus the compared values | Yes | EM |
| Superlative | "Which Acme office is the oldest?" | Entity (plus tie handling) | Yes | EM. Emit all ties as gold. |
| Intersection | "Which people worked at both X and Y?" | Set | Yes | Set F1 |
| Negation/exclusion | "Which Acme employees are not in Berlin?" | Set | Yes, but closed-world only | Set F1. Wrongness risk: the facts are open-world, so a missing fact looks like a "not". Restrict to relation types the generator guarantees complete. |
| Temporal as-of | "Who led Acme in 2015?" | Entity valid at that year | Yes | EM |
| Temporal supersession / knowledge update | "Where does X work now?" after an update | Latest valid fact, with the superseded fact listed as forbidden | Yes | EM, plus a stale-answer rate |
| Retracted / deleted fact | "Where does X work?" after a deletion | Abstain, or the earlier fact marked as retracted | Yes | Abstain-correct, plus a stale-answer rate |
| Ambiguous homonym | "Where was John Smith born?" | Clarify, or list one answer per candidate entity | Yes (mapping) | Per-entity answers, or a clarification flag |
| False premise | "Why did Acme acquire Beta?" (no such fact) | Premise-false flag, the missing or contradicting fact, optional corrected answer | Yes | Premise-rejection rate, no invented answer |
| Unanswerable / no evidence | "Who is X's manager?" (not in the corpus) | Abstain | Yes | Abstain-correct rate, false-refusal rate |
| Path/explanation | "How are A and B connected?" | Set of valid paths as fact IDs | Yes | Path recall (any valid path counts), plus no unsupported edges |
| Conflicting sources | Two docs disagree on a date | The conflict itself: both values, plus the authoritative one if the generator defines one | Partly | Needs a rule for "authoritative". Report both sides or the resolved value. |
| Adversarial distractors | Near-duplicate name or alias with different facts | Same gold as the base question | Yes | EM, and the answer-from-distractor rate |
| Counterfactual | "If A had not joined B, who would lead C?" | No gold derivable from the data. Treat as unsupported, so the gold is abstain. | Mostly no | Abstain-appropriate, or LLM-judged |
| Multi-turn follow-up | "And his successor?" after a first question | Chain of programmatic questions with coreference to the prior answer | Yes | EM per turn |
| Open-ended synthesis | "Summarise the history of Acme's expansion" | Set of required facts plus forbidden claims | Partly | See below |

Where an LLM is needed, keep the gold exact by using it only to render text (for example, the question wording or document prose). Derive the gold from the graph, and let the LLM only paraphrase.

### Open-ended answers: gold as facts, not a string
- **Nugget method.** TREC's AutoNuggetizer uses an LLM to create nuggets and to assign nuggets to system answers, and compares results with human assessment (about 21 topics, 45 runs; [2411.09607](https://arxiv.org/abs/2411.09607)). A strict score counts vital nuggets only and a weighted score uses nugget importance. I took the scoring definitions from a secondary summary (unverified).
- **Claim-level scoring.** GraphRAG itself uses Claimify to split answers into atomic claims ([2404.16130](https://arxiv.org/abs/2404.16130)). Claim-level checking against the graph is the natural counterpart to our generator, since every true claim is a fact ID.
- **Our advantage.** The generator knows the graph, so the nuggets are programmatic: required fact IDs (with their rendered phrases) and forbidden claims such as superseded facts, distractor-entity facts, and false-premise assertions. Only the matching of a free-text answer to those facts needs an LLM or an entailment model, and that can be run in a cached, deterministic mode.
- **Measuring the judge.** Report judge-vs-human agreement on a small labelled set, as LongMemEval (over 97%) and CRAG (F1 94.7-98.9%) do. Prefer exact match first, with a judge only as fallback (CRAG's two-step approach).

## 4. Scoring false-premise and abstention
The standard quantities are:
- **Correct refusal / abstention recall.** The fraction of unanswerable or false-premise questions where the system abstains or rejects the premise. AbstentionBench uses this as its primary metric. RGB calls it rejection rate.
- **False-refusal rate.** The fraction of answerable questions where the system abstains. SQuAD 2.0 reports HasAns and NoAns scores separately, which makes this visible. Pair it with the correct-refusal rate so that refusing everything does not win. This pairing is my recommendation, not something I verified in a single source.
- **Hallucination rate.** Wrong non-refusal answers over all questions. CRAG scores an incorrect answer at -1 and a missing answer at 0, so truthfulness = mean(1, 0.5, 0, -1) rewards abstaining over guessing.
- **Premise identification.** FreshQA requires the answer to point out the false premise. Merely answering "no" does not count. CREPE and (QA)² additionally score whether the false presupposition is identified.
- **Extras.** RGB adds error detection and error correction rates for counterfactual documents.
- **Stale-answer rate.** For updated or retracted facts, count answers matching a superseded or retracted value. This is my suggestion; I did not find a standard name.

## 5. Long, realistic documents with exact gold
What I found, with no direct match for "KG facts to long text with verification":
- **RIKER** ([2601.08847](https://arxiv.org/abs/2601.08847)) generates documents from known ground truth instead of extracting ground truth from text, with a coherent simulated universe across document types. This is the nearest match in idea. I did not read it in full.
- **LV-Eval** ([2402.05136](https://arxiv.org/abs/2402.05136)) inserts confusing facts for interference and replaces keywords to limit knowledge leakage. It has lengths from 16K up to 256K.
- **MLRBench** ([2504.12845](https://arxiv.org/abs/2504.12845)) uses about 8,500 atomic facts as seeds for distractors, plus pure-filler tokens as a length control.
- **LongMemEval** builds long histories by embedding evidence sessions in filler sessions, at about 115K or 1.5M tokens per question ([2410.10813](https://arxiv.org/abs/2410.10813)). LoCoMo has up to 35 sessions per conversation. Neither is a pattern for 500-5,000-word documents; they are the "needle in filler" pattern.
- **Typical lengths.** I could not verify typical RAG test-corpus document lengths in this spike. The sources above suggest the long-context benchmarks are far longer (tens of thousands of tokens) than the 500-5,000-word target.

Techniques that suit our setup, proposed by me rather than taken from a paper:
1. **Discourse templates.** Group facts into paragraphs by entity or theme. Add structure such as a biography, a timeline, an organisational overview, a news item, or an email thread. Add connective sentences, with offsets recomputed after assembly.
2. **Filler without facts.** Add boilerplate and generic scene-setting that contains no entity mentions and no assertable relations.
3. **Distractor facts.** Use true facts about distractor entities, homonyms, and unrelated entities. These are already in the generator, and their facts are known to the gold.
4. **Fact-to-passage expansion by LLM.** Expand each fact into a passage with the optional cached LLM renderer, then verify. Verification means an entailment or extraction pass that finds every fact the passage asserts, and the passage is rejected if it asserts a fact not in the graph or omits a required one.
5. **Cached renders.** Keep renders deterministic by hashing prompt plus seed.

Risks:
- **Answer leakage.** A question's wording can reveal the answer, or an answer may be stated in the prompt-visible text near the question. Keep question text out of the rendering prompt.
- **Fact drift.** LLM expansion can change dates, swap names, or merge entities.
- **Accidental extra facts.** The prose invents a statement ("founded in the early 1990s") that contradicts or competes with the gold. This is the biggest risk for exact-gold evaluation, because it makes the gold wrong without anyone noticing. Verify with the extraction pass above.
- **Offset breakage.** Any edit to the text invalidates mention offsets, so compute them after the final text is assembled, not before.
- **Real-world knowledge contamination.** A model may answer from pretraining. Use invented names, as LV-Eval does with keyword replacement.

## 6. Recommendation

### Build first (7 types, 6 of them no-LLM)
1. **Set/list answers** (no LLM). It is the most direct fix for "too simplistic" single-entity answers and maps to STaRK and CRAG set queries.
2. **Unanswerable / no-evidence** (no LLM). Delete or never render the fact; the abstention is exactly checkable.
3. **False-premise** (no LLM). Substitute a wrong entity or relation into an otherwise valid question template. Gold: premise-false plus the real fact.
4. **Temporal as-of and supersession/retraction** (no LLM). It uses the existing valid_from/valid_to and update/delete events, and covers LongMemEval's knowledge-update and temporal categories.
5. **Counting, aggregation, comparison, and superlative** (no LLM). Numeric or small-set gold with exact match.
6. **Path/explanation** (no LLM). Gold is the set of valid paths, graded on path recall and unsupported edges.
7. **Open-ended synthesis**, graded as required facts plus forbidden claims (needs an LLM or entailment model only for matching, not for gold).

Ambiguous homonym questions and multi-turn follow-ups are next. Both are programmatic (a mapping from the existing homonym data, and chained questions). Counterfactual questions are last, because they have no derivable gold.

### Gold-answer schema (one record per question)
- `question_id`, `type`, `hops`, `as_of` (optional date), `seed`.
- `answer_kind`: one of `entity`, `set`, `number`, `abstain`, `false_premise`, `open_ended`.
- `answers`: list of entity IDs with accepted aliases and surface forms, or a number (with unit), or empty.
- `ordered`: boolean, and tie-handling rule for superlatives.
- `gold_paths`: one or more lists of fact IDs. Support facts also appear here for set/number answers.
- `abstain`: `{expected: bool, reason: "no_evidence" | "false_premise" | "retracted" | "ambiguous"}`.
- `premise`: for false-premise questions, `{asserted_fact, contradicting_fact_ids}`, and an optional `correction`.
- `required_facts`: fact IDs plus a weight (vital / okay), for open-ended answers.
- `forbidden_claims`: fact IDs or claim templates for superseded or retracted facts, distractor-entity facts, and the false premise.
- `stale_values`: values that were true earlier and must not be returned.
- `distractors`: entity IDs for homonyms and near-duplicates (used to compute the answer-from-distractor rate).
- `evidence_docs`: document IDs plus mention offsets for each gold fact.
- `grading`: `{primary: "exact" | "set_f1" | "numeric" | "abstain" | "nugget", judge: null | "llm"}`.

Report the answerable and unanswerable splits separately, with correct-refusal and false-refusal rates side by side, plus a CRAG-style truthfulness score.

## Not verified
- HotpotQA, ComplexWebQuestions, and KGQAn details.
- FRAMES question types and grading.
- LoCoMo scoring.
- STaRK metric definitions (the names came from a grep of the text, not a read of the evaluation section).
- Typical document lengths for RAG and graph-RAG corpora.
- AutoNuggetizer score definitions (from a secondary summary).
- Whether RIKER or SynClaimEval build long documents from KG facts.
