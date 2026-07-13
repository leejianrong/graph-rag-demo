# QUESTIONS — Graph RAG Demo (grilling log)

Working log for the grilling step of `build-plan-product`. Questions are dumped
up front and added as we go, so you can answer several per turn.

## Legend

- **Priority** — `P0` blocks the architecture (must resolve before PRD) ·
  `P1` important, shapes a stage · `P2` nice to pin down, can default.
- **Status** — `OPEN` · `ANSWERED` · `DEFERRED` · `ASSUMED` (I picked a
  sensible default; correct me if wrong).

Once answered, decisions graduate into `CONTEXT.md` (glossary + decision
register) and, where architectural, into `docs/adr/*.md`.

---

## A. Pipeline orchestration & topology (P0)

- Q1 - Let's make the pipeline (read -> NER -> coref -> EL -> KG build) a single consumer that runs all stages in process for a document. However, in terms of the code, I want it to be modular and these functional stages should be organized accordingly in the code.
- Q2 - NA I believe
- Q3 - Only the bucket name and object key travels on the Kafka message to kick-off the process for a single file.
- Q4 - Error handling - log and drop is fine for the local demo
- Q5 - Let's go with what you recommend

## B. Document handling & chunking (P0)

- Q6 - Let's go with what you recommend
- Q7 - Recommend size of document so it's appropriate for testing (balance building a substantial knowledge graph but not tearing through tokens; help me estimate how many tokens the testing will consume too). For testing and the demo, maybe just tens or hundreds of docs. However, in the future (future iterations, out of scope), I want to be able to handle tens / hundreds of thousands of docs, so keep the implementation flexible enough to be upgraded to that in the future.
- Q8 - Let's do English only 

## C. Named Entity Recognition (P0)

- Q9 - Yes, local NER please, and recommend how we should do it
- Q10 - Let's go with what you recommend
- Q11 - Again, Let's go with what you recommend. Explain your recommendations so I can learn

## D. Coreference resolution (P1)

- Q12 - Yes, coref scope is within a single document.
- Q13 - Let's do a cluster map

## E. Entity Linking (P0 — biggest open area)

- Q14 - Corpus local entity store - The ES Entities index is the set of entities we've discovered so far in the ingested corpus, and EL means 'match this mention to an existing corpus entity, or create a new one'
- Q15 - Yes, I confirm that the per-document EL result belongs in the document result in Elasticsearch-Documents, and the deduped canonical entities live in Elasticsearch-Entities
- Q16 - No need to link to Wikidata.
- Q17 - Yes, I would like to create a new unlinked entity keyed within the corpus (if a mention can't be confidently linked). However, this is a nice to have, and I would like it to be implemented later on / gated (able to switch on and off) if possible.

## F. Knowledge-graph builder (P0)

- Q18 - Help me weigh the options and give examples of both including tradeoffs, but ok to go with what you recommend
- Q19 - Can you make a recommendation again? Explain the options and help me weigh them
- Q20 - Yes, KG-builder should read the document text and the document's linked entity clusters so it can ground triplets in real entity IDs.
- Q21 - Yes, for provenance let's make each triple/edge record which document and sentence/offset it came from. 
- Q22 - Make a recommendation and justify it.

## G. Graph RAG query / retrieval side (P0 — scope question)

- Q23 - Yes, you're absolutely right. Thanks for catching that. I want benchmakring, so I want a question -> answer path. (is it possible to get an answer without an LLM?)
- Q24 - I would like to visualize the graph, but for question and answer, let's go with a method that doesn't require an LLM for now (I'm assuming the vector-anchored + graph expansion doesn't need an LLM)
- Q25 - Can you make a recommendation? I'm leaning towards yes for using Elasticsearch's vector capabilities.
- Q26 - Yeah, quert interface can be a REST API endpoint. However, keep in mind that I still want the kafka message consumer to be the first stage in the pipeline.


## H. LLM usage & cost control (P1)

- Q27 - Yes, absolutely interested in cheaper models like GPT-4o-mini or even Deepseek API / other APIs that are more cost effective.
- Q28 - Yes, please cache LLM responses keyed by prompt hash so re-running pipelines / benchmarks doesn't burn credits.
- Q29 - Let's go with your recommendation

## I. Benchmarking (P1)

- Q30 - Is it possible to do both Retrieval/answer quality, and pipeline extraction quality (NER/EL correctness)? Please provide more advice
- Q31 - Help me compare the datasets you suggested and weigh them so i can decide better
- Q32 - Advice me on how people usually evaluate these graph RAG pipelines; I'm leaning towards non-LLM methods
- Q33 - Yes, agree with all your recommendations.

## J. Frontend & visualization (P2)

- Q34 - Let's have the frontend as a later add on
- Q35 - leave for later. Out of scope for v1

## K. Non-functional / cross-cutting (P1)y

- Q36 - Yes, two indices in a single ES container is fine.
- Q37 - I don't understand the question, but yes, OpenAI key and service endpoints in .env makes sense
- Q38 - Basic logging is enough for the demo, but keep it flexible so we can add structured logging next time.
- Q39 - I imagine there will be an API service that I can hit, and this service will upload to MinIO and then publish to Kafka. Maybe we can have a simple FastAPI server for this?

---

## Answered / decisions (moves to CONTEXT.md as we go)

_(nothing yet)_
