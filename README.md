# Graph RAG Demo

A local Graph RAG pipeline that turns a pile of text and Markdown files into a
knowledge graph you can ask multi-hop questions against. Drop a document into
object storage, and a Kafka-triggered pipeline reads it, pulls out the people,
organisations, places and events, works out which mentions refer to the same
thing, links those to a corpus-wide identity, and writes the relationships into
Neo4j. Then you query the graph.

It runs entirely on one machine through Docker Compose. There's no external
knowledge base, no cloud dependency beyond an optional LLM API, and the default
query path never calls an LLM at all.

**Status: design complete, implementation not started.** This repo currently
holds the plan, not the code. The [PRD](docs/PRD.md) and nine
[ADRs](docs/adr/) pin down every decision; the pipeline itself is the next step.
Read [where the design lives](#where-the-design-lives) below to navigate it.

## The problem

Say you're watching a stream of news articles and reports, and the connection
you care about isn't stated in any single document. Supplier A is mentioned in
one file as being based in Berlin. Another file, written months earlier by
someone else, notes that a German supply-chain law applies to companies in
Berlin. No one document links the supplier to the law, but the graph does, in
two hops.

Ordinary vector RAG struggles here. It retrieves the paragraphs that mention
your search terms and hands them to a model, but the paragraphs that matter
might share no keywords with your question or with each other. The link lives in
the *structure* between documents, which is exactly what a knowledge graph
captures and a bag of text chunks throws away.

## How it works

Ingestion is one in-process consumer that runs five stages for each document,
in order. Keeping it in a single process was a deliberate choice: for a
local demo, being able to trace one document through the whole flow in one
stack trace beats the independent scaling you'd get from wiring five services
together with intermediate Kafka topics. Each stage is still a separate,
swappable module, so the microservices version stays possible later without a
rewrite.

1. **Read.** A Kafka message carries a bucket and object key, nothing more. The
   stage fetches that file from MinIO.
2. **NER.** spaCy extracts typed mentions locally: people, organisations,
   locations, dates, events. This runs on your CPU and costs nothing per token,
   which matters because it touches every word of every document.
3. **Coreference.** An LLM groups the mentions within a single document that
   refer to the same thing, pronouns included, and returns a cluster map rather
   than rewriting the text. The original characters stay put, so provenance
   offsets stay valid.
4. **Entity linking.** Each cluster is matched against a corpus-local store of
   canonical entities: blocked by type and normalised name, scored by embedding
   similarity, then either merged into an existing entity or created as a new
   one. This is the step that lets "Angela Merkel" in one file and a later
   mention in another become a single graph node. There's no Wikidata, no
   external KB. The corpus is its own authority.
5. **Knowledge-graph build.** An LLM reads the document and its linked entities
   and emits subject–predicate–object triples grounded in canonical entity IDs.
   Predicates map to a closed set of about twelve relations (`LOCATED_IN`,
   `WORKS_FOR`, `FOUNDED`, and so on) with a `RELATED_TO` fallback that keeps
   the model's original phrasing so nothing is lost. Every edge records where it
   came from: source document, sentence, and the sentence text itself.

Dates are the one type that doesn't become a node. A graph full of date nodes
turns into a hairball, so a date rides along as an attribute on the edge it
qualifies instead.

## Answering without an LLM

Retrieval and synthesis are two different jobs, and conflating them is what
makes most Graph RAG demos quietly expensive. Finding the right entities and
evidence for a question can be done with embeddings and graph traversal alone.
Only turning that evidence into fluent prose actually needs a language model.

So the default query path uses no LLM and costs nothing. It embeds your
question, finds seed nodes by vector search over entity and passage embeddings
in Elasticsearch, expands a few hops out in Neo4j, and returns a ranked subgraph
with the supporting sentences and their provenance. For a question whose answer
is an entity, the answer is the top-ranked node.

That path answers "who" and "where" questions well and explanatory ones poorly,
which is an honest limitation rather than a bug. When you want prose, an
optional synthesis mode feeds the retrieved subgraph to an LLM. It's gated off
by default.

## The stack

Everything comes up together under Docker Compose:

- **Kafka** — the ingestion trigger
- **MinIO** — S3-compatible object storage for the source files
- **Elasticsearch** — one cluster, two indices: documents (with their
  processing results and passage vectors) and canonical entities (the dedup
  store, which doubles as the vector index for entity linking and query anchoring)
- **Neo4j** — the knowledge graph
- **FastAPI service** — two endpoints: one to upload a file and publish the
  Kafka trigger, one synchronous `/query` for read-only retrieval

The LLM is reached through a provider-agnostic client, so the model is a config
choice per stage. The default leans on `gpt-4o-mini` for the high-volume
extraction work and reserves a fuller model for optional synthesis. Any
OpenAI-compatible endpoint, DeepSeek included, swaps in through `.env`. Every
call is cached by a hash of the model, prompt and parameters, which is what
makes re-running the pipeline or the benchmark cost nothing the second time.

## Benchmarking

The pipeline is evaluated on **2WikiMultihopQA**, whose evidence is expressed as
`(entity, relation, entity)` reasoning paths. That's a close match for what the
graph builds, which makes it a fair test of both construction and retrieval.
A fixed subset of 100 to 200 questions keeps runs quick.

The metrics are deliberately non-LLM: supporting-fact precision, recall and F1
for whether retrieval surfaced the gold evidence, plus exact-match and token-F1
on the answer, scored against each node's name and its aliases under standard
normalisation. Corpus-local entity linking is order-sensitive, since the first
document to mention an entity seeds its record, so benchmark runs fix the
ingestion order and the linking thresholds to stay reproducible.

## Where the design lives

The thinking is all written down, and it's worth reading before any code:

- **[docs/PRD.md](docs/PRD.md)** — the product requirements: problem, user
  stories, implementation and testing decisions, scope.
- **[docs/adr/](docs/adr/)** — nine architecture decision records, one per major
  choice, each with the alternatives weighed and the reasoning kept.
- **[docs/CONTEXT.md](docs/CONTEXT.md)** — the glossary and decision register.
  Start here if a term in the PRD is unfamiliar.
- **[docs/REQS.md](docs/REQS.md)** — the original idea, before any grilling.
- **[docs/QUESTIONS.md](docs/QUESTIONS.md)** and
  **[docs/ANSWERS.md](docs/ANSWERS.md)** — the full interview log behind the
  decisions.

## Conventions

Python 3.12, dependencies managed with [uv](https://github.com/astral-sh/uv),
modular layout with type hints and docstrings throughout. English-only corpus.
No authentication and no multi-tenancy: this is built to run on your laptop, not
to be deployed.
