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

There's a visual walkthrough of the whole idea at
**<https://leejianrong.github.io/graph-rag-demo/>**.

## Status

The pipeline is built and runs end to end. It came together in eight vertical
slices, from the ingest-and-store skeleton through NER, coreference, entity
linking, graph building, and retrieval, to a small benchmark harness. All eight
have landed. You can bring the whole stack up with one command, feed it
documents, and query the resulting graph; jump to [Running it
locally](#running-it-locally) for that.

The design was written down before the code, and it's worth reading if you want
the reasoning rather than just the moving parts. The [PRD](docs/PRD.md), the ten
[ADRs](docs/adr/), and the [slice plan](docs/SLICES.md) cover it. Where a doc and
the code disagree, trust the code.

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
   stage fetches that file from MinIO. The raw text is stored in Elasticsearch
   up front, keyed by a deterministic document ID, before any processing runs.
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
   external KB. The corpus is its own authority, and linking is order-sensitive:
   the first document to mention an entity seeds its record.
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

## Running it locally

You'll need Docker with Compose. For the LLM-backed stages you'll also need a key
for any OpenAI-compatible provider; the query path itself runs without one.

```bash
git clone https://github.com/leejianrong/graph-rag-demo
cd graph-rag-demo
cp .env.example .env          # then set OPENAI_API_KEY — see below
docker compose up --build     # or: make up
```

The first build is large and slow. The image installs PyTorch and downloads
spaCy's transformer model and the bge embedding model, which runs to a few
gigabytes; after that it's cached. Compose brings up Kafka, MinIO, Elasticsearch,
Neo4j, and the FastAPI service on port 8000. The MinIO console is on 9001 and the
Neo4j browser on 7474 if you want to poke at the stores directly.

Ingest a document with a multipart upload:

```bash
curl -F "file=@some-report.md" http://localhost:8000/ingest
# {"document_id": "...", "bucket": "documents", "object_key": "some-report.md"}
```

Ingestion is asynchronous. `/ingest` writes the file to MinIO, publishes a Kafka
trigger, and returns straight away; the consumer then runs the five stages. Follow
it with `make logs` (which is `docker compose logs -f app`).

Once a few documents are in, query the graph. This call makes no LLM request:

```bash
curl -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "How is Supplier A connected to the German supply-chain law?"}'
```

You get back the connected subgraph, the supporting sentences with their
provenance, and the top-ranked entity as the predicted answer. Add
`"synthesize": true` to also get an LLM-written prose answer grounded in that same
evidence; that one does call the model.

### The bundled demo

If you'd rather not assemble a corpus by hand, there's a ready-made one under
[`examples/supply-chain/`](examples/supply-chain/) that reproduces the Supplier-A
story from the top of this README: three short documents where no single file
connects a company to the law that governs it, but the graph does, through the
city they share.

```bash
make demo-offline   # no Docker, no API key — deterministic, $0
make demo-live      # one command: brings the stack up, then runs the real demo
make demo           # the real stack over HTTP (needs a stack already up via `make up`)
```

All three ingest the same files and answer *"How is Aurelia Components connected to
the German Supply Chain Act?"*, then print the two-hop connection, the subgraph, and
the supporting sentences. `make demo-offline` runs the whole pipeline in-process
with heuristic stages instead of the LLM, so it needs nothing installed beyond the
Python deps and every edge is a generic `RELATED_TO`.

`make demo-live` is the self-sufficient real-stack path: it runs `docker compose up
-d --build --wait`, blocking until every service — including the app's own
healthcheck — is ready, then ingests and queries. You need Docker and an
`OPENAI_API_KEY` in `.env` (coreference and graph-building call an LLM). It leaves
the stack **up** afterwards, so you can browse Neo4j at `localhost:7474`, re-run the
query cheaply with `make demo`, or stop everything with `make down`.

`make demo` is the fast inner loop: it drives a stack you already brought up (via
`make up` or `make demo-live`) without touching containers, so re-runs take seconds.
Edges are typed on both real-stack paths, and `SYNTHESIZE=1` (e.g. `make demo-live
SYNTHESIZE=1`) adds an LLM prose answer. It's the fastest way to see what the
pipeline is for before pointing it at your own documents.

### What to watch for

A few things trip people up the first time:

- **The graph needs an API key.** Coreference and graph-building go through an LLM.
  With no key set, those stages raise, and the pipeline logs the error and drops
  the document after its raw text is already stored. You end up with documents in
  Elasticsearch and an empty graph, which looks like nothing happened. Set
  `OPENAI_API_KEY` in `.env`, or point `COREF_MODEL` / `KG_BUILD_MODEL` at another
  OpenAI-compatible endpoint (DeepSeek included) and set that provider's key.
- **Multi-hop needs more than one document.** The whole point is connections that
  cross documents, so feed it a handful of related files rather than one. And
  because linking is order-sensitive, the ingestion order is part of the result.
- **Re-running is free.** Every LLM call is cached by a hash of the model, prompt,
  and parameters, so re-ingesting the same corpus or repeating a query costs
  nothing the second time.
- **Elasticsearch wants memory.** It runs single-node with security off for local
  use. If the container keeps exiting, it usually needs more RAM given to Docker.

### Tests

The fast suite runs against in-memory fakes: no Docker, no model, no API key. It's
the gate to run before pushing.

```bash
uv sync --frozen --extra dev
uv run pytest -m "not contract and not model and not llm and not benchmark"  # make test
```

The slower layers are opt-in and split by pytest marker: `contract` proves each
real adapter against a throwaway container (MinIO, Elasticsearch, Kafka, Neo4j),
`model` loads real spaCy and embedding models, `llm` hits a real provider and skips
cleanly without a key, and `benchmark` runs the whole pipeline over a fixture.
[CLAUDE.md](CLAUDE.md) has the full command list and the reasoning behind the
layering.

## Benchmarking

The pipeline is evaluated on **2WikiMultihopQA**, whose evidence is expressed as
`(entity, relation, entity)` reasoning paths. That's a close match for what the
graph builds, which makes it a fair test of both construction and retrieval. A
fixed subset keeps runs quick, and the harness is a console command:

```bash
uv run benchmark run --subset small --dataset path/to/2wiki.json
# or, over the in-repo mini fixture: make benchmark
```

`make benchmark` runs the same thing against a small bundled fixture with no
arguments, so you can see the scorecard immediately; pass `DATASET=path/to/2wiki.json`
for the real corpus or `REAL=1` for the real adapter stack.

The metrics are deliberately non-LLM: supporting-fact precision, recall and F1
for whether retrieval surfaced the gold evidence, plus exact-match and token-F1
on the answer, scored against each node's name and its aliases under standard
normalisation. Corpus-local entity linking is order-sensitive, so benchmark runs
fix the ingestion order and the linking thresholds to stay reproducible. A warm
run reuses the pre-built graph and the response cache, so it costs about nothing.

## Where the design lives

The thinking is all written down, and it's worth reading before any code:

- **[docs/PRD.md](docs/PRD.md)** — the product requirements: problem, user
  stories, implementation and testing decisions, scope.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — topology, the six ports, the
  data model, and the stage/checkpoint flow, consolidated from the ADRs.
- **[docs/adr/](docs/adr/)** — ten architecture decision records, one per major
  choice, each with the alternatives weighed and the reasoning kept.
- **[docs/SLICES.md](docs/SLICES.md)** — the eight vertical slices the build
  followed, each with its own test plan.
- **[docs/CONTEXT.md](docs/CONTEXT.md)** — the glossary and decision register.
  Start here if a term is unfamiliar.

## Conventions

Python 3.12, dependencies managed with [uv](https://github.com/astral-sh/uv),
modular layout with type hints and docstrings throughout. English-only corpus.
No authentication and no multi-tenancy: this is built to run on your laptop, not
to be deployed.
