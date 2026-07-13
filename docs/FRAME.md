---
shaping: true
---

# FRAME — Graph RAG Demo

> The "why" behind this work, at stakeholder level. **Source** is the raw
> material verbatim; **Problem** and **Outcome** are distilled from it.
> Settled decisions live in [`PRD.md`](./PRD.md), [`CONTEXT.md`](./CONTEXT.md)
> (glossary + D1–D9), and [`adr/0001–0009`](./adr/) — this frame does not
> re-open them.

---

## Source

Verbatim from [`REQS.md`](./REQS.md) — the raw idea, problem, audience, and
non-goals as originally stated.

> ## The idea
>
> I would like to make a graph RAG pipeline. I want the pipeline to ingest text
> (like markdown files, txt files) and do named entity recognition, coreference
> resolution, and entity linking. At the end, I want the result to be a
> knowledge graph.
>
> ## Why / the problem
>
> I am monitoring large amounts of data coming in from news sources, documents,
> etc. I am trying to identify connections between entities like places and
> people from these sources and try to look for patterns in the data. I need a
> knowledge graph that allows me to do multi-hop reasoning, connect disjointed
> documents, and find complex, indirect relationships.
>
> ## Who it's for
>
> For myself, I am trying to showcase that I can ingest a large amount of data
> (assume it comes in as markdown or plain text) and build a knowledge graph out
> of it. I am a software engineer and trying to learn more about Graph RAG, so
> this is educational for me too. Ultimately, I want to be able to run the
> pipeline locally using Docker compose, and to benchmark the capability of this
> graph RAG pipeline using some benchmarks.

The full "What it should do" walkthrough (Kafka trigger → read from object
storage → NER → coreference → entity linking → Graph RAG / KG-builder), the
external open-source services, the nice-to-haves, and the explicit non-goals
are recorded in [`REQS.md`](./REQS.md) and have since been settled into
[`PRD.md`](./PRD.md) and the ADRs. Key non-goals from the source, verbatim:

> ## Explicitly NOT doing
>
> - No authentication
> - No self hosting of models; use external API for LLM.
> - Supporting multi users / deploying this publicly (yet), I just want to be
>   able to run the whole thing locally in docker compose.

> ## Constraints / notes
>
> - Make sure the code is modular
> - Use python 3.12 for this project
> - Use uv to manage dependencies
> - Use Svelte + Vite if building any frontend
> - Follow coding best practices (docstrings for functions and classes, type
>   hints, etc.)

---

## Problem

I monitor large volumes of unstructured text — news articles, documents — and
the connections I care about **span multiple documents**. The questions worth
asking are **multi-hop** and **cross-document**: "how is X connected to Y?",
where the link only exists by chaining facts that live in *separate* files.

- **Keyword / single-document search can't see the chain.** It finds a passage
  mentioning X and a passage mentioning Y, but never the path between them.
- **Standard vector RAG retrieves passages, not relationships.** It surfaces
  individually-relevant chunks and misses the connected structure, so the big
  picture — the indirect, multi-hop relationship — is lost.
- **The entities are scattered and un-reconciled.** The same real-world person,
  place, or organization appears under different surface forms across documents
  (and via pronouns within one), so there is no single thing to reason *about*
  until those mentions are unified.

Separately, I'm a software engineer learning Graph RAG, so a black box is not
enough: I need a system that is **inspectable**, **runs entirely on my own
machine**, and whose multi-hop QA capability I can **measure with standard,
reproducible metrics** — without an LLM bill that grows every time I re-run.

## Outcome

Success looks like a working, locally-runnable system that turns a pile of
unstructured text into a queryable knowledge graph and answers the multi-hop
questions the source problem is about — cheaply, reproducibly, and
transparently.

Concretely, I have succeeded when:

- **Ingest is one gesture.** I drop a Markdown/plain-text document into local
  storage, and the system extracts its entities, unifies mentions that refer to
  the same thing (within *and* across documents), and folds the facts into a
  persistent knowledge graph — with no manual wiring per document.
- **Multi-hop questions get connected answers.** I can ask "how is X related to
  Y?" and get back the *connected path* through the graph plus the specific
  source sentences that support it — not a bag of separately-matching passages.
- **Every answer is traceable.** Each fact in the graph points back to the exact
  document and sentence it came from, so I can inspect *why* an answer was
  produced and trust (or challenge) it.
- **It runs on my machine, end to end.** The whole stack comes up locally with a
  single command; nothing depends on a hosted service I have to deploy or
  authenticate to.
- **Capability is measured, not asserted.** I can run a standard multi-hop QA
  benchmark over the system and report reproducible numbers for how well it
  retrieves the right evidence and answers correctly.
- **Re-running is effectively free.** The expensive parts are cached and the
  cheap parts are local, so iterating on the pipeline and re-running the
  benchmark does not burn API credits.

*(Success is deliberately stated as capability, not mechanism. The "how" —
spaCy NER, corpus-local entity linking, Neo4j, the non-LLM retrieval path,
2WikiMultihopQA, response caching — is already settled in the PRD and ADRs and
is not re-litigated here.)*
