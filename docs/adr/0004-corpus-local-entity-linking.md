# ADR-0004 — Corpus-local entity linking (no external knowledge base)

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

Entity linking maps each doc-level entity (a coref cluster) to a stable,
corpus-wide identity so the same real-world entity from different documents
becomes one graph node — the mechanism that enables multi-hop reasoning across
disjoint documents. Options considered:

- **External KB (Wikidata/Wikipedia):** link mentions to canonical public IDs.
  Strong disambiguation, but needs a local dump or live API and adds coupling.
- **Corpus-local store:** the corpus is its own authority; match each mention to
  an entity already discovered in the corpus, or create a new one.
- **Hybrid:** both.

The demo does not need external-world grounding; it needs internally consistent
entities across its own corpus.

## Decision

Use a **corpus-local entity store** with **no external knowledge base**. Entity
linking is entity resolution against the `ES-Entities` index:

1. **Block** candidate matches by entity type + normalized name.
2. **Score** with embedding similarity using a **local sentence-transformer**
   over the mention-in-context.
3. **Merge** into the existing canonical entity above a confidence threshold;
   otherwise **create a new canonical entity** — this create-new path is the
   normal, always-on behavior (every genuinely new entity is created).
4. **Optional, gated LLM tie-breaker** for borderline matches — **off by
   default** to conserve credits.

The **canonical entity ID is the merge key / node identity** for the knowledge
graph. Per-document EL results are stored with the document in `ES-Documents`;
the deduped canonical entities live in `ES-Entities`
(see [ADR-0005](./0005-elasticsearch-index-split.md)).

## Consequences

- NER and core EL cost **$0** in API tokens (both local); the LLM tie-breaker is
  opt-in. This supersedes the original REQS assumption that EL uses the LLM.
- Disambiguation ("Apple Inc." vs "Apple" the fruit) is **heuristic** (type +
  context embeddings) and **order-sensitive** — the first document mentioning an
  entity seeds its canonical record. Acceptable for a demo.
- No external dependency, fully offline-capable, no KB licensing/version
  concerns.
- Entity-resolution quality is a tunable threshold, a natural thing to inspect
  in the frontend and to evaluate in benchmarking.
