# ADR-0006 — Knowledge-graph builder & Neo4j graph model

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

The KG-builder turns a processed document into graph structure. Open questions:
what predicate vocabulary to use, how nodes are identified across documents,
what the Neo4j node model looks like, and how provenance is recorded.

### Predicate vocabulary — open vs. closed

- **Open** (LLM emits free-text predicates): flexible, captures nuance, no
  schema upfront. But noisy — `works for`, `employed by`, `is employed at`
  become three edge types; hard to query and to benchmark consistently.
- **Closed** (fixed relation set): clean, queryable, consistent multi-hop paths,
  easy to evaluate. But rigid and may drop relations that don't fit.

## Decision

**Schema-guided extraction with a closed predicate set + open fallback.** The
LLM (default `gpt-4o-mini`) receives the document text **and that document's
canonical linked entities**, and emits triples `(subject_id, predicate,
object_id)` referencing **canonical entity IDs** (not raw strings). It maps each
relation to the closest of a curated set; if nothing fits it emits `RELATED_TO`
and stores the model's original phrase as an edge property `raw_predicate` — so
nothing is lost.

Starter closed set (~12, extensible): `LOCATED_IN`, `PART_OF`, `MEMBER_OF`,
`WORKS_FOR`, `HAS_ROLE`, `FOUNDED`, `OWNS`, `PRODUCES`, `PARTICIPATED_IN`,
`OCCURRED_ON`, `AFFILIATED_WITH`, `RELATED_TO` (fallback).

**Node identity / merge key:** one Neo4j node per **canonical entity ID** (from
corpus-local EL, [ADR-0004](./0004-corpus-local-entity-linking.md)). This is the
mechanism that lets the same entity from different documents become one node —
i.e. what makes cross-document multi-hop reasoning possible. Alternatives
rejected: one node per doc-level mention (graph stays fragmented) or merge by
surface string (conflates homonyms, splits aliases).

**Node model:** multi-label — a shared `:Entity` label **plus** a type label
(`:Entity:Person`, `:Entity:Organization`, …). Query all entities via `:Entity`,
or one type via its label; idiomatic Neo4j with per-label indexes. Properties:
`canonical_id`, `name`, `type`, `aliases`.

**Which entity types become nodes vs. attributes:** `PERSON`, `ORG`,
`LOCATION`, `EVENT`, and `PRODUCT` are **first-class nodes**. `DATE` is modeled
as an **edge attribute / qualifier** (e.g. an `OCCURRED_ON` property or date
qualifier on the relevant edge), **not** a standalone node — dates as nodes
clutter the graph and hurt multi-hop readability. A lightweight date node is
introduced only if a question genuinely needs to hop *through* time. (This
refines the extracted-type set of [ADR-0002](./0002-local-ner-with-spacy.md):
`DATE` is still extracted, just modeled as an attribute.)

**Provenance (Q21):** every edge records `source_doc_id`, `sentence_index`, the
`source_sentence` text, `raw_predicate`, and a `confidence`. The LLM only cites
the **sentence index** for each triple; the exact `char_start`/`char_end` are
resolved by **our own spaCy sentence segmentation** (ADR-0002), not produced by
the LLM (which cannot count characters reliably). Answers can therefore cite the
exact sentence they came from, with trustworthy offsets.

## Consequences

- Clean, benchmark-friendly primary edges without losing rare relations.
- Grounding triples in canonical IDs (not strings) keeps the graph consistent
  with the EL store and enables reliable multi-hop traversal.
- Multi-label nodes give both uniform and per-type queries.
- Edge-level provenance is load-bearing for trustworthy, traceable answers.
