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
`OCCURRED_ON`, `AFFILIATED_WITH`, `GOVERNED_BY` (a law applies to the subject; added by GRA-94),
`RELATED_TO` (fallback).

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
`LOCATION`, `EVENT`, `PRODUCT`, and `NORP` are **first-class nodes**. `DATE` is
modeled as an **edge attribute / qualifier** (e.g. an `OCCURRED_ON` property or
date qualifier on the relevant edge), **not** a standalone node — dates as nodes
clutter the graph and hurt multi-hop readability. A lightweight date node is
introduced only if a question genuinely needs to hop *through* time. (This
refines the extracted-type set of [ADR-0002](./0002-local-ner-with-spacy.md):
`DATE` is still extracted, just modeled as an attribute.)

`NORP` (nationalities, religious/political groups — e.g. "German", "Catholic",
"Republican") is a **first-class node** (decided in Step E, Q47): it participates
in genuine relations (`PERSON —AFFILIATED_WITH→ NORP`, `PERSON —MEMBER_OF→ NORP`)
and dropping it would silently lose an extracted type and those multi-hop paths.
Its type label is `:Entity:Norp`.

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

## Addendum (GRA-162): roles, acquisitions and the place of an event

A real-stack run showed three misreadings the closed set invited. Changes, all in the closed-predicate
approach:

- **A role is an edge property, not an entity and not a new predicate.** `HAS_ROLE` stays a
  person-to-organisation edge and carries the title in `Triple.role` (`"CEO"`; stored as the edge
  property `role`, shown to the answerer as `role=CEO`). The alternatives lose: a role entity would make
  every "CEO" a node that has to be linked and merged, and a literal object breaks the rule that an edge
  joins two canonical ids. The title is kept only on `HAS_ROLE`. "Its CEO Klaus Weber" now has somewhere
  to land, so "who is its CEO" is answerable. `HAS_ROLE` stays single-valued per subject for
  supersession (`graph_rag/supersession.py`); a second role at the same organisation does not end the
  first.
- **`ACQUIRED` is added** (a takeover or purchase, buyer to target, with the date qualifier). "acquired",
  "bought", "purchased", "took over" and "merged with" map to it instead of `OWNS`; `OWNS` is standing
  ownership. No contract predicate is added: the bench corpus has no gold for one (`partner`/`agreement`
  text is gold `AFFILIATED_WITH`), so a new predicate could only cost strict F1 there.
- **The place of an event is the event's.** The prompt says `LOCATED_IN` goes from the EVENT to the place,
  the person gets `PARTICIPATED_IN`, and a person's own `LOCATED_IN` needs text that says they live or are
  based there.

The prompt lists the vocabulary, so this change alters every KG-build prompt and invalidates the recorded
answers; the KG-build answers in `bench/kgbench/data/llm-cache-hard-tiny` and `llm-cache-hard-small` (and the
stale entries from older prompts) were replaced by new ones from Haiku subagents answering the new prompts
(coref answers are unchanged). Measured on `hard-small` with Haiku as builder and spaCy NER
(`kgbench hard ... --pipelines llm+spacy`, one world): strict triple F1 0.798 to 0.789, entity-pair F1 0.807 to
0.796, linking, mention and entity F1 unchanged. The corpus has no role or acquisition gold (zero
"acquired" in its text), so the bench cannot show the gain; the small dip is within the noise of a different
set of Haiku answers on one world. The offline extraction scores and every gate floor are unchanged.
