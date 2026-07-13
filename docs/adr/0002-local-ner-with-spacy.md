# ADR-0002 — Local NER with spaCy

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

NER is the first extraction stage. It can run via an external LLM or a local
model. A stated project goal is to avoid burning LLM API credits, and NER is a
high-volume, per-token operation applied to every document. Standard entity
types (people, organizations, locations, dates) suffice, and English is the only
target language.

## Decision

Perform NER **locally with spaCy**, using the transformer pipeline
**`en_core_web_trf`** for accuracy (fall back to `en_core_web_lg` where CPU
speed matters). No LLM, no API cost.

- **Entity types:** curate spaCy's OntoNotes set down to the types this use case
  needs — **PERSON, ORG, LOCATION** (merging `GPE` + `LOC`), **DATE, EVENT,
  NORP**, and optionally **PRODUCT**. Fewer types → cleaner graph, less noise,
  more readable multi-hop paths. Widen later if needed.
- **Spans:** retain **character offsets** for every mention. They are required
  to align coreference mentions, to attach provenance (sentence/char range) to
  triples, and to support dedup and UI highlighting. spaCy provides them at no
  extra cost.

Alternative noted for the future: **GLiNER** (zero-shot NER by type name) if
custom entity types are wanted without retraining.

## Consequences

- NER contributes **$0** in API tokens.
- spaCy also provides sentence segmentation in the same pass, feeding the
  provenance model.
- The curated type set is a deliberate narrowing; broadening it later is cheap
  but re-processing the corpus would be needed to backfill new types.
- Not every extracted type becomes a graph node: `PERSON, ORG, LOCATION, EVENT,
  PRODUCT, NORP` are first-class nodes while `DATE` is modeled as an edge
  attribute — see [ADR-0006](./0006-knowledge-graph-builder-and-model.md).
