# ADR-0005 — Elasticsearch split: Documents index + Entities index

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

Two distinct kinds of state need storing: (1) per-document records — the
original text plus that document's processing results — and (2) the deduplicated
corpus-wide canonical entities used for entity resolution. REQS raised whether
these should be two separate Elasticsearch *instances* or two *indices*, and
where the per-document EL result should live.

## Decision

Use **two indices in a single Elasticsearch cluster**: `ES-Documents` and
`ES-Entities`. Separate instances are unnecessary weight for a local demo;
two indices give clean separation with one container to run.

- **`ES-Documents`** — one record per document: original text, NER mentions
  (with spans), coreference cluster map, and the **per-document EL result**
  (which doc-level entity resolved to which canonical entity ID).
- **`ES-Entities`** — the **corpus-local canonical entity store**: one record
  per deduplicated entity, keyed by the canonical entity ID; the source of truth
  that entity linking blocks/searches against.

The knowledge-graph builder reads a document's text + its linked-entity clusters
from `ES-Documents` (grounding triples in canonical entity IDs) and may read
`ES-Entities` for entity metadata. *(Exact read pattern finalized in section F.)*

## Consequences

- Single ES container to operate locally; lighter than dual instances.
- The two indices have very different shapes and write patterns
  (document-append vs. canonical-upsert), so separating them keeps mappings
  clean and queries simple.
- `ES-Entities` doubles as the substrate for the EL embedding search
  (vector field on the canonical entity record).
