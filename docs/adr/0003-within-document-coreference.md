# ADR-0003 — Within-document coreference, cluster-map output

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

Coreference groups mentions that refer to the same thing (including pronouns:
"she", "they", "it"). It can be scoped within a single document or across the
whole corpus. Cross-document coreference is substantially harder and overlaps
conceptually with entity linking (deciding that "John Smith" in doc A and doc B
are the same person).

## Decision

Resolve coreference **within a single document only**, LLM-backed. Cross-document
identity is **not** coref's job — it is handled entirely at the entity-linking
stage against the corpus-local entity store (see
[ADR-0004](./0004-corpus-local-entity-linking.md)).

Output is a **cluster map** — original text is preserved and each cluster maps
its mentions to a chosen canonical in-document mention — rather than an inline
pronoun rewrite. The cluster map is more useful downstream (feeds EL and
provenance) and is non-destructive.

## Consequences

- Clean separation of concerns: coref = intra-document; EL = cross-document.
- Each document's coref clusters become the **doc-level entities** handed to EL.
- Preserving original text keeps character spans valid for provenance.
- LLM is used here; cost is bounded by the per-document token budget (see
  sizing decisions in `CONTEXT.md`).
