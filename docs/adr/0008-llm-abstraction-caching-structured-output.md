# ADR-0008 — LLM provider abstraction, response caching, structured output

- **Status:** Accepted
- **Date:** 2026-07-14
- **Deciders:** Jian

## Context

The LLM is used by coreference, KG-building, the optional EL tie-breaker, and
the optional answer-synthesis mode. Goals: keep costs low, allow cheaper /
alternative providers (GPT-4o-mini, DeepSeek, etc.), and get reliably parseable
output.

## Decision

**Provider-agnostic LLM client.** Wrap model calls behind a thin interface (e.g.
LiteLLM, which speaks the OpenAI API and also DeepSeek and many others) so the
provider/model is a **config choice per stage**. Defaults: `gpt-4o-mini` for the
high-volume extraction stages (coref, KG-build); the fuller model reserved for
the optional answer-synthesis mode. Any OpenAI-compatible endpoint (incl.
DeepSeek) is swappable via `.env`.

**Response caching (Q28).** A persistent cache keyed by
`sha256(model + prompt + params)`. Repeated pipeline runs and, crucially,
benchmark re-runs hit the cache and cost **nothing**.

**Structured output (Q29).** Request JSON/structured output and validate against
**Pydantic** models (coref clusters, triples, links); retry on parse failure.
This keeps parsing reliable across providers whose JSON-mode support varies.

## Consequences

- Cost is minimized: cheap model for volume, caching removes repeat cost,
  provider swappable to whatever is cheapest.
- Extraction stages are decoupled from any single vendor.
- Validation + retry adds a little complexity but removes a large class of
  brittle-parsing failures.
- Cache invalidation is implicit: changing the prompt or model changes the key,
  so improvements naturally bypass stale entries.
