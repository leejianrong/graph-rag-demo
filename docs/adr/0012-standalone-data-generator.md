# ADR-0012: The synthetic data generator is a standalone package

- Status: Accepted (2026-11-03)
- Amends: ADR-0011 (the generator no longer lives inside the bench)
- Related: ADR-0010 (port boundary)

## Context

ADR-0011 put the world generator inside `bench/`, which may import `graph_rag` through its ports. That kept gold
independent of graph-rag's *extraction*, but the generator still took its entity types and relations from graph-rag's
own schema and installed with it. The generator is useful on its own (for a UI, for other pipelines, for fixtures), and
it should not need the pipeline that consumes its data.

## Decision

Split it in two, in the same repo, as separate uv workspace members:

- **`datagen/` (package `kggen`, CLI `kggen`)** generates seeded worlds, and later documents and questions with gold
  paths. It owns its vocabulary, depends only on `pydantic` and `pyyaml`, and writes plain files. It never imports
  `graph_rag`, `kgbench` or `litellm`; `datagen/tests/test_isolation.py` enforces that.
- **`bench/` (package `kgbench`)** is the only place that knows both sides: scoring, adapters and baselines. It still
  imports `graph_rag` only through ports and public entry points. It depends on `kggen`, and
  `bench/tests/test_vocabulary_alignment.py` checks that every type and relation kggen emits exists in graph-rag.

The LLM document renderer lives in `kggen` with its own small OpenAI-compatible client and cache, not LiteLLM.

## Consequences

- A change to graph-rag's schema fails the bench's alignment test, not the generator.
- The generator can move to its own repo later without code changes.
- The Dockerfile must copy both `bench/pyproject.toml` and `datagen/pyproject.toml` (uv parses the workspace).
