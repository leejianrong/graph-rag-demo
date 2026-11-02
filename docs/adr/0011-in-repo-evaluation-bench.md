# ADR-0011: An in-repo, independent knowledge graph evaluation bench

- Status: Accepted (2026-11-02)
- Supersedes the external eval-bench integration (never adopted as a dependency; its export stage was removed in #32).
- Related: ADR-0009 (benchmarking strategy), ADR-0010 (port boundary)

## Context

We are turning the demo into a product (a memory and knowledge store for agents and developers). Product claims need
numbers we can defend. The external eval-bench could not give them:

1. **Circular gold.** Its multi-hop chains were mined from the graph under test.
2. **Wrong thing measured.** It scored an agent plus its tools, not our retrieval.
3. **Not reproducible.** Private package index, placeholder image registry, undistributed corpus.

The V8 harness (`graph_rag/benchmark/`) is deterministic and non-LLM but only covers 2Wiki-shaped answer scoring.

## Decision

Build our own bench in a new top-level `bench/` package (own `pyproject.toml`, CLI `kgbench`, tests and data),
depending on `graph_rag` only through its ports and public entry points.

1. **Ground truth is a programmatically generated world model, with complexity under our control.** A declarative,
   seeded `WorldSpec` (YAML) drives the generator, so the same spec and seed always give the same graph. Documents are
   rendered from it and questions plus gold paths are derived from it, so gold never depends on our extraction.
   The spec exposes knobs in three groups:
   - **Graph shape:** entity count, entity-type mix, predicate mix (from the closed set), degree distribution
     (uniform to power-law), community count and mixing, cycle rate, maximum path length.
   - **Surface form and ambiguity:** aliases per entity, homonym rate (distinct entities sharing a name), distractor
     documents, relation noise, and temporal churn (facts that change or are retracted).
   - **Question mix:** hop-count distribution and answer-type mix.

   Named presets (`tiny`, `small`, `medium`, `large`, `stress`) fix the knobs. Every generated world emits a **stats
   report** (realized degree distribution, homonym count, path-length histogram, and so on) so a knob is verified, not
   assumed. Complexity **sweeps** vary one knob at a time to show where each system degrades.
2. **Two renderers.** A template renderer (deterministic, `$0`, runs in the fast gate) and a cached-LLM renderer
   (realistic text). Rendered fixtures are committed so a run needs no provider.
3. **Public sets sit alongside** for outside comparability: 2WikiMultihopQA and LongMemEval.
4. **Layered metrics**, each against the world model:
   extraction (mention, entity, triple P/R/F1), entity linking (B-cubed), retrieval (supporting-fact recall@k, path
   recovery), answers (EM, token-F1), cost (dollars, latency, LLM calls), and later memory behaviour (incremental equals
   batch, supersession, forget, order-insensitivity, tenant isolation).
5. **Baselines on the same questions:** BM25 and dense-passage retrieval, then graph-rag. A lift claim needs them.
6. **Judging is non-LLM by default.** An optional LLM judge is allowed for open-ended answers, only with its error rate
   measured against labelled cases and reported with the results.
7. **Two tiers.** Offline deterministic tier gates PRs; a real-stack tier runs nightly and its traces are published.
8. **First milestone: retrieval lift** over the baselines on multi-hop questions.

## Consequences

- No dependency on eval-bench code, data or infrastructure.
- Synthetic text may be easier or stranger than real text, so public sets stay in the report and conclusions name that gap.
- The V8 metrics, dataset loader and harness move into `bench/`, and `graph_rag/benchmark/` is then removed.
- The world-model generator doubles as a test fixture source for the incremental, temporal and tenancy epics.

## Rejected

- Reusing eval-bench: circular and not reproducible (above).
- Public datasets only: no gold for updates, deletes, ordering or tenancy.
- LLM judge as primary: cost, variance and bias that need calibration first.
