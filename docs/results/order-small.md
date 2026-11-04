# Ingest order sensitivity: small world, ambiguous ambiguity

Entity linking can depend on what the pipeline has already seen, and incremental ingest does not control arrival order. The same corpus was ingested 7 times, once as generated and 6 times in shuffled orders, each into a fresh pipeline, and scored against the generator's gold.

## Setup

- Preset `small` with the `ambiguous` ambiguity setting, world seed 0, 145 documents.
- Orders: as generated, then shuffle seeds [1, 2, 3, 4, 5, 6].
- Reproduce: `uv run kgbench order --preset small --difficulty ambiguous --orders 6 --out docs/results/order-small.md --json docs/results/order-small.json`

## Result

Every metric came out the same in every order.

| metric | mean | std | min | max |
|---|---|---|---|---|
| canonical entities | 195.000 | 0.000 | 195.000 | 195.000 |
| linking precision | 0.933 | 0.000 | 0.933 | 0.933 |
| linking recall | 0.769 | 0.000 | 0.769 | 0.769 |
| linking F1 | 0.843 | 0.000 | 0.843 | 0.843 |
| linking F1 (homonym) | 0.758 | 0.000 | 0.758 | 0.758 |
| linking F1 (alias) | 0.689 | 0.000 | 0.689 | 0.689 |
| linking F1 (plain) | 0.920 | 0.000 | 0.920 | 0.920 |
| entities F1 | 0.740 | 0.000 | 0.740 | 0.740 |
| triples F1 | 0.000 | 0.000 | 0.000 | 0.000 |
| edges F1 | 0.681 | 0.000 | 0.681 | 0.681 |

## What this does and does not show

- This is the offline pipeline: heuristic entity detection and linking that merges by name. Merging by name does not depend on what came first, so no spread is expected here, and none is the finding. It shows the harness reports no order effect when there is none (the tests show it reports one when there is).
- The real stack (spaCy, an LLM-built graph, linking against a persistent entity store) was not run. Its linking may well depend on order; run `kgbench order --real` with the services up to measure that.
- Scores depend only on which mentions are grouped, not on canonical ids. Entity alignment breaks vote ties by entity id, so the scorer adds no order effect of its own (it used to: ties went to whichever mention arrived first).
