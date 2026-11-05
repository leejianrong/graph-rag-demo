# A harder `tiny` dataset, written by Claude Haiku

The `tiny` world with the same extra ambiguity as `../hard-small` (alias rate 0.7, two aliases per entity,
homonym rate 0.15, distractor rate 0.2), rendered by Haiku in the `rich` style: 26 documents of about 36
words, 60 questions (15 per hop count). It is small on purpose: it is cheap enough to run the real LLM
pipeline stages over, which `hard-small` is not (see below).

Same layout as `hard-small`: `world.json`, `questions.json`, `documents.json`, `world.stats.json`, the three
specs, and `cache/` (the 26 recorded writer answers). Replay the text offline:

    kggen docs generate --spec datagen/examples/hard-tiny/spec.yaml --renderer llm --style rich \
      --model claude-haiku --cache-dir datagen/examples/hard-tiny/cache

The recorded answers of the pipeline's own LLM stages (coref and graph build, Haiku through semula) are in
`bench/kgbench/data/llm-cache-hard-tiny/`; `kgbench hard --dir datagen/examples/hard-tiny --llm-cache
bench/kgbench/data/llm-cache-hard-tiny` replays them with no endpoint. Results: `docs/results/hard-tiny.md`.
The writers were told to add nothing beyond the facts, but the articles were not audited for it.
