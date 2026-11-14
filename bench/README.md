# bench (kgbench)

The evaluation bench: it scores graph-rag, BM25 and a dense retriever against gold that comes from the generator
(`datagen/`), never from graph-rag's own extraction. Design: [ADR-0011](../docs/adr/0011-in-repo-evaluation-bench.md).
Method, every command and every result: [docs/BENCH.md](../docs/BENCH.md). Output of past runs is in `docs/results/`.

```bash
uv sync --frozen --extra dev          # installs kgbench as a workspace member
uv run kgbench --help                 # every subcommand
make benchmark                        # answer EM / F1 over the bundled 2Wiki mini fixture, offline
make gate                             # the CI quality floors (bench/kgbench/gate.json), about 15 seconds
make compare                          # BM25, dense and graph-rag per hop count (needs the embed extra and `make embed-model`)
```

`kgbench/data/` holds committed fixtures and LLM response caches (Claude Haiku answers recorded through semula, so replays need no
endpoint). `data/` is in `.gitignore`, so a new file there needs `git add -f`. How each cache was made and how to
re-record it: [docs/DATA-PROVENANCE.md](../docs/DATA-PROVENANCE.md). The real 2WikiMultihopQA slice is not committed;
the same file explains how to fetch it. Tests (`bench/tests`) run in the fast suite, including the import-boundary test:
the bench may import `graph_rag` only through its ports and public entry points.
