# datagen (kggen)

A standalone, seeded generator of synthetic knowledge graphs, documents and multi-hop questions with exact gold. It
imports nothing from `graph_rag`, `kgbench` or `litellm` (`datagen/tests/test_isolation.py`). Design:
[ADR-0012](../docs/adr/0012-standalone-data-generator.md); question schema: [ADR-0013](../docs/adr/0013-question-gold-schema.md).

```bash
uv run kggen --help
uv run kggen world generate --preset medium --out world.json
uv run kggen export --spec spec.json --out dataset.zip     # world + questions + documents + specs, reproducible
make ui                                                     # the generator UI (indah), see the root README
```

`fixtures/` holds committed small and medium datasets (regenerate with `make fixtures`; a test fails on drift) and
`examples/` holds LLM-written sets with their response caches (`hard-tiny`, `hard-small`, `llm-cache-tiny`), each with its own
README. The UI is the optional `ui` extra (`kggen[ui]`, indah). The root README has the full tour.
