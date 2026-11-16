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

## Making a world less clean

A preset sets only the graph's size and density, so on its own every preset world is clean. These options layer realism
over any preset or spec. Each is opt-in, and the default output of every preset and spec is byte-identical to before
(a test fails on drift). The generator UI starts with the first three on.

| option | what it does | CLI |
|---|---|---|
| difficulty | `aliases` or `hard` sets alias, shared-name and look-alike rates (`hard` is what `hard-small` used) | `--difficulty hard` |
| names | `bank` draws realistic names from a bank checked so no two entities share an alias word; `invented` is the default | `--names bank` |
| evolution | `light` or `heavy` lets the world change over its years: facts are replaced or retracted, entities appear later or close (and their facts end), organisations and products are renamed. Left at `none` it adds nothing | `--evolution heavy` |
| dated documents | `--timeline` writes short dated documents in date order: the start year, then each change in the year it happens (a fact begins or ends, an entity is established, closes or is renamed), each entity under the name it had that year | `docs generate --timeline` |
| persona | who asks: `newcomer` (discovery), `learner` (a conversation, each turn referring back), `expert` (what two entities share, links, false premises), `mixed`; sets the mix of types and hops | `questions generate --persona learner` |
| phrasing | `varied` picks sentence and question wording from a planned bank, on a stream of its own: same facts, questions and gold, different words; some sentences are passive and name the object first | `--phrasing varied` |
| ambiguity | `mark` records `ambiguous_under_names` on chain questions (a reader who sees only names would find a second answer); `drop` never emits one | `--ambiguity drop` |

```bash
uv run kggen world stats --preset small --difficulty hard --names bank
uv run kggen questions generate --preset small --difficulty hard --names bank --phrasing varied --ambiguity mark --n 20
uv run kggen questions generate --preset small --evolution heavy --types when:1,before:1,duration:1,as_of:1,latest:1 --n 20
uv run kggen docs generate --preset small --evolution heavy --timeline
```

The name bank covers every preset up to `large`; `stress` asks for more than it holds and says so (use `--names invented`).
Bank and invented worlds from one seed are not the same graph, and the bank does not make NER harder (every bank name is
still one capitalised word): see [generator-realism.md](../docs/results/generator-realism.md).

`fixtures/` holds committed small and medium datasets (regenerate with `make fixtures`; a test fails on drift) and
`examples/` holds LLM-written sets with their response caches (`hard-tiny`, `hard-small`, `llm-cache-tiny`), each with its own
README. The UI is the optional `ui` extra (`kggen[ui]`, indah). The root README has the full tour.

In the UI, an evolving world (Change over time `light` or `heavy`) can be stepped through its years: switch on *Step through the years* under the graph, then move the slider or use *Previous/Next year* and *Previous/Next change* (the next year anything changes). The graph keeps one layout for every year and recolours it (green new, purple renamed, rust ending, pale not in that year), rings what the year changed and dims the rest; a table lists the year's changes, and the document list shows only documents dated up to that year. A world that never changes says so. The slider's range is fixed at 50 years after the first (an indah slider cannot change its range).
