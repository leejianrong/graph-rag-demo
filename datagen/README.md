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
| time unit | `--time-unit month` or `day` lets facts, births, closures and renames fall on a month or a day instead of a year. Documents and questions write the date to that precision ("since March 2003", "In what month did ...?"); a `when` answer is a date accepted as written or ISO, a `duration` counts the unit. Year is the default and nothing about a year world changes. Not in the UI yet: its year slider needs GRA-193 | `--time-unit month` |
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

**Workspaces (GRA-194).** The UI is a small Studio: a rail of saved workspaces beside tabs (Setup, World, Questions, Documents, Export, History). Name a workspace in the rail to keep the form's settings; every Generate that changed them adds a revision (up to 50). Only the settings are stored, never the data, because generation is deterministic: opening a workspace puts its latest settings back and makes the same data again (not for `large` or `stress`, which wait for Generate). History lists the revisions and can restore one, rename, duplicate (latest settings only) or delete the workspace. Without a workspace the UI is a scratch session that saves nothing. Files are `<root>/<slug>/workspace.json`, plain JSON; the root is `$KGGEN_STUDIO_DIR` or `~/.kggen/studio`. Not yet: a URL per workspace, and users (workspaces are shared by everyone who can reach the server, so run it locally).

**The wizard (GRA-195).** The Setup tab is five steps instead of one long form: Start (a size card), World (difficulty and names, with the numbers behind *Fine-tune*), Time (change over time and dated documents), Questions (who asks, and wording), Review. Each choice is a card that says what it produces; the numbers in the cards come from the generator's own tables. A line under the steps estimates the entities and facts from the form's numbers without generating. Review lists the choices and every number that differs from what the preset would set; *Generate world* makes the data and opens the World tab. Every step edits the same form values as before, so workspaces save and restore them unchanged. The cards are buttons for now: indah has no radio with rich options (GRA-193).

**The look (GRA-196).** `kggen_ui/theme.py` is one indah `Theme`: cool paper and ink, one blue accent for "selected", and a fixed code for time (green born, rust ended, purple renamed, amber replaced) that is the same colours the year view paints on the graph. Display type is Bricolage Grotesque, text IBM Plex Sans, ids and numbers IBM Plex Mono, self-hosted from `kggen_ui/fonts/` (SIL OFL; latin subset), so the page asks no other host for anything. Light and dark are both defined; a test checks WCAG AA contrast for every text, control and accent pair.
