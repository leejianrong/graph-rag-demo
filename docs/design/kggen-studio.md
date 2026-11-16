# kggen Studio: a proposed redesign of the generator UI

Proposal, 2026-11-16. Nothing here is built. It answers four asks: a year slider that shows a world evolving, workspaces that persist, a click-through wizard instead of one long scroll, and a design that is not limited to what indah does today. The last section says what indah already has and what it would need.

## The job

**Subject.** `kggen` makes a synthetic world (entities, relationships, facts that start and end), writes documents and questions about it, and exports the lot. **Audience.** Engineers and ML engineers who need a dataset they trust, and who want to look at it before they measure anything against it. **The one job of the UI.** Make a world with a few decisions, then see what you got, including how it changes over time.

Today's UI is one page: a card of ~30 knobs, an "asked vs. realized" card, then the world, questions, documents and export, stacked. It works for someone who already knows the knobs. It does not help a newcomer decide, it loses everything on reload, and it has no way to see time.

## Shape of the app

```
┌────────────┬───────────────────────────────────────────────────────────┐
│ Studio     │  Harbour towns, heavy churn            saved 2 min ago    │
│            │  Overview  World  Documents  Questions  Export  Settings  │
│ Workspaces ├───────────────────────────────────────────────────────────┤
│ ● Harbour  │                                                           │
│   towns    │                      (the tab's content)                  │
│ ○ Persona  │                                                           │
│   bench    │                                                           │
│ ○ Tiny     │                                                           │
│   smoke    │                                                           │
│            │                                                           │
│ + New      │                                                           │
└────────────┴───────────────────────────────────────────────────────────┘
```

- **Workspace** is the unit. It holds a name, the three spec objects that already exist (`WorldSpec`, `RenderSpec`, `QuestionSpec`), and the generated data. Each has its own URL (`/w/harbour-towns/world`), so a view can be linked.
- **Persistence is cheap here because generation is deterministic.** A workspace stores only its specs (small JSON, one file per revision under `~/.kggen/studio/<slug>/`) and a cache of what they generate. Delete the cache and the same data comes back from the seed. Changing a setting saves a new revision; the Settings tab lists revisions and can restore one or compare two.
- **Switching is a rail click.** New workspace opens the wizard. Duplicate copies the latest specs, which is how you try "the same world, but heavy churn."
- **Tabs replace the long scroll.** Each tab is one screen. Nothing needs a page-long scroll except the document and question lists, which are tables with their own scroll.

## The wizard (new workspace and "edit setup")

Five steps, shown as a horizontal progress line that is clickable once a step is done. Each step has one decision, large choice cards, and a live summary on the right ("about 480 entities, 1,900 facts, 140 documents"). The summary comes from the spec's own arithmetic, not a generation, so it updates as you click. **Back** keeps everything. **Skip to generate** is available from step 1, since a preset alone is a complete answer.

```
 Start ─── World ─── Time ─── Questions ─── Review
   ●─────────○────────○─────────○───────────○

 How big a world?                                   You'll get
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  ~480 entities
 │ tiny     │ │ small    │ │ medium   │ │ large  │  ~1,900 facts
 │ 50       │ │ 500      │ │ 2,000    │ │ 10,000 │  ~140 documents
 │ ▪ ▪ ▪    │ │ ▪▪▪ ▪▪▪  │ │ ▪▪▪▪▪▪   │ │ ▪▪▪▪▪▪ │
 └──────────┘ └──────────┘ └──────────┘ └────────┘  [ Next: World ]
```

1. **Start.** Name, then a preset as a card (tiny, small, medium, large, plus "from a saved spec" and "copy a workspace"). Presets stay; they are the fast path.
2. **World.** Difficulty as three cards (clean, aliases, hard), each showing a sample: the same sentence written once clean and once hard. Names (invented or realistic) as two cards. A "Fine-tune" drawer holds the ~30 knobs for people who want them, grouped as they are now.
3. **Time.** Static, light or heavy as cards, each with a small preview strip of the years (births, closures, renames drawn as ticks), and a switch for dated documents. This is where the new evolution controls get explained instead of being one select among many.
4. **Questions.** Who asks, as cards: anyone, newcomer, learner, expert, mixed. Each shows two sample questions. Below: how many, and phrasing (plain or varied). Question types sit behind "Choose types."
5. **Review.** Every setting that differs from the preset's default, as plain sentences ("heavy churn: a third of facts change"), with an edit link back to its step. **Generate** runs it and lands on the World tab. The existing "asked vs. realized" check becomes a panel on the Overview tab after generation.

## The World tab and the time scrubber

This is the signature of the design. The graph is not a picture of the end state; it is a film you can step through.

```
┌─────────────────────────────────────────────┬────────────────────────┐
│                                             │ 2016  step 4 of 11     │
│        (graph, positions fixed for          │ ─────────────────────  │
│         all years so nodes don't jump)      │ + Quillan Foods born   │
│                                             │ ~ Harlow Ltd → Harlow  │
│    ● Harlow ──── makes ──── ▢ Anvil         │   Group (renamed)      │
│    │                                        │ – Abel R. left Marsh   │
│    ╰── employs ── ○ Abel R.  ┄┄ Marsh ┄┄    │   Co. (ended)          │
│                                             │ + Abel R. joined       │
│                                             │   Quillan Foods        │
├─────────────────────────────────────────────┴────────────────────────┤
│ ◀◀  ◀  ▶  ▶▶    year ▾      ▁▂▁▁▅▃▁▂▁▇▂▁▁▃▁▁▂▁     [as of 2016]       │
│ 2008 ┃━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃ 2031            │
└──────────────────────────────────────────────────────────────────────┘
```

- **What changes on screen.** For the selected step, an entity born this step gets a halo; a closed one fades to an outlined ghost; a renamed one shows its new label with the old one struck through for that step only; a new relationship draws in; an ended relationship turns dashed and fades; a replaced fact (new employer) shows both, old dashed and new solid. Colours are paired with shape and a sign (+ ~ –) so none is colour alone.
- **The track under the slider is an event histogram.** Bar height is how many things changed that year, so you can see where the world is busy and jump to it. Left and right arrows step; space plays; **Next change** jumps to the next year that has any event, whatever the unit.
- **Layout is computed once over everything that ever existed** and held fixed, so a node stays put as its neighbours come and go. Only what is true in the chosen year is drawn in full.
- **The side list is the same step as text**, and every line is clickable to select that node or edge. The Documents tab uses the same control and shows the documents dated up to the chosen year.
- **Granularity.** The control is built on an ordered list of change points, not on integers, so a world with month, week or day dates needs a unit select and nothing else. The generator side needs date-valued `valid_from` and `valid_to` (today they are years); that is its own card.

## Look

The generator is a workbench, so it should feel like an instrument panel for looking at a made thing, not a dashboard of cards. Choices, with the reason for each:

- **Colour.** Cool paper `#EEF1F0`, ink `#14212B`, one accent for "selected" `#2457D6`, and a fixed four-colour code that always means time: born `#1F8A5F`, ended `#B8452A`, renamed `#6A4BB8`, replaced `#B7791F`. Dark mode swaps the paper and keeps the four. The time code is the only saturated colour on the screen, so the eye goes to change.
- **Type.** Display in Bricolage Grotesque (workspace names, the year in the scrubber, big numbers), body in IBM Plex Sans, data and ids in IBM Plex Mono. The year readout is the largest text on the World tab.
- **Not a card grid.** Tabs and a rail give structure; cards appear only as choices in the wizard, where each one is a real alternative.
- **Motion** is for time only: the graph's transitions between steps (about 250 ms, off under reduced motion) and the playhead. Everything else is still.
- **Copy.** Controls say what they do ("Generate world", "Next change"); settings are described in what they produce ("a third of facts change"), not in field names.

Self-critique against the brief: a pale-cool look with one saturated code is a choice made for time encoding, not the cream/serif/terracotta default or the black/acid-green one. The risk I'm taking is the scrubber taking the whole bottom of the screen and the year being the biggest type; if that proves too heavy on the Documents tab it can collapse to a single line there.

## What indah has, and what it would need

Checked in indah 0.14.0 (`components.py`). Already there: `AppShell` (rail plus main, drawer on a narrow screen), `Router` (URL per workspace, needs `spa=True`), `Tabs`, `Steps` (display of step state), `Slider` (marks, `commit="release"`), `Graph` (`layout="given"` with x and y, `highlight`, `selected`, `focus`), `Drawer`, `Modal`, `Command`/`CommandPalette`, `Segmented`, `Radio`. So the shell, routing and a first scrubber can be built on what exists. What it lacks:

| # | Enhancement | Why | Today's workaround |
|---|---|---|---|
| 1 | **Wizard**: steps with per-step validity, back and next, a step in the URL, and the choice to jump back to a finished step | `Steps` only displays; every wizard would re-do this | `Steps` plus `Show` and signals, by hand |
| 2 | **Choice cards**: a radio whose options are rich (title, sample, small graphic) | Presets, difficulty and persona are choices with content | `Card` plus `Button`, one signal per card |
| 3 | **Timeline control**: slider over an ordered list of change points, play/pause, step, next-event, speed, an event-density track, arrow and space keys | `Slider` is numeric and `marks` stops at 12 ticks | `Slider` with `marks`, play driven from Python |
| 4 | **Graph change styles**: per-node and per-edge style (dashed, ghost, halo, strike-through label) and a transition between updates | The whole point of the World tab | Colour and `highlight` only; can't show "ended" vs "never" |
| 5 | **Graph frames**: ship every year's diff once and let the browser step | Each step now round-trips to Python; the `/api/event` bucket is 5 per second, which limits play speed | One replace per step |
| 6 | **Persistence**: a small per-user store (JSON documents with revisions) and an unsaved-changes guard | Workspaces must survive a restart | Plain files in the app |
| 7 | **Theme tokens**: colour, type and radius as settings, with dark mode | The look above | Whatever the shell ships |
| 8 | **Live summary rail**: a side panel bound to a computed value that updates as inputs change | The "you'll get" panel | A `Computed` and a `Card` |

I have not verified whether `Graph` supports a dashed edge or a per-node label style; the docstring mentions only `color`, `label` and `size`. Item 4 may be partly there.

## Build order

1. **Year slider in the current UI** (the world graph and the documents list, using what indah has). It is self-contained, so it moves into the World tab unchanged.
2. **Indah: timeline control, graph change styles and transitions** (items 3 to 5), so the slider becomes the scrubber above.
3. **Workspaces and persistence** (items 6, `AppShell`, `Router`): the rail, the tabs, specs saved as revisions.
4. **Wizard and choice cards** (items 1, 2, 8) over the same specs.
5. **Look** (item 7): tokens, type, dark mode.
6. **Generator**: date-valued facts for month, week and day steps.

Each is a separate PR. Steps 3 and 4 do not depend on 2.
