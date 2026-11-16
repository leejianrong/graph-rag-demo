# What the Studio needs from indah

Written 2026-11-16 (GRA-193), from building the Studio on indah 0.14.0, the version installed here.
The indah repo is at 0.15.0; I did not check whether any of this already landed there. Each item says what the Studio
does today to work around the gap, so it can be judged by what it would remove. The API shapes are
proposals for indah's maintainers to change. Cards are on the IND board.

| # | Request | Studio cost today | Size |
|---|---|---|---|
| 1 | Reactive `Slider` range | a fixed 0 to 50 track, with dead length past the last year | small |
| 2 | Timeline control | slider, four buttons and a text readout built by hand | medium |
| 3 | Graph element styles and stable layout | colour only; the year view can't draw "ended" as a dashed edge, and node positions depend on `group` | medium |
| 4 | Choice cards | each card is a `Card` + `Text` + `Button` with a "Chosen" line | small |
| 5 | Client-side graph frames | one server round trip per step; play speed is capped by the 5 events/s bucket | large |
| 6 | Display text size | the year can't be the biggest type on the page | small |

## 1. Reactive `Slider` range

`Slider(min=, max=)` take numbers, read once. The year slider's range depends on the world, which
does not exist when the page is built, so it is index 0..50 with the year printed beside it.
**Proposal:** `min`, `max` and `step` accept a `Signal`, a `Computed` or a callable, like `value`
already does; a value outside a new range is clamped and reported once. **Done when:** a test
changes `max` from Python and the slider's range updates without a rebuild.

## 2. Timeline control

Stepping a changing thing through time is a pattern (this graph, a log, a run's history).
**Proposal:** `Timeline(value, points, *, labels=None, density=None, unit=None, playing=None)`:
- `points`: an ordered list of the moments that exist (any number, any spacing); `value` is one of
  them. Stepping goes to the next point, so a world that changes on irregular dates steps through
  exactly its dates.
- `density`: an optional number per point, drawn as a small histogram on the track (where the busy
  moments are).
- Buttons for previous and next point, previous and next *event* (a point with density above 0),
  play and pause with a speed; arrow keys step, space plays, Home and End jump.
- `labels`: how a point prints (`2003`, `March 2003`, `5 March 2003`); `unit` is for the screen
  reader ("year 2003, step 4 of 11").
**Done when:** the Studio's year view uses it with no extra buttons, and a keyboard-only user can
play, pause and step.

## 3. Graph element styles and stable layout

The year view needs to say "this edge ended", "this node was renamed" and "this one is not there
yet" without changing where anything sits. Today `Graph` takes a `color` per node and edge, plus
`group`, and the starting layout is clustered by group, so a group that changes with the year moves
the nodes; the Studio keeps the type as the group and paints state as a colour, which a node outside
the focus window loses (it is a dot in its group's colour).
**Proposal:**
- node: `ring` (a colour or `True`), `opacity` (0 to 1), `label_style` (`"strike"`, `"bold"`);
- edge: `dash` (`True` or a pattern), `opacity`, `width`;
- both drawn the same inside and outside the focus window;
- layout seeded from node ids and edges only (never `group` or style), documented as stable, and
  `Graph(update="keep")` so a replaced graph keeps the position of every id it already had and
  places only new ids;
- `transition_ms` to ease colour, opacity and position changes between updates (off under reduced
  motion).
**Done when:** stepping a year changes only styles in a test that reads node positions before and
after, and a dashed edge is visible outside the focus window.

## 4. Choice cards

A choice with a sentence of explanation each (a preset, a difficulty) is a radio with rich options.
**Proposal:** `ChoiceCards(value, options=[{"value", "title", "body", "badge"?}], columns=None)`,
a radiogroup with arrow-key movement, the chosen card marked by more than colour, and each card one
click target. **Done when:** the wizard's cards are one component per question, and a screen reader
hears the group's label, each title, and which is chosen.

## 5. Client-side graph frames

Each year is a different graph; sending the whole graph per step is wasteful and capped by the
event bucket. **Proposal:** `Graph(frames=..., frame=Signal[int])`: a base graph plus, per frame, a
list of style and visibility patches (ids and the style to apply); the browser applies the frame
locally and only reports the frame number. Play then needs no round trips. **Done when:** playing
twenty frames at 4 per second sends no events.

## 6. Display text size

`Text` has one size. **Proposal:** `Text(..., size="display" | "title" | "label")` mapped to the
theme's `font-display`, so a readout (the year) can be the largest type on the screen.

## Not asked for

Persistence (workspaces) stays in the app: it is a few JSON files and indah has no business owning a
store. Authentication is untouched (the Studio is local).
