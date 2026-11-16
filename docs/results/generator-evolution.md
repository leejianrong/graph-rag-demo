# Worlds that change over time (GRA-190, generator half)

Done on 2026-11-16. A world could already replace or retract a fact (`churn_rate`, `retraction_rate`) and the generator asked `as_of`, `latest` and `retracted` questions, but the entities themselves never changed, the documents were undated (one article per subject stating every version of a fact at once), and nothing asked when, how long or what came before. This adds the rest. Everything is opt-in: the default output of every preset and spec is unchanged (the committed fixtures regenerate byte for byte; only their README text, which describes the files, changed).

## What a world can now do

`--evolution light|heavy` (or the knobs `birth_rate`, `closure_rate`, `rename_rate` plus churn and retraction):

| change | what it does | what keeps it consistent |
|---|---|---|
| birth | an entity first appears in a later year; its facts begin then | no fact starts before an endpoint exists |
| closure | an entity closes; every fact about it ends then | no fact outlives an endpoint |
| rename | an organisation or product takes a new ending ("Gaze Industries" to "Gaze Partners") | the first word stays, so the name bank's alias rules still hold; the old name is kept as an alias and as `former_name` |
| churn, retraction | as before | |

`World.entities_as_of(year)`, `Entity.name_in(year)` and `Entity.exists_in(year)` give the ground truth at any year (`edges_as_of` existed). `light` is churn 0.15, retraction 0.05, birth 0.10, closure 0.05, rename 0.10; `heavy` is 0.35, 0.10, 0.25, 0.12, 0.25. On `small` with `hard` difficulty and `heavy`, 180 entities get 33 births, 16 closures and 13 renames.

## Dated documents

`docs generate --timeline` writes one short document per (year, subject), in date order, each with a `date` (`YYYY-07-01`). The start year states what is true then. After that a fact is stated in the year it begins (present tense, "since 2011") and again in the year it ends (past tense, "from 2011 until 2019"), and an entity's appearance, closure and renaming are sentences with no fact behind them ("X was established in 2008", "X ceased operations in 2019", "Gaze Industries was renamed Gaze Partners in 2001"). A mention uses the name the entity had that year. This is the shape a memory store receives: a stream of dated documents, each superseding some of the last, which is what `document_date` and supersession (GRA-42, GRA-43) are for.

## New question types

| type | asks | answer |
|---|---|---|
| `when` | the year a fact began, or the last year it was true | a number |
| `before` | what the same subject and relation held immediately before another fact | an entity |
| `duration` | how many years a fact held | a number |

Only a fact whose dates the documents state is asked about: `when` and `duration` need a start after the world's first year, and a test checks that the year is readable in the timeline documents. A world with no change yields none of them (the existing fallback to plain entity questions applies). With `heavy` on `small` all three come back.

## Not done

- The bench cannot score any of this yet (GRA-190's second half): no temporal row, no recorded text, no gate floor.
- `before` and `when` use one wording; the wording bank and the question-style work (GRA-186, GRA-187) do not cover them.
- Renames cover organisations and products only (not people or places), and a rename keeps the first word, so it does not test linking two wholly different names.
- Relative dates ("last year") and implicit changes ("after the merger") are not written; every date is an explicit year. GRA-188 needs those.
- Facts that began before the first year carry no date in the text, so `when` never asks about them.
