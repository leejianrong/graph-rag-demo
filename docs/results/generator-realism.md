# Generator realism: what changed, and what it did and did not fix (GRA-183)

Done on 2026-11-16. The generator's worlds were easy to read in four ways that showed up in this project's own results: every preset was clean, names were all made-up capitalised words, questions and sentences had almost one wording, and `shared_name` over-warned. Each is now an opt-in option. **The default output of every preset and spec is byte-identical to before** (the committed fixtures regenerate unchanged, and a test fails on drift); an option changes a world only when asked for. The generator UI starts with difficulty `hard`, names `bank` and wording `varied`.

| option | CLI | spec field |
|---|---|---|
| difficulty `clean`, `aliases`, `hard` | `--difficulty` | the knobs `alias_rate`, `aliases_per_entity`, `homonym_rate`, `distractor_rate` |
| names `invented`, `bank` | `--names` | `WorldSpec.name_style` |
| phrasing `plain`, `varied` | `--phrasing` | `QuestionSpec.phrasing`, `RenderSpec.phrasing` |
| ambiguity `ignore`, `mark`, `drop` | `--ambiguity` | `QuestionSpec.ambiguity`, `Question.ambiguous_under_names` |

A plain, invented, clean, ignoring spec serialises exactly as before: the new fields are left out of a spec file while they are at their defaults, and `ambiguous_under_names` is left out of a question until it was assessed.

## 1. Difficulty

A preset sets only the graph's size and density. None sets an alias, shared-name, distractor, noise or churn rate, so every preset world was clean, and only the hand-made `hard-small` spec was not. `hard` layers alias rate 0.7 (2 per entity), homonym rate 0.15 and distractor rate 0.2 over any preset or spec; it is exactly what `hard-small` used (a test compares them). `aliases` is the middle step.

## 2. Phrasing

`varied` draws wording from a bank that is written down in the code, not invented at generation time, on a random stream of its own. So a plain and a varied run ask the **same questions with the same ids, answers and gold paths** and lay out the **same facts in the same documents**; only the words differ (tests assert both). Sentences gain synonyms and, for most relations, a passive form that names the object first ("`{O}` was founded by `{S}`"), which is how a reader tells a relation from its reverse; the faithfulness check found a judge misses reversed facts. Questions gain stems for entity, set, count, compare, unanswerable and false-premise questions ("Between A and B, which has more…"). Questions of the other types (path, open-ended, as-of and the like) keep one wording. The nested "the location that the location that…" shape is unchanged.

## 3. Names

`bank` draws realistic names without replacement (people as a first name and a surname, organisations as a word plus the usual ending, places, events, products, demonym groups). The bank is checked by `check_bank` before any generation:

- every word is unique across all pools, so no surname, head word or place can double as another entity's alias (a person's aliases are "F. Last" and "Last", an organisation's its first word);
- first names are disjoint from every other pool, and no word is a word the generator adds ("Holdings", "Summit", "Greater");
- no word is a prefix of another in its pool, which would make one a short form of the other.

Curated names are drawn first, then built ones (a stem and an ending: `ash` + `worth`, `Blue` + `bridge`), each family with endings of its own. The validator caught real conflicts while it was built (a surname that was also a place, a word the generator already uses, and an earlier bug of mine where only the first of forty "Granite…" names was pruned; the validator's own prefix test had the same flaw and under-reported).

| pool | curated | total |
|---|---|---|
| surnames (people) | 150 | 3,808 |
| organisation heads | 95 | 4,645 |
| places | 65 | 2,680 |
| event heads | 50 | 1,149 |
| product heads | 50 | 1,274 |
| demonym groups | 25 | 2,702 |
| first names | 80 | 190 |

That covers every preset up to `large` (10,000 entities; a test checks each preset's need against the capacity, and a real `medium` world is generated from it). `stress` (50,000 entities) asks for more than the bank holds and fails with a message that says to use `--names invented`. A small world draws curated names, apart from the shared names (homonyms) that the `hard` difficulty adds, which can belong to another type.

**What the bank did not do.** It was meant to stop NER being an oracle. On the same small hard world (656 and 658 gold mentions), the detectors find the same share of mentions whether names are invented or from the bank:

| NER | invented, overlap / exact | bank, overlap / exact |
|---|---|---|
| rule-based | 100% / 94% | 100% / 94% |
| spaCy `sm` | 85% / 83% | 84% / 82% |
| spaCy `trf` | 100% / 96% | 100% / 99% |

Every bank name is still one capitalised word, so a capitalised-span detector finds all of it. The bank buys realism of reading, uniqueness and capacity, not a harder extraction test. What would is names with connectors, particles, bare common words and abbreviations: GRA-184.

Also not guaranteed: a bank world and an invented world from one seed are **not the same graph**. The bank draws names from the world's random stream and an invented name draws a different amount from it, so everything after shifts. Making them identical would change every committed fixture.

## 4. Ambiguity

`shared_name` marks a question whose path touches an entity whose name another entity also carries. It is a warning, and it over-warns. `kggen.ambiguity` ports the bench's check: treat entities that share a name or alias as one, re-walk the chain keeping each step's type, and see whether an answer appears beyond the gold. `mark` records the result as `ambiguous_under_names` on chain questions (entity, set, count; the others cannot be re-walked and stay unassessed); `drop` never emits an ambiguous chain question and still delivers the requested count.

On the `hard-small` world, 300 questions:

| | chain questions | `shared_name` warns | truly ambiguous | of the warned, ambiguous |
|---|---|---|---|---|
| entity only (as committed) | 300 | 177 (59%) | 22 (7%) | 22 (12%) |
| mix of entity, set and count | 300 | 142 (47%) | 37 (12%) | 37 (26%) |

So 88% of the entity questions the warning flags have one answer even to a reader who sees only names. The true flag is always a subset of the warning (a test checks it). With `drop`, 300 questions are delivered with a balanced hop mix (75 per hop, or 76, 76, 75, 73 with sets and counts in the mix).

A hand-built four-entity world pins the algorithm (Ann works for Acme, which makes Widget; a second "Acme" makes Gadget): names alone reach both products, and renaming the second company removes the flag.

## 5. Found on the way

`datagen/tests/test_ui_app.py` had seven tests that no longer matched the UI (the question note now lists the mix by type, documents open with a filler sentence that states no fact, the hop weights steer only the chain questions, a sampled graph highlights what is drawn). CI never ran them because `indah` is not installed there, so they skipped and rotted. They are fixed here, and the UI now has tests for the new defaults. Nothing in CI would catch the next drift: the UI tests still skip in CI.

## Not done

- Names that are hard for NER (GRA-184); pronouns and coref gold (GRA-180); more than one LLM-written world (GRA-181).
- The UI has no control for `ambiguity`; it is a CLI and spec option.
- None of this changes the recorded `hard-small` caches or any published number: those datasets were made by the old generator and are unchanged. A bench result on a new world needs new LLM-written text and recordings (GRA-181).
