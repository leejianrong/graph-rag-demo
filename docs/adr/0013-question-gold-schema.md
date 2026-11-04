# ADR-0013: Questions carry a typed gold record, not one answer string

- Status: Accepted (2026-11-04)
- Related: ADR-0011 (bench), ADR-0012 (standalone generator); research in `docs/research/question-types.md`

## Context

`kggen` questions were all "Name the X that ..." with one entity as the answer, kept only when exactly one entity
fits. That is too simple: real evaluations ask for lists, counts and comparisons, ask about things that changed or were
withdrawn, include questions whose premise is false or whose answer is not in the data, and ask open-ended questions.
The unique-answer rule also fails on large skewed worlds (a 50,000-entity stress world yielded 1 of 5 four-hop
questions), because most long paths end in several valid answers.

## Decision

Keep the current fields (`id`, `question`, `hops`, `start`, `answer`, `answer_name`, `accepted_answers`,
`supporting_facts`) so existing consumers work, and add a typed gold record. Every field below has a default, so a
plain entity question is unchanged.

| Field | Meaning |
|---|---|
| `type` | The question family: `entity`, `set`, `count`, `compare`, `superlative`, `unanswerable`, `false_premise`, `as_of`, `latest`, `retracted`, `path`, `open_ended` |
| `answer_kind` | How to grade: `entity`, `set`, `number`, `path`, `abstain`, `false_premise`, `open_ended` |
| `answers` | Entity ids that are correct (all of them, for a set); `number` holds a count |
| `answer_names` | Every accepted surface form of every correct answer |
| `ordered`, `tie_rule` | Whether order matters; how ties are handled for a superlative |
| `gold_paths` | One or more lists of fact ids; `supporting_facts` stays the first |
| `abstain` | `{expected, reason}`; reason is `no_evidence`, `false_premise`, `retracted` or `ambiguous` |
| `premise` | For a false premise: `{asserted_fact, contradicting_fact_ids, correction}` |
| `as_of` | The date a temporal question asks about |
| `stale_values` | Names that were true earlier and must not be given as current |
| `required_facts`, `forbidden_claims` | For open-ended questions: fact ids an answer must cover (weighted), and claims it must not make |
| `grading` | `{primary: exact|set_f1|numeric|path|abstain|nugget, judge: null|llm}` |

Rules that keep gold honest:

- Gold comes from the world's facts, never from a pipeline's output. No question's gold depends on an LLM.
- A question the data cannot answer is `answer_kind: abstain`. Its correct response is a refusal, never a guess.
- A false premise is built by changing one true fact, so its correction is known exactly.
- Open-ended gold is a set of required facts plus forbidden claims. The offline matcher checks sentences against
  fact surface forms; an LLM judge is opt-in and cached, and is never needed to produce gold.
- Reports show answerable and unanswerable splits separately, with correct-refusal and false-refusal rates side by side.

## Consequences

- Set-valued answers let large skewed worlds supply questions, because multi-answer paths stop being discarded.
- A `path` question ("How is A connected to B?") has every shortest chain of facts as `gold_paths`, in order; a
  scorer credits a chain that matches one of them.
- Open-ended answers are scored by `kgbench.openended`: a required fact is covered when one sentence of the answer
  names its subject and object, and a forbidden claim is made when one sentence states a stale pair without marking
  it as past. It is a heuristic. A judge can be passed in to decide instead; none ships, and no provider is called.
  Stale claims between two entities that a current fact also links are not asked about, since names cannot tell
  them apart.
- `questions.json` gains fields; old readers that only use the original fields keep working.
- The bench gains scorers per `answer_kind`; they are added with each question family, not up front.
