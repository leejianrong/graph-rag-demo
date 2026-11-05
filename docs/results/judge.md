# LLM judge error rates (GRA-89)

Can a small model stand in for the offline matcher when scoring open-ended answers? This run
measures it on 69 labelled cases built from the `small` world's open-ended questions
(`kgbench judge --build small`). Labels are true by construction, not from any model.

The judge is Claude Haiku, answering through semula (a recording OpenAI-compatible server): two
Haiku subagents answered the 69 parked prompts, with no API spend. Their own tallies (45 and 12) add up to
57, not 69, so I cannot say which of the two answered each prompt; every answer is a Haiku verdict. The answers are committed in
`bench/kgbench/data/judge-cache/`, so `kgbench judge` replays them offline and the fast suite checks
that they still give the numbers below.

| scorer | false positive rate | false negative rate |
|---|---|---|
| offline matcher | 53% (24/45) | 50% (12/24) |
| LLM judge | 0% (0/45) | 0% (0/24) |

| case kind | label | matcher right | judge right |
|---|---|---|---|
| stated | covered | 12/12 | 12/12 |
| pronoun | covered | 0/12 | 12/12 |
| negated | not covered | 0/12 | 12/12 |
| co_mention | not covered | 0/12 | 12/12 |
| wrong_object | not covered | 9/9 | 9/9 |
| absent | not covered | 12/12 | 12/12 |
69 cases; the judge gave no usable verdict on 0.

## How to read it

- **The matcher's errors are the point of the cases.** It ties a fact to one sentence containing
  both names, so it reads a denial or a bare co-mention as covered and misses a fact stated about
  "it". The judge got all three kinds right.
- **0 errors is a floor, not proof of perfection.** 45 uncovered and 24 covered cases bound the
  true error rate at about 6% (false positive) and 11% (false negative) at 95% confidence, and these
  cases are plain, templated sentences. Hedged, long or ambiguous prose is untested.
- **One model, one prompt, one run.** Temperature 0, but a subagent is not a deterministic API, so a
  fresh recording could differ slightly; the committed answers are what the numbers describe.
- **Stale claims are not judged.** `forbidden_claims` still go to the matcher; "states X as
  currently true" needs its own prompt and its own labelled cases.
- Off by default: nothing calls a model unless `kgbench judge --base-url` is given or a caller
  passes `llm_judge(...)` to `score_open_ended`.

Reproduce: `kgbench judge` (replay) or `kgbench judge --base-url http://127.0.0.1:8080/v1` to record.
