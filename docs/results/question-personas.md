# Question personas: who is asking? (GRA-187)

Done on 2026-11-16. Questions came in one voice: graph-shaped, standalone, asked by no one in particular. A persona now sets what is asked and how, in `kggen` and in the generator UI ("Who asks"), and `kgbench personas` scores the three over one world. Everything is opt-in: with no persona the output is byte-identical to before (a test and the committed fixtures check it).

## The three askers

| persona | what it asks | types it draws on | shape |
|---|---|---|---|
| `newcomer` | discovery: "I'm new to this. Who is Abel's employer?", "Tell me about Saltmarsh" | entity, open-ended, set, unanswerable | mostly one hop; "who" for a person, "what" otherwise |
| `learner` | a conversation that follows what each answer reveals | `conversation`, entity, set | two to three turns; each turn after the first refers back with a pronoun |
| `expert` | links across entities and what does not hold | `common`, path, compare, superlative, false premise, count | two to four hops |

`--persona newcomer|learner|expert|mixed` (`mixed` is equal shares of the three) takes a persona's mix of types and hops unless `--types` or `--hops` is given. `with_persona` does the same in code. Questions carry `persona`, left out when none was chosen.

Two new question types:

- **`conversation`** (the learner). A chain asked one hop at a time: "Which organization is Mireille's employer?" then "Next, name the organization that owns *it*." `turns` holds each turn's question, answer and gold fact; `question` is the last turn alone, which only makes sense with the earlier ones; `Question.transcript()` writes them all. Every turn has exactly one answer among the facts, and no two turns share an answer. Gold is the whole chain.
- **`common`** (the expert). "What do A and B have in common?": the entities both are directly connected to (one to five), where A and B are not themselves connected. Gold is one two-fact chain per answer; graded as a set.

## What `kgbench personas` found

`small` world, hard difficulty, bank names, bge-small-en-v1.5, offline graph-rag, 90 questions per persona, whole-path@5 with 95% Wilson intervals. A conversation is scored on the one fact that answers its last turn (the earlier turns' evidence was found when they were asked), asked two ways, and once as the whole chain.

| asker | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| newcomer | 90 | 62% (56/90, 52%–72%) | 62% (56/90, 52%–72%) | 59% (53/90, 49%–68%) | 56% (50/90, 45%–65%) |
| expert | 90 | 39% (35/90, 29%–49%) | 34% (31/90, 25%–45%) | 30% (27/90, 22%–40%) | 24% (22/90, 17%–34%) |
| learner: last turn alone, its own fact | 60 | 15% (9/60, 8%–26%) | 5% (3/60, 2%–14%) | 3% (2/60, 1%–11%) | 3% (2/60, 1%–11%) |
| learner: last turn with the earlier turns, its own fact | 60 | 70% (42/60, 57%–80%) | 45% (27/60, 33%–58%) | 42% (25/60, 30%–54%) | 47% (28/60, 35%–59%) |
| learner: whole chain, asked with the earlier turns | 60 | 70% (42/60, 57%–80%) | 45% (27/60, 33%–58%) | 38% (23/60, 27%–51%) | 40% (24/60, 29%–53%) |
| expert: what two entities share | 30 | 47% (14/30, 30%–64%) | 43% (13/30, 27%–61%) | 40% (12/30, 25%–58%) | 27% (8/30, 14%–44%) |


- **A pronoun makes a question unanswerable for search.** A last turn on its own ("Next, name the organization that owns it.") is solved 3-15% of the time; the same turn with the earlier turns written out before it is solved 42-70%. Every system gains 27 to 55 points, so resolving what "it" stands for is worth more than the choice of system. BM25 gains most (15% to 70%) because the earlier answer's name is what the words need.
- **The expert's links are harder than a chain** (24-39% against 56-62% for the newcomer), partly because a link needs two facts from two different places. `common` alone is 27-47%.
- **Dense search does worse than BM25 on the conversation**, 45% against 70%, with the history written out: a long query of earlier turns dilutes the vector, while the exact names it contains help BM25.
- The newcomer rows are not a clean comparison with the others: a mix of types, many of them one hop.

## Caveats

- One world, one seed; 60 conversations, 90 and 30 questions in the other rows: intervals are 10-15 points wide.
- The offline graph-rag pipeline only (a co-occurrence graph, `RELATED_TO` edges). A real-stack run would change the graph rows.
- The transcript is the earlier questions with the gold answers; a real conversation would carry the system's own answers, which can be wrong. The history is therefore a best case.
- Pronouns are "it" and "they" only. There is no "he" or "she" (the generator has no genders), and no coreference gold for documents (GRA-180).
- The learner's first turn is an ordinary question, and each later turn is one hop. A learner who jumps, or a turn that adds a second entity, is not generated.
- The wording is a small seeded bank per persona; GRA-186 (intent-driven templates and paraphrase) would widen it.

## Reproduce

```bash
export LITELLM_MODE=PRODUCTION UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=
uv run --extra embed kgbench personas --embedder bge
uv run kggen questions generate --preset small --difficulty hard --names bank --phrasing varied --persona mixed --n 30
```
