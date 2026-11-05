# A harder `small` dataset, written by Claude Haiku

The `small` world with more ambiguity (`spec.yaml`: alias rate 0.7, two aliases per entity, homonym
rate 0.15, distractor rate 0.2), rendered by Claude Haiku in the `rich` style instead of from templates:
facts may be merged into one sentence or restated with a pronoun, and two or three background sentences
of general context sit between them. 145 documents, about 43 words each, every fact once.

Haiku answered through semula (a recording OpenAI-compatible server), so there was no API spend. The
145 answers are in `cache/`; nothing the model said is trusted (`kggen.render_llm.parse_rich` checks each
fact appears once, its object and years are in its sentence, its subject has been named, and no other
entity of the world is named), and mention offsets are found by the code, so the gold is as exact as the
template renderer's.

Replay with no endpoint (0 calls) and get `documents.json` back byte for byte:

    kggen docs generate --spec datagen/examples/hard-small/spec.yaml --renderer llm --style rich \
      --model claude-haiku --cache-dir datagen/examples/hard-small/cache

What it is not: Haiku mostly names entities again instead of using pronouns (5 of 602 sentences have
fewer than two names), so coreference is lightly tested; and the facts are the same synthetic ones, so
it is harder prose, not harder reasoning.

Two of the 145 articles add a small claim that was not in the facts ("at another point", "which has drawn attention from
several buyers"). Neither names an entity, so the gold is unaffected, but the text says slightly more than the gold does.
