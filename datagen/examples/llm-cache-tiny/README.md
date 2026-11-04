# LLM render cache for the `tiny` preset

Answers for `kggen docs generate --preset tiny --renderer llm --model claude-haiku` (render seed 0).
Each `*.json` records the model, the full prompt and the response. They were written by a Claude Haiku
agent answering through semula (a recording OpenAI-compatible server), not by a paid API run.

Replay with no endpoint (0 calls):

    kggen docs generate --preset tiny --renderer llm --model claude-haiku --cache-dir datagen/examples/llm-cache-tiny
