# Quickstart: give an agent a memory in five minutes

This runs the service in one process with no Docker, model download or LLM key, then connects Claude Code to it.
The offline service uses simple rules to find entities and facts, so its answers are rougher than the full
stack's (see "Running it locally" in the README). The evidence it returns is what matters here. It keeps
nothing: stop it and the memory is gone.

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

## 1. Install and start the service

```bash
git clone https://github.com/leejianrong/graph-rag-demo && cd graph-rag-demo
uv sync --extra cli --extra mcp
uv run graph-rag-offline          # serves http://127.0.0.1:8000; leave it running
```

The web UI is at http://127.0.0.1:8000/ once it has been built (`make ui-build`; the Docker image has it built in, so after `docker compose up` open http://localhost:8000/). Sign in with an API key (step 4 shows how to make one).

## 2. Remember something and ask about it

In a second terminal, in the same directory:

```bash
uv run graphrag remember "Ada Lovelace works for Acme Corp. Acme Corp is based in London." --source-id note-1
uv run graphrag recall "Where is Acme Corp based?"
```

```text
accepted: note-1 (document 6f64139b9258e1eee71ff8d5df47d9845b63bc9dbfcdb7d9978a6000a2cc6cfb)

answer: London

sources:
  1. Acme Corp is based in London.
     document 6f64139b..., characters 34-63  score 0.73
  2. Ada Lovelace works for Acme Corp.
     document 6f64139b..., characters 0-33  score 0.37
```

Each source carries the document and character offsets it came from, so an agent can cite it.
`uv run graphrag lookup --name "Acme Corp"` shows an entity's neighbouring facts, and
`uv run graphrag forget --source-id note-1` removes the source again. Add `--json` to any command for the API's own
JSON, `--namespace my-project` to keep one project's memory apart, and `--token-budget 500` to `recall` to cap how
much evidence comes back.

## 3. Connect Claude Code

```bash
claude mcp add graph-rag --env GRAPH_RAG_URL=http://localhost:8000 -- uv run --directory "$PWD" graph-rag-mcp
```

Start Claude Code and ask it to remember a fact, then to recall it in a new conversation. It has four tools:
`remember`, `recall`, `forget` and `lookup`. The same four are plain HTTP under `/v1`
(`docs/api/agent-api.openapi.yaml`):

```bash
curl -X POST localhost:8000/v1/recall -H 'content-type: application/json' \
  -d '{"question": "Where is Acme Corp based?"}'
```

## 4. Optional: require an API key

Stop the service, then:

```bash
uv run graph-rag-keys create my-team --label laptop        # prints the key once; only its hash is stored
REQUIRE_AUTH=true uv run graph-rag-offline
export GRAPH_RAG_API_KEY=grk_...                            # the key it printed; the CLI and MCP server send it
```

Every endpoint except `/health` now answers 401 without the key, and a key reaches only its own tenant's memory.
Pass `--env GRAPH_RAG_API_KEY=grk_...` to `claude mcp add` for the agent.

## Going further

For real extraction over Elasticsearch, Neo4j, MinIO and Valkey, run `docker compose up --build` (README, "Running
it locally") and use the same commands against it.

### Give ingest an LLM

This quickstart needs no LLM because `graph-rag-offline` extracts with simple rules. The full stack does:
coreference and fact extraction call one, so `/ingest` and `/v1/remember` (CLI `remember`, MCP `remember`)
are refused with `503` and `details.reason` `llm_not_configured` until you set one up, and nothing is stored.
`/query`, `/v1/recall`, `/v1/lookup` and `/v1/forget` never need an LLM, and neither does the offline server.
Put one of these in `.env` and restart:

- `OPENAI_API_KEY=...` for OpenAI (the default `gpt-4o-mini`), or another provider's key with `COREF_MODEL`
  and `KG_BUILD_MODEL` set to a model that provider serves.
- `OPENAI_API_BASE=http://host:port/v1` for any OpenAI-compatible endpoint: a hosted provider, or a local
  server such as Ollama or vLLM (set `OPENAI_API_KEY` to any word if the server ignores it).
- semula, a recording OpenAI-compatible server whose prompts are answered by an agent. It suits the bench and
  agent-driven sessions; it is not an unattended local model.

If the endpoint is set but nothing answers (a closed port, a wrong host, a timeout), the document is accepted
and its job fails with, for example, `coreference failed: InternalServerError. The LLM endpoint at
llm.internal is unreachable (check OPENAI_API_BASE).` Only the host is shown, never the URL or a key.
Post the document again once the endpoint is up.

Cost on a small hosted model is roughly $0.2 to $0.3 per 1,000 small documents, uncached (an estimate).

## What is tested

Steps 1 and 2 and the API-key step run in CI against the same app `graph-rag-offline` serves
(`tests/unit/test_quickstart.py`), and the MCP tools are tested through the real protocol
(`tests/unit/test_mcp_server.py`). Registering the server with Claude Code (step 3) is checked by hand.
