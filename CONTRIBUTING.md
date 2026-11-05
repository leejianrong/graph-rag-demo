# Contributing

Thanks for helping. This file is the short version; [AGENTS.md](AGENTS.md) has the rules in full and
[docs/TESTING.md](docs/TESTING.md) the testing approach.

## Set up

```bash
uv sync --frozen --extra dev        # installs the pipeline, the generator (kggen) and the bench (kgbench)
ln -sf ../../scripts/pre-push .git/hooks/pre-push   # the cheap mirror of CI, run before every push
```

Python 3.12 and [uv](https://github.com/astral-sh/uv). Docker is only needed for `make contract` and the
full stack. `make` lists every command.

## Before you push

```bash
make check      # lint, format check and the fast suite: fakes only, no Docker, model or key
```

## Pull requests

- By opening a pull request you license your contribution under Apache-2.0 (section 5 of the licence).
- One branch per change, off fresh `main`; `main` is protected, so everything lands by pull request.
- Merge (squash) once `lint`, `fast`, `contract`, `secret-scan` and `image` are green on the commit you pushed. The rule applies to admins too.
- Commit with your GitHub noreply address; pushes with a private address are rejected.
- Update the docs, README and `AGENTS.md` in the same PR that changes what they describe.
- A change that alters live behaviour ships behind a flag that defaults to the current behaviour.

## Two rules worth knowing

- The generator, `datagen/`, must never import `graph_rag`, `kgbench` or `litellm`; the bench may use
  `graph_rag` only through its ports and public entry points. Tests enforce both.
- Tests assert behaviour at the port seam, using the in-memory fakes in `graph_rag/fakes.py`.

## Tests

The fast suite runs against in-memory fakes: no Docker, no model, no API key. It's
the gate to run before pushing.

```bash
uv sync --frozen --extra dev
uv run pytest -m "not contract and not model and not llm and not benchmark"  # make test
```

The slower layers are opt-in and split by pytest marker: `contract` proves each
real adapter against a throwaway container (MinIO, Elasticsearch, Kafka, Neo4j),
`model` loads real spaCy and embedding models, `llm` hits a real provider and skips
cleanly without a key, and `benchmark` runs the whole pipeline over a fixture.
[AGENTS.md](AGENTS.md) has the full command list and the reasoning behind the
layering. Tests for the generator and the evaluation bench live in `datagen/tests` and `bench/tests` and run in the fast
suite.
