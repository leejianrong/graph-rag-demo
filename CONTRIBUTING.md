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

- One branch per change, off fresh `main`; `main` is protected, so everything lands by pull request.
- Merge (squash) once `lint`, `fast`, `contract` and `secret-scan` are green on the commit you pushed.
- Commit with your GitHub noreply address; pushes with a private address are rejected.
- Update the docs, README and `AGENTS.md` in the same PR that changes what they describe.
- A change that alters live behaviour ships behind a flag that defaults to the current behaviour.

## Two rules worth knowing

- The generator, `datagen/`, must never import `graph_rag`, `kgbench` or `litellm`; the bench may use
  `graph_rag` only through its ports and public entry points. Tests enforce both.
- Tests assert behaviour at the port seam, using the in-memory fakes in `graph_rag/fakes.py`.
