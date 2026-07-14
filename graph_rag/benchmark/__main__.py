"""``python -m graph_rag.benchmark`` entry point (delegates to the CLI, U4)."""

from __future__ import annotations

from graph_rag.benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
