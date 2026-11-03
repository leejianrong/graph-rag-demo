"""``python -m kggen`` entry point (delegates to the CLI)."""

from __future__ import annotations

from kggen.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
