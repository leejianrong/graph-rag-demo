"""kggen is standalone: it must not import, or depend on, the pipeline that consumes its data."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"graph_rag", "kgbench", "litellm"}


def test_kggen_imports_nothing_from_graph_rag_or_the_bench() -> None:
    offenders: dict[str, set[str]] = {}
    for path in (ROOT / "kggen").rglob("*.py"):
        tree = ast.parse(path.read_text())
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        if roots & FORBIDDEN:
            offenders[str(path.relative_to(ROOT))] = roots & FORBIDDEN
    assert not offenders


def test_kggen_declares_no_dependency_on_the_pipeline() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert not [d for d in deps if any(f.replace("_", "-") in d.lower() for f in FORBIDDEN)]
    assert "graph-rag-demo" not in " ".join(deps)
