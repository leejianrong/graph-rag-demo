"""kgbench may use graph_rag only through its ports and public entry points (ADR-0011).

Ground truth must not depend on our extraction, so the extraction stages, adapters and
the retiring ``graph_rag.benchmark`` package are off limits to the bench.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "kgbench"

ALLOWED = {
    "graph_rag.ports",
    "graph_rag.models",
    "graph_rag.fakes",
    "graph_rag.config",
    "graph_rag.ids",
    "graph_rag.logging",
    "graph_rag.orchestrator",
    "graph_rag.query",
    "graph_rag.main",
}


def _graph_rag_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "graph_rag"}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "graph_rag":
            found.add(node.module or "")
    return found


def _allowed(module: str) -> bool:
    return any(module == a or module.startswith(a + ".") for a in ALLOWED)


def test_bench_imports_only_allowed_graph_rag_modules() -> None:
    offenders = {
        f"{path.relative_to(PACKAGE)}: {module}"
        for path in PACKAGE.rglob("*.py")
        for module in _graph_rag_imports(path)
        if not _allowed(module)
    }
    assert not offenders, f"kgbench imports off-limits graph_rag modules: {sorted(offenders)}"


def test_boundary_check_catches_a_violation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("from graph_rag.stages.ner import SpacyNerStage\n")
    assert [m for m in _graph_rag_imports(bad) if not _allowed(m)] == ["graph_rag.stages.ner"]
