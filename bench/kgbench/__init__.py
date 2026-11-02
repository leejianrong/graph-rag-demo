"""kgbench — an independent knowledge graph evaluation bench (ADR-0011).

Ground truth is a programmatically generated world model, never an extraction of
ours. This package may depend on ``graph_rag`` only through its ports and public
entry points; ``tests/test_import_boundary.py`` enforces that.
"""

__version__ = "0.1.0"
