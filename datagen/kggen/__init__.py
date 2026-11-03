"""kggen — a standalone, seeded generator of synthetic knowledge graphs and their documents.

A ``WorldSpec`` plus a seed fully determines a world: entities, typed relations, and (in later
slices) documents and multi-hop questions with gold paths. kggen knows nothing about graph-rag
or any other pipeline; it writes plain files that a pipeline, or the evaluation bench, consumes.
``tests/test_isolation.py`` enforces that it never imports ``graph_rag``.
"""

__version__ = "0.1.0"
