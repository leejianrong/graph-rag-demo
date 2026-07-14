"""Pipeline stages — the modular, swappable enrichment steps (ADR-0001).

Each stage runs behind a narrow interface and is constructor-injected into the
:class:`~graph_rag.orchestrator.Orchestrator`, so the fast suite can swap a real
stage for a canned fake without touching pipeline code. V2 introduces the first
stage, :mod:`graph_rag.stages.ner`.
"""

from __future__ import annotations
