"""Query-side (read path) package for Graph RAG retrieval (V6, ADR-0007).

Holds the ``/query`` path pieces: :mod:`graph_rag.query.ranking` — the pinned,
I/O-free subgraph ranking function (B4) and answer/sentence selection — and
:mod:`graph_rag.query.retriever` — the :class:`QueryRetriever` (N16) that composes
the ranker over the live query-side ports (Embedder + EntityStore.knn +
DocumentStore.search_sentences + GraphStore.khop) for the deterministic, ``$0``
seed → expand → rank → answer flow the FastAPI ``/query`` endpoint serves.
"""

from __future__ import annotations

from graph_rag.query.ranking import rank_nodes, rank_sentences, select_answer
from graph_rag.query.retriever import QueryRetriever

__all__ = ["rank_nodes", "select_answer", "rank_sentences", "QueryRetriever"]
