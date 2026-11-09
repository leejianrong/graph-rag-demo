"""Rewiring a graph's edges when duplicate entities are merged into one (GRA-30).

Shared by the in-memory graph store and the Neo4j adapter, so both treat a merge the same way.
Provenance, dates and validity travel with each edge untouched; only its endpoints change.
"""

from __future__ import annotations

from graph_rag.models import Triple

__all__ = ["edge_key", "rewire"]


def edge_key(edge: Triple) -> tuple[str, str, str, str, int]:
    """What identifies one stored edge: its endpoints, predicate and the sentence it came from."""
    return (
        edge.subject_id,
        edge.predicate,
        edge.object_id,
        edge.provenance.source_doc_id,
        edge.provenance.sentence_index,
    )


def rewire(edges: list[Triple], keep: str, drop: set[str], present: list[Triple]) -> list[Triple]:
    """The edges to add to ``keep`` so ``drop``'s edges survive the merge.

    Each edge that touches a dropped id is moved onto ``keep``. An edge that becomes a loop on
    ``keep`` (it joined two entities that are now one) is left out, and so is one ``present``
    already holds, or one repeated within ``edges``. Order follows ``edges``.
    """
    seen = {edge_key(edge) for edge in present}
    out: list[Triple] = []
    for edge in edges:
        moved = edge.model_copy(
            update={
                "subject_id": keep if edge.subject_id in drop else edge.subject_id,
                "object_id": keep if edge.object_id in drop else edge.object_id,
            }
        )
        if moved.subject_id == moved.object_id or edge_key(moved) in seen:
            continue
        seen.add(edge_key(moved))
        out.append(moved)
    return out
