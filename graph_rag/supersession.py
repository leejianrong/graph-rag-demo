"""Close the interval of a fact that a newer one replaces (GRA-43).

A document that says Ada works for Initech, dated after one that says she works for Acme, ends the
Acme fact on that date. Nothing is deleted: the older edge keeps its provenance and gets a
``valid_to``, so it stays in the graph as history and ``as_of`` can still find it.

Only predicates that hold one object at a time are superseded. A person can have founded two
companies, so a new FOUNDED edge replaces nothing; WORKS_FOR, LOCATED_IN, HAS_ROLE and GOVERNED_BY
name one current answer. That list is my reading of the closed predicate set, not measured.

The function is pure so it can be tested without a store: it takes the new edges and the edges
already there and returns what to write and what to close.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_rag.models import Triple

SINGLE_VALUED: frozenset[str] = frozenset({"LOCATED_IN", "WORKS_FOR", "HAS_ROLE", "GOVERNED_BY"})


@dataclass(frozen=True)
class Closure:
    """An existing edge to end on ``valid_to``, because ``by_doc`` states a newer fact."""

    edge: Triple
    valid_to: str
    by_doc: str


def supersede(
    new: list[Triple], existing: list[Triple], doc_id: str
) -> tuple[list[Triple], list[Closure]]:
    """The new edges (some possibly already ended) and the older edges they close.

    For a new edge with a known ``valid_from`` and a single-valued predicate, any other edge with
    the same subject and predicate but a different object, from another document, is compared by
    start date. An earlier-or-undated one that is still open (or ends later) is closed on the new
    edge's start. A later one means the new edge is the older fact: it ends where that one starts.
    Equal start dates, an unknown date on the new edge, and edges from the same document are left
    alone, since order cannot be told.
    """
    closures: dict[tuple[str, str, str, str, int], Closure] = {}
    written: list[Triple] = []
    for edge in new:
        ends: tuple[str, str] | None = None  # (date, doc) that end this new edge
        if edge.predicate in SINGLE_VALUED and edge.valid_from is not None:
            for other in existing:
                if (
                    other.subject_id != edge.subject_id
                    or other.predicate != edge.predicate
                    or other.object_id == edge.object_id
                    or other.provenance.source_doc_id == doc_id
                ):
                    continue
                if other.valid_from is None or other.valid_from < edge.valid_from:
                    if other.valid_to is None or other.valid_to > edge.valid_from:
                        key = (
                            other.subject_id,
                            other.predicate,
                            other.object_id,
                            other.provenance.source_doc_id,
                            other.provenance.sentence_index,
                        )
                        closures[key] = Closure(other, edge.valid_from, doc_id)
                elif other.valid_from > edge.valid_from and (
                    ends is None or other.valid_from < ends[0]
                ):
                    ends = (other.valid_from, other.provenance.source_doc_id)
        if ends is not None and (edge.valid_to is None or ends[0] < edge.valid_to):
            edge = edge.model_copy(update={"valid_to": ends[0], "superseded_by": ends[1]})
        written.append(edge)
    return written, list(closures.values())
