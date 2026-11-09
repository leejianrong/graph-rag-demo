"""Answer as of a date: keep the edges that held then (GRA-44).

An edge holds on a day when ``valid_from <= day < valid_to``. An unknown bound (``None``) does not
exclude: the pipeline only knows a date when the caller gave one, so an undated fact is kept rather
than guessed away. Dates are ``YYYY-MM-DD`` strings, which sort the way time does.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from graph_rag.models import Subgraph, SupportingSentence, Triple


def holds(edge: Triple, day: str) -> bool:
    """True when ``edge`` held on ``day`` (a bound that is unknown does not exclude it)."""
    return (edge.valid_from is None or edge.valid_from <= day) and (
        edge.valid_to is None or day < edge.valid_to
    )


def _evidence(edge: Triple) -> tuple[str, int]:
    return (edge.provenance.source_doc_id, edge.provenance.sentence_index)


def has_ended(edge: Triple, day: str) -> bool:
    """True when ``edge`` stopped holding on or before ``day`` (an unknown end has not ended)."""
    return edge.valid_to is not None and edge.valid_to <= day


def subgraph_current(
    subgraph: Subgraph, seed_ids: list[str], hops: int, today: str
) -> tuple[Subgraph, set[tuple[str, int]]]:
    """The slice of ``subgraph`` with the edges that have ended removed (GRA-43).

    The default view when no ``as_of`` is given: a superseded fact stays in the graph as history
    but does not come back next to the one that replaced it. Unlike ``as_of`` this does not hide an
    edge for starting later than today, so a document dated ahead is still found.
    """
    return _restrict(subgraph, seed_ids, hops, lambda e: not has_ended(e, today))


def subgraph_as_of(
    subgraph: Subgraph, seed_ids: list[str], hops: int, day: str
) -> tuple[Subgraph, set[tuple[str, int]]]:
    """The slice of ``subgraph`` that held on ``day``, and the sentences only excluded edges cite.

    Edges that did not hold are dropped. A node stays when it is reachable from a seed within
    ``hops`` over the edges that remain, or when it had no edges at all (nothing dates it). A seed
    whose every edge was excluded goes too: no fact about it held then. The second result is the
    ``(document_id, sentence_index)`` of each excluded edge's sentence that no kept edge also
    cites, for the caller to drop from the evidence.
    """
    return _restrict(subgraph, seed_ids, hops, lambda e: holds(e, day))


def _restrict(
    subgraph: Subgraph, seed_ids: list[str], hops: int, keep: Callable[[Triple], bool]
) -> tuple[Subgraph, set[tuple[str, int]]]:
    """Keep the edges ``keep`` accepts and what still hangs off the seeds (see the callers)."""
    kept = [e for e in subgraph.edges if keep(e)]
    dropped = [e for e in subgraph.edges if not keep(e)]
    had_edges = {n for e in subgraph.edges for n in (e.subject_id, e.object_id)}
    node_ids = {n.canonical_id for n in subgraph.nodes}

    adjacency: dict[str, set[str]] = {cid: set() for cid in node_ids}
    for edge in kept:
        if edge.subject_id in node_ids and edge.object_id in node_ids:
            adjacency[edge.subject_id].add(edge.object_id)
            adjacency[edge.object_id].add(edge.subject_id)

    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for seed in seed_ids:
        if seed in adjacency and seed not in depth and (adjacency[seed] or seed not in had_edges):
            depth[seed] = 0
            queue.append(seed)
    while queue:
        current = queue.popleft()
        if depth[current] == hops:
            continue
        for neighbour in adjacency[current]:
            if neighbour not in depth:
                depth[neighbour] = depth[current] + 1
                queue.append(neighbour)

    nodes = [n for n in subgraph.nodes if n.canonical_id in depth]
    edges = [e for e in kept if e.subject_id in depth and e.object_id in depth]
    gone = {_evidence(e) for e in dropped} - {_evidence(e) for e in edges}
    return Subgraph(nodes=nodes, edges=edges), gone


def drop_sentences(
    sentences: list[SupportingSentence], gone: set[tuple[str, int]]
) -> list[SupportingSentence]:
    """Sentences minus the ones whose only supporting edges did not hold (see above)."""
    return [s for s in sentences if (s.document_id, s.sentence_index) not in gone]
