"""Fit a recall result into a token budget (GRA-38, ADR-0014).

The budget covers the evidence: ``supporting_sentences``, ``ranked_nodes`` and the ``subgraph``'s
nodes and edges. ``answer``, ``answer_entity``, ``answer_entities`` and ``prose`` are never cut.
An item costs the length of its JSON, and tokens are estimated at three characters each, so the
count is conservative for English (about four characters a token in practice) but not any
model's exact tokenizer. The envelope around the items (braces, field names of the response itself) is not
counted.

Each list is cut to its best-ranked prefix, so what stays is always the top of the ranking. The
budget is spent on sentences first (at most 60% of it), then ranked nodes (at most 25% of it,
and no more than is left), then the subgraph's edges with the nodes they join.
"""

from __future__ import annotations

import math

from graph_rag.models import CanonicalEntity, QueryResponse, Subgraph, Triple

__all__ = ["CHARS_PER_TOKEN", "evidence_tokens", "pack"]

CHARS_PER_TOKEN = 3
_SENTENCE_SHARE = 0.60
_NODE_SHARE = 0.25


def _chars(item: object) -> int:
    return len(item.model_dump_json())  # type: ignore[attr-defined,no-any-return]


def _tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def evidence_tokens(response: QueryResponse) -> int:
    """The estimated tokens of the response's evidence items."""
    items = [*response.supporting_sentences, *response.ranked_nodes]
    items += [*response.subgraph.nodes, *response.subgraph.edges]
    return _tokens(sum(_chars(i) for i in items))


def _prefix(items: list, allowance: int) -> tuple[list, int]:
    """The longest leading run of ``items`` costing at most ``allowance`` characters."""
    kept, spent = [], 0
    for item in items:
        cost = _chars(item)
        if spent + cost > allowance:
            break
        kept.append(item)
        spent += cost
    return kept, spent


def pack(response: QueryResponse, budget: int) -> tuple[QueryResponse, bool]:
    """``response`` cut to fit ``budget`` tokens, and whether anything was cut.

    The result's :func:`evidence_tokens` never exceeds ``budget``.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    total = budget * CHARS_PER_TOKEN

    sentences, spent = _prefix(response.supporting_sentences, int(total * _SENTENCE_SHARE))
    nodes, node_spent = _prefix(response.ranked_nodes, min(total - spent, int(total * _NODE_SHARE)))
    spent += node_spent

    rank = {n.canonical_id: i for i, n in enumerate(response.ranked_nodes)}
    by_id = {n.canonical_id: n for n in response.subgraph.nodes}
    worst = len(rank)

    def order(edge: Triple) -> int:
        return min(rank.get(edge.subject_id, worst), rank.get(edge.object_id, worst))

    edges: list[Triple] = []
    graph_nodes: dict[str, CanonicalEntity] = {}
    for edge in sorted(response.subgraph.edges, key=order):
        ends = [by_id[i] for i in (edge.subject_id, edge.object_id) if i in by_id]
        new_nodes = {n.canonical_id: n for n in ends if n.canonical_id not in graph_nodes}
        cost = _chars(edge) + sum(_chars(n) for n in new_nodes.values())
        if spent + cost > total:
            break
        spent += cost
        edges.append(edge)
        graph_nodes.update(new_nodes)

    packed = response.model_copy(
        update={
            "supporting_sentences": sentences,
            "ranked_nodes": nodes,
            "subgraph": Subgraph(nodes=list(graph_nodes.values()), edges=edges),
        }
    )
    cut = (
        len(sentences) < len(response.supporting_sentences)
        or len(nodes) < len(response.ranked_nodes)
        or len(edges) < len(response.subgraph.edges)
    )
    return packed, cut
