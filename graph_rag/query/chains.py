"""Order a subgraph's edges so the chain that best answers the question comes first (GRA-170).

A question over several hops is answered by a chain of connected edges, and no single edge's
sentence looks like the question: the sentence for the last hop names an entity the question never
mentions. So edges ranked one at a time bury a long chain among the dozens of edges a 2-hop
subgraph holds. This ranks chains instead: a beam search walks outward from the seed entities, one
edge per step, and scores each chain by the cosine between the question and its sentences joined.

The beam search needs no LLM and no new store: it uses the retriever's embedder on the sentences
the edges already carry. It costs one batch of embeddings per step, at most ``beam`` chains grown
by the node's degree, so it is opt-in (``PATH_RETRIEVAL``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_rag.models import Subgraph, Triple

__all__ = ["chain_first"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors; the embedders return unit-length vectors."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def chain_first(
    subgraph: Subgraph,
    seed_ids: Iterable[str],
    question_vector: list[float],
    embed: Callable[[list[str]], list[list[float]]],
    *,
    beam: int = 8,
    depth: int = 4,
    fanout: int = 20,
    sentence_vectors: dict[str, list[float]] | None = None,
) -> list[Triple]:
    """Return ``subgraph.edges`` with the edges of the best-scoring chains first.

    Chains start at an edge touching a seed and grow one edge at a time from the far end, up to
    ``depth`` edges, keeping the ``beam`` best per length (undirected, no edge twice). All chains
    of every length compete on one score, so the question decides how long the chain is. The edges
    are listed in the order their chain ranks, each once; edges on no chain follow in their
    original order. With no seed in the subgraph the order is unchanged.

    A node with many edges (a country, a common first name) would add a candidate chain, and so
    an embedding, for each of them. Only the ``fanout`` edges at a node whose own sentence is
    closest to the question are tried; a node with no more edges than that is unaffected.

    Args:
        subgraph: The retrieved subgraph.
        seed_ids: The entity seeds the chains start from.
        question_vector: The embedded question (unit length).
        embed: Embeds a batch of texts to unit-length vectors (the retriever's embedder).
        beam: How many chains to keep per length.
        depth: The longest chain, in edges.
        fanout: The most edges tried at one node, best sentence first.
        sentence_vectors: A cache of sentence vectors the caller keeps between questions, so a
            sentence is embedded once; a fresh one is used when omitted.
    """
    edges = list(subgraph.edges)
    cache = {} if sentence_vectors is None else sentence_vectors
    adjacent: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, edge in enumerate(edges):
        adjacent[edge.subject_id].append((i, edge.object_id))
        adjacent[edge.object_id].append((i, edge.subject_id))

    def sentence(i: int) -> str:
        return edges[i].provenance.source_sentence.strip()

    def closest(extensions: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """The ``fanout`` extensions whose own sentence is nearest the question."""
        if len(extensions) <= fanout:
            return extensions
        missing = sorted({sentence(i) for i, _ in extensions if sentence(i) not in cache})
        if missing:
            cache.update(zip(missing, embed(missing), strict=True))
        near = {i: _cosine(question_vector, cache[sentence(i)]) for i, _ in extensions}
        return sorted(extensions, key=lambda item: (-near[item[0]], item[0]))[:fanout]

    level: list[tuple[tuple[int, ...], str]] = [
        ((i,), far) for seed in sorted(set(seed_ids)) for i, far in closest(adjacent.get(seed, []))
    ]
    ranked: list[tuple[float, tuple[int, ...]]] = []
    for _ in range(depth):
        if not level:
            break
        texts = [" ".join(sentence(i) for i in chain) for chain, _ in level]
        vectors = embed(texts)
        scored = sorted(
            (
                (_cosine(question_vector, v), chain, tail)
                for v, (chain, tail) in zip(vectors, level, strict=True)
            ),
            key=lambda item: (-item[0], item[1]),
        )[:beam]
        ranked.extend((score, chain) for score, chain, _ in scored)
        level = [
            (chain + (i,), far)
            for _, chain, tail in scored
            for i, far in closest([(i, far) for i, far in adjacent.get(tail, []) if i not in chain])
        ]

    ranked.sort(key=lambda item: (-item[0], item[1]))
    order: list[int] = []
    seen: set[int] = set()
    for _, chain in ranked:
        for i in chain:
            if i not in seen:
                seen.add(i)
                order.append(i)
    order.extend(i for i in range(len(edges)) if i not in seen)
    return [edges[i] for i in order]
