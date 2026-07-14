"""The pinned subgraph ranking function + selection helpers (V6, B4, ADR-0007).

These functions are the load-bearing scoring that turns a retrieved subgraph +
seed similarities into a ranked list of candidate answer nodes, picks the
top-entity answer, and orders the supporting sentences. They are **pure** — no
I/O, no port access — so the retriever composes them over already-fetched data
and the unit test asserts their ordering directly (unblocking the TESTING §5 V6
ranking-function gap, which was blocked until B4 was pinned).

**B4 — the pinned node ranking formula.** For each node in the subgraph::

    score(node) = w_seed * seed_similarity(node)
                + w_prox * proximity(node)

where ``seed_similarity(node)`` is the node's kNN seed cosine (``0.0`` if the
node was not itself a seed), and ``proximity(node) = 1 / (1 + hop_distance(node))``
is a graph-closeness term that decays with the node's BFS hop distance from the
nearest seed (a seed is hop 0 → proximity ``1.0``; one hop out → ``0.5``; two →
``0.333``; unreachable → ``0.0``). The default weights are ``w_seed = 0.7`` and
``w_prox = 0.3`` (seed similarity dominates, graph proximity breaks near-ties and
rewards multi-hop-connected nodes). Ties are broken deterministically by
``canonical_id`` ascending. The weights are config-overridable
(``Settings.rank_weight_seed`` / ``rank_weight_proximity``) but fixed by default
for benchmark reproducibility (ADR-0009).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from graph_rag.models import RankedNode, SupportingSentence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graph_rag.models import CuratedType, Subgraph

__all__ = ["rank_nodes", "select_answer", "rank_sentences"]

# B4 default weights — see the module docstring. Config can override via
# ``Settings.rank_weight_seed`` / ``Settings.rank_weight_proximity``.
DEFAULT_WEIGHT_SEED = 0.7
DEFAULT_WEIGHT_PROXIMITY = 0.3


def rank_nodes(
    subgraph: Subgraph,
    seed_scores: dict[str, float],
    *,
    hop_distance: dict[str, float],
    w_seed: float = DEFAULT_WEIGHT_SEED,
    w_prox: float = DEFAULT_WEIGHT_PROXIMITY,
) -> list[RankedNode]:
    """Score and rank every node in ``subgraph`` (B4).

    Args:
        subgraph: The k-hop traversal result to rank over (its ``nodes``).
        seed_scores: ``canonical_id`` -> seed kNN cosine similarity for nodes that
            were themselves retrieved as seeds; a node absent here contributes a
            ``0.0`` seed term.
        hop_distance: ``canonical_id`` -> BFS hop distance from the nearest seed
            (seed = ``0``). A node absent here (or mapped to ``inf``) is treated as
            unreachable, so its proximity term is ``0.0``.
        w_seed: Weight on the seed-similarity term (B4 default ``0.7``).
        w_prox: Weight on the graph-proximity term (B4 default ``0.3``).

    Returns:
        The nodes as :class:`~graph_rag.models.RankedNode` s, score descending,
        ties broken by ``canonical_id`` ascending (deterministic).
    """
    ranked: list[RankedNode] = []
    for node in subgraph.nodes:
        seed_term = seed_scores.get(node.canonical_id, 0.0)
        distance = hop_distance.get(node.canonical_id, math.inf)
        prox_term = 1.0 / (1.0 + distance) if distance != math.inf else 0.0
        score = w_seed * seed_term + w_prox * prox_term
        ranked.append(
            RankedNode(
                canonical_id=node.canonical_id,
                name=node.name,
                type=node.type,
                score=score,
            )
        )
    ranked.sort(key=lambda n: (-n.score, n.canonical_id))
    return ranked


def select_answer(
    ranked_nodes: list[RankedNode],
    *,
    expected_type: CuratedType | None = None,
) -> RankedNode | None:
    """Return the top-ranked node — the predicted entity answer (ADR-0007).

    Args:
        ranked_nodes: The output of :func:`rank_nodes` (already score-descending).
        expected_type: When set, only nodes of this curated type are eligible, so
            a type-constrained question ("which PERSON…") skips higher-scoring
            nodes of the wrong type.

    Returns:
        The first eligible node, or ``None`` when there is no candidate.
    """
    for node in ranked_nodes:
        if expected_type is None or node.type == expected_type:
            return node
    return None


def rank_sentences(
    candidates: Iterable[tuple[SupportingSentence, list[float]]],
    question_vector: list[float],
) -> list[SupportingSentence]:
    """Order supporting sentences by cosine similarity to the question.

    Pure re-ranking helper: takes ``(sentence, sentence_vector)`` pairs (a
    :class:`~graph_rag.models.SupportingSentence` lacks its own vector, so the
    caller supplies it), scores each by cosine to ``question_vector``, and returns
    copies with that ``score`` set, sorted score descending then ``document_id``
    then ``sentence_index`` (deterministic — the same tie-break the document store
    uses).

    Args:
        candidates: The sentences to rank, each paired with its dense vector.
        question_vector: The embedded question to score against.

    Returns:
        The re-scored sentences, best cosine first.
    """
    scored = [
        sentence.model_copy(update={"score": _cosine(question_vector, vector)})
        for sentence, vector in candidates
    ]
    scored.sort(key=lambda s: (-s.score, s.document_id, s.sentence_index))
    return scored


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (``0.0`` if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
