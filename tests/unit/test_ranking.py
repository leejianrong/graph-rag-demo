"""Unit tests for the pinned V6 subgraph ranking function (B4, fast, offline).

Exercises the pure scoring/selection helpers in :mod:`graph_rag.query.ranking`
directly (no I/O, no ports), asserting the B4 ordering. This is the test that was
blocked in TESTING §5 until B4 was pinned — it now asserts the ordering of the
documented formula:

    score = w_seed * seed_similarity + w_prox * 1/(1 + hop_distance)   (0.7 / 0.3)
"""

from __future__ import annotations

from graph_rag.models import CanonicalEntity, RankedNode, Subgraph, SupportingSentence
from graph_rag.query.ranking import rank_nodes, rank_sentences, select_answer


def _entity(canonical_id: str, name: str, type_: str = "PERSON") -> CanonicalEntity:
    """Build a bare :class:`CanonicalEntity` node for the subgraph."""
    return CanonicalEntity(canonical_id=canonical_id, name=name, type=type_)


# --- rank_nodes -------------------------------------------------------------


def test_seed_and_close_node_outranks_far_node() -> None:
    """A seed-similar + graph-close node outranks a far, non-seed node."""
    subgraph = Subgraph(
        nodes=[
            _entity("e1", "Alice"),
            _entity("e2", "Bob"),
            _entity("e3", "Carol"),
        ]
    )
    seed_scores = {"e1": 0.9}  # e1 is the seed
    hop_distance = {"e1": 0.0, "e2": 1.0, "e3": 3.0}

    ranked = rank_nodes(subgraph, seed_scores, hop_distance=hop_distance)

    # e1: 0.7*0.9 + 0.3*1.0     = 0.93
    # e2: 0.7*0.0 + 0.3*(1/2)   = 0.15
    # e3: 0.7*0.0 + 0.3*(1/4)   = 0.075
    assert [n.canonical_id for n in ranked] == ["e1", "e2", "e3"]
    assert ranked[0].score > ranked[1].score > ranked[2].score
    assert abs(ranked[0].score - 0.93) < 1e-9


def test_unreachable_node_has_zero_proximity() -> None:
    """A node absent from ``hop_distance`` gets a 0 proximity term (unreachable)."""
    subgraph = Subgraph(nodes=[_entity("e1", "Alice"), _entity("e2", "Bob")])
    ranked = rank_nodes(
        subgraph,
        seed_scores={"e2": 0.5},
        hop_distance={"e1": 0.0},  # e2 unreachable
    )
    by_id = {n.canonical_id: n for n in ranked}
    assert by_id["e1"].score == 0.3  # 0.7*0 + 0.3*1
    assert abs(by_id["e2"].score - 0.35) < 1e-9  # 0.7*0.5 + 0.3*0


def test_tie_broken_by_canonical_id() -> None:
    """Equal scores are ordered deterministically by ``canonical_id`` ascending."""
    subgraph = Subgraph(nodes=[_entity("z", "Z"), _entity("a", "A"), _entity("m", "M")])
    # All same hop distance, no seeds -> identical scores.
    ranked = rank_nodes(
        subgraph,
        seed_scores={},
        hop_distance={"z": 1.0, "a": 1.0, "m": 1.0},
    )
    assert [n.canonical_id for n in ranked] == ["a", "m", "z"]


def test_custom_weights_respected() -> None:
    """Overriding the weights changes the blend as documented."""
    subgraph = Subgraph(nodes=[_entity("e1", "Alice")])
    ranked = rank_nodes(
        subgraph,
        seed_scores={"e1": 1.0},
        hop_distance={"e1": 1.0},  # proximity 0.5
        w_seed=1.0,
        w_prox=0.0,
    )
    assert ranked[0].score == 1.0


# --- select_answer ----------------------------------------------------------


def test_select_answer_returns_top_node() -> None:
    """``select_answer`` returns the first (top-ranked) node."""
    ranked = [
        RankedNode(canonical_id="e1", name="Alice", type="PERSON", score=0.9),
        RankedNode(canonical_id="e2", name="Acme", type="ORG", score=0.2),
    ]
    answer = select_answer(ranked)
    assert answer is not None
    assert answer.canonical_id == "e1"


def test_select_answer_type_filter_skips_wrong_type() -> None:
    """A type filter skips higher-scoring nodes of the wrong type."""
    ranked = [
        RankedNode(canonical_id="e1", name="Alice", type="PERSON", score=0.9),
        RankedNode(canonical_id="e2", name="Acme", type="ORG", score=0.5),
    ]
    answer = select_answer(ranked, expected_type="ORG")
    assert answer is not None
    assert answer.canonical_id == "e2"


def test_select_answer_empty_returns_none() -> None:
    """No candidates -> ``None``; no matching type -> ``None``."""
    assert select_answer([]) is None
    ranked = [RankedNode(canonical_id="e1", name="Alice", type="PERSON", score=0.9)]
    assert select_answer(ranked, expected_type="ORG") is None


# --- rank_sentences ---------------------------------------------------------


def _sentence(document_id: str, index: int) -> SupportingSentence:
    """Build a :class:`SupportingSentence` with a placeholder score."""
    return SupportingSentence(
        document_id=document_id,
        text=f"sentence {index}",
        char_start=index * 10,
        char_end=index * 10 + 5,
        sentence_index=index,
        score=0.0,
    )


def test_rank_sentences_orders_by_cosine() -> None:
    """Sentences are ordered by cosine to the question, best first."""
    question = [1.0, 0.0, 0.0]
    candidates = [
        (_sentence("d1", 0), [0.0, 1.0, 0.0]),  # orthogonal -> 0.0
        (_sentence("d1", 1), [1.0, 0.0, 0.0]),  # identical  -> 1.0
        (_sentence("d2", 0), [0.7, 0.7, 0.0]),  # ~0.707
    ]
    ranked = rank_sentences(candidates, question)
    assert [(s.document_id, s.sentence_index) for s in ranked] == [
        ("d1", 1),
        ("d2", 0),
        ("d1", 0),
    ]
    assert abs(ranked[0].score - 1.0) < 1e-9


def test_rank_sentences_tie_broken_by_doc_and_index() -> None:
    """Equal cosine ties break by ``document_id`` then ``sentence_index``."""
    question = [1.0, 0.0]
    same = [1.0, 0.0]
    candidates = [
        (_sentence("d2", 5), same),
        (_sentence("d1", 9), same),
        (_sentence("d1", 2), same),
    ]
    ranked = rank_sentences(candidates, question)
    assert [(s.document_id, s.sentence_index) for s in ranked] == [
        ("d1", 2),
        ("d1", 9),
        ("d2", 5),
    ]
