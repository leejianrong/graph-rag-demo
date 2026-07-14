"""Unit tests for the V4 entity-linking fakes (fast, offline, ``$0``).

Exercises the deterministic :class:`~graph_rag.fakes.FakeEmbedder` and the
in-memory :class:`~graph_rag.fakes.InMemoryEntityStore` at the port seam — the
backbone the EL fast E2E runs against (ADR-0010). No model download, no Docker.
"""

from __future__ import annotations

import math

from graph_rag.fakes import FakeEmbedder, InMemoryEntityStore
from graph_rag.models import CanonicalEntity
from graph_rag.normalize import normalize_name

# --- FakeEmbedder -----------------------------------------------------------


def test_embedder_dim_default_and_custom() -> None:
    """``dim`` defaults to B1's 384 and honours an override; vectors match it."""
    assert FakeEmbedder().dim == 384
    small = FakeEmbedder(dim=16)
    assert small.dim == 16
    [vec] = small.embed(["hello world"])
    assert len(vec) == 16


def test_embedder_is_deterministic() -> None:
    """Identical text embeds to an identical vector across calls and instances."""
    a = FakeEmbedder(dim=64).embed(["Apple Inc."])[0]
    b = FakeEmbedder(dim=64).embed(["Apple Inc."])[0]
    assert a == b


def test_embedder_batch_preserves_order() -> None:
    """embed returns one vector per input, in input order."""
    embedder = FakeEmbedder(dim=32)
    texts = ["one", "two", "three"]
    vectors = embedder.embed(texts)
    assert len(vectors) == len(texts)
    # The batched vector equals the singly-embedded vector for each text.
    for text, vector in zip(texts, vectors, strict=True):
        assert vector == embedder.embed([text])[0]


def test_embedder_vectors_are_unit_norm() -> None:
    """Every vector is L2-normalized (norm ~= 1)."""
    for vec in FakeEmbedder(dim=48).embed(["London", "", "!!!"]):
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_embedder_normalization_makes_variants_identical() -> None:
    """Case/punctuation/whitespace variants normalize equal → identical vectors."""
    embedder = FakeEmbedder(dim=128)
    base = embedder.embed(["Apple Inc."])[0]
    for variant in ("apple inc", "APPLE   INC", "Apple, Inc.!"):
        assert normalize_name(variant) == normalize_name("Apple Inc.")
        assert embedder.embed([variant])[0] == base


def test_embedder_shared_tokens_score_higher_than_disjoint() -> None:
    """Texts sharing normalized tokens are more similar than disjoint ones."""
    embedder = FakeEmbedder(dim=256)
    barack = embedder.embed(["Barack Obama"])[0]
    obama_barack = embedder.embed(["Obama, Barack"])[0]  # same tokens, reordered
    unrelated = embedder.embed(["Microsoft Corporation"])[0]

    def cos(u: list[float], v: list[float]) -> float:
        return sum(a * b for a, b in zip(u, v, strict=True))

    # Same tokens (order-independent bag) → identical vector → cosine 1.0.
    assert math.isclose(cos(barack, obama_barack), 1.0, rel_tol=1e-9)
    assert cos(barack, unrelated) < cos(barack, obama_barack)


# --- InMemoryEntityStore ----------------------------------------------------


def _entity(cid: str, name: str, type_: str = "ORG", **kw: object) -> CanonicalEntity:
    return CanonicalEntity(canonical_id=cid, name=name, type=type_, **kw)  # type: ignore[arg-type]


def test_store_upsert_is_idempotent_by_canonical_id() -> None:
    """Re-upserting the same ``canonical_id`` overwrites, never duplicates."""
    store = InMemoryEntityStore()
    store.upsert(_entity("e1", "Apple Inc."))
    store.upsert(_entity("e1", "Apple Inc.", aliases=["Apple"]))

    assert store.count() == 1
    fetched = store.get("e1")
    assert fetched is not None
    assert fetched.aliases == ["Apple"]


def test_store_get_missing_returns_none_and_count_all() -> None:
    """get of an unknown id is None; count/all reflect what was upserted."""
    store = InMemoryEntityStore()
    assert store.get("nope") is None
    assert store.count() == 0
    assert store.all() == []

    e1, e2 = _entity("e1", "Apple"), _entity("e2", "Microsoft")
    store.upsert(e1)
    store.upsert(e2)
    assert store.count() == 2
    assert store.all() == [e1, e2]


def test_block_candidates_matches_type_and_normalized_name() -> None:
    """Blocking returns only same-type entities whose name normalizes to the key."""
    store = InMemoryEntityStore()
    store.upsert(_entity("e1", "Apple, Inc.", type_="ORG"))
    store.upsert(_entity("e2", "Apple", type_="ORG"))  # different normalized name
    store.upsert(_entity("e3", "Apple Inc", type_="PRODUCT"))  # matches name, wrong type

    hits = store.block_candidates(entity_type="ORG", normalized_name=normalize_name("Apple Inc."))
    assert [e.canonical_id for e in hits] == ["e1"]


def test_block_candidates_matches_on_aliases() -> None:
    """An alias whose normalized form matches the key blocks in (name need not)."""
    store = InMemoryEntityStore()
    # Name normalizes to "apple computer inc" — the KEY only matches via the alias.
    store.upsert(_entity("e1", "Apple Computer Inc.", type_="ORG", aliases=["Apple Inc."]))

    hits = store.block_candidates(entity_type="ORG", normalized_name=normalize_name("APPLE, INC."))
    assert normalize_name("APPLE, INC.") == "apple inc"
    assert [e.canonical_id for e in hits] == ["e1"]


def test_block_candidates_empty_when_nothing_matches() -> None:
    """No same-type name/alias match → no candidates (drives create-new)."""
    store = InMemoryEntityStore()
    store.upsert(_entity("e1", "Apple", type_="ORG"))
    assert store.block_candidates(entity_type="PERSON", normalized_name="apple") == []
    assert store.block_candidates(entity_type="ORG", normalized_name="banana") == []


def test_knn_orders_by_cosine_descending() -> None:
    """kNN returns ``(entity, score)`` pairs ordered by descending cosine."""
    embedder = FakeEmbedder(dim=256)
    store = InMemoryEntityStore()
    store.upsert(_entity("apple", "Apple Inc.", vector=embedder.embed(["Apple Inc."])[0]))
    store.upsert(_entity("ms", "Microsoft", vector=embedder.embed(["Microsoft Corp"])[0]))

    query = embedder.embed(["Apple Incorporated"])[0]
    ranked = store.knn(vector=query, top_k=2)

    assert [e.canonical_id for e, _ in ranked] == ["apple", "ms"]
    # Scores are descending and the top score beats the runner-up.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[1]


def test_knn_respects_top_k_and_type_filter_and_skips_vectorless() -> None:
    """kNN caps at ``top_k``, filters by type, and skips entities with no vector."""
    embedder = FakeEmbedder(dim=64)
    store = InMemoryEntityStore()
    store.upsert(_entity("p1", "Ada Lovelace", type_="PERSON", vector=embedder.embed(["Ada"])[0]))
    store.upsert(_entity("o1", "Apple", type_="ORG", vector=embedder.embed(["Apple"])[0]))
    store.upsert(_entity("o2", "Novec", type_="ORG"))  # no vector → skipped

    query = embedder.embed(["Apple"])[0]
    org_hits = store.knn(vector=query, entity_type="ORG", top_k=5)
    assert [e.canonical_id for e, _ in org_hits] == ["o1"]  # o2 skipped, p1 filtered out

    capped = store.knn(vector=query, top_k=1)
    assert len(capped) == 1
