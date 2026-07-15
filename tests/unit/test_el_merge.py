"""Unit tests for the EL merge decision (TESTING §4, ADR-0004) — fast, offline.

Exercises :class:`~graph_rag.stages.entity_linking.EntityLinkingStage`'s
merge-vs-create-new decision at the threshold boundary (B2), normalized-name
blocking candidate selection, ``canonical_id`` reuse + stability, and the
gated-off tie-breaker / NIL paths. Uses a fixed-output embedder so the cosine
score is controlled exactly (no model), and the in-memory
:class:`~graph_rag.fakes.InMemoryEntityStore` (no Docker).
"""

from __future__ import annotations

import math

from graph_rag.fakes import FakeEmbedder, FakeLLMClient, InMemoryEntityStore
from graph_rag.models import CanonicalEntity, CorefCluster, Mention
from graph_rag.stages.entity_linking import EntityLinkingStage

SURFACE = "Apple Inc"
MENTIONS = [Mention(text=SURFACE, type="ORG", char_start=0, char_end=len(SURFACE))]
CLUSTERS = [CorefCluster(canonical=SURFACE, members=[SURFACE])]


class _FixedEmbedder:
    """Deterministic embedder returning one canned unit vector for every text.

    Lets a unit test control the mention vector exactly, so the cosine against a
    preseeded candidate is a known number and the threshold boundary is testable.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dim(self) -> int:
        return len(self._vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _unit(a: float) -> list[float]:
    """A 4-d unit vector ``[a, sqrt(1-a^2), 0, 0]`` — cosine ``a`` with ``[1,0,0,0]``."""
    return [a, math.sqrt(1.0 - a * a), 0.0, 0.0]


def _link_once(store: InMemoryEntityStore, embedder: object, **kwargs: object):
    stage = EntityLinkingStage(store, embedder, **kwargs)  # type: ignore[arg-type]
    return stage.link("", MENTIONS, [], CLUSTERS)


# --- threshold boundary ------------------------------------------------------


def test_merges_when_score_at_or_above_threshold() -> None:
    """A blocked candidate scoring just above the threshold → merge (reuse id)."""
    store = InMemoryEntityStore()
    store.upsert(CanonicalEntity(canonical_id="apple", name=SURFACE, type="ORG", vector=_unit(0.9)))

    result = _link_once(store, _FixedEmbedder([1.0, 0.0, 0.0, 0.0]), threshold=0.85)

    (link,) = result.links
    assert link.is_new is False
    assert link.canonical_id == "apple"
    assert link.score > 0.85
    assert store.count() == 1  # merged, not duplicated


def test_creates_new_when_score_just_below_threshold() -> None:
    """A FUZZY (non-blocking) candidate scoring just below the threshold → create-new.

    The seeded entity has a DIFFERENT normalized name (``"Apple Store"``), so it is
    NOT an exact-key block match — it reaches the decision only via kNN, where the
    cosine gate applies. Below the threshold → the mention mints its own canonical.
    (An exact-name match, by contrast, is decisive and merges regardless of cosine —
    see ``test_exact_name_block_match_merges_below_threshold``.)
    """
    store = InMemoryEntityStore()
    store.upsert(
        CanonicalEntity(canonical_id="store", name="Apple Store", type="ORG", vector=_unit(0.8))
    )

    result = _link_once(store, _FixedEmbedder([1.0, 0.0, 0.0, 0.0]), threshold=0.85)

    (link,) = result.links
    assert link.is_new is True
    assert link.canonical_id != "store"
    assert link.score < 0.85
    assert store.count() == 2  # the fuzzy near-miss becomes its own canonical


def test_exact_name_block_match_merges_below_threshold() -> None:
    """An exact type+normalized-name block match unifies even when the cosine is low.

    Regression guard for cross-document splitting (the "Berlin" bug): the same
    entity's mention-in-context embedding can drift well below the threshold between
    documents, but the exact-key block match is decisive, so it merges into the
    existing canonical instead of minting a divergent duplicate node.
    """
    store = InMemoryEntityStore()
    store.upsert(
        CanonicalEntity(canonical_id="berlin", name=SURFACE, type="ORG", vector=_unit(0.5))
    )

    # Cosine 0.5 is far below the 0.85 threshold, yet the exact name+type block wins.
    result = _link_once(store, _FixedEmbedder([1.0, 0.0, 0.0, 0.0]), threshold=0.85)

    (link,) = result.links
    assert link.is_new is False
    assert link.canonical_id == "berlin"
    assert store.count() == 1  # unified across documents, not duplicated


# --- normalized-name blocking -----------------------------------------------


def test_blocking_selects_the_type_and_name_matching_candidate() -> None:
    """Blocking narrows to the same-type, same-normalized-name candidate only.

    With kNN disabled, only the blocking candidate is scored: an entity with a
    different normalized name (``"Apple Store"``) is never considered, so the
    merge resolves to the correct canonical.
    """
    store = InMemoryEntityStore()
    store.upsert(CanonicalEntity(canonical_id="apple", name=SURFACE, type="ORG", vector=_unit(0.9)))
    store.upsert(
        CanonicalEntity(
            canonical_id="store", name="Apple Store", type="ORG", vector=[1.0, 0.0, 0.0, 0.0]
        )
    )

    result = _link_once(store, _FixedEmbedder([1.0, 0.0, 0.0, 0.0]), threshold=0.85, knn_top_k=0)

    (link,) = result.links
    assert link.is_new is False
    assert link.canonical_id == "apple"  # not "store", despite its perfect vector
    assert store.count() == 2


# --- canonical_id reuse + stability -----------------------------------------


def test_canonical_id_is_stable_across_fresh_stores() -> None:
    """Create-new mints a deterministic id: same corpus/order → same id (R6.4)."""
    embedder = FakeEmbedder(dim=64)
    first = _link_once(InMemoryEntityStore(), embedder)
    second = _link_once(InMemoryEntityStore(), embedder)
    assert first.links[0].canonical_id == second.links[0].canonical_id


def test_second_mention_reuses_the_seeded_id() -> None:
    """Linking the same entity twice reuses (merges into) the seeded canonical."""
    store = InMemoryEntityStore()
    embedder = FakeEmbedder(dim=64)
    stage = EntityLinkingStage(store, embedder)

    first = stage.link("", MENTIONS, [], CLUSTERS)
    second = stage.link("", MENTIONS, [], CLUSTERS)

    assert first.links[0].is_new is True
    assert second.links[0].is_new is False
    assert first.links[0].canonical_id == second.links[0].canonical_id
    assert store.count() == 1


# --- gated paths default OFF -------------------------------------------------


def test_tiebreaker_off_makes_no_llm_call_near_threshold() -> None:
    """Default (gate off): a near-threshold decision does NOT call the LLM.

    The fuzzy candidate scores 0.84 with threshold 0.85 and margin 0.05 — squarely
    in the band the tie-breaker would arbitrate if enabled. With the gate off the
    deterministic decision stands (create-new) and the LLM is never touched. (A
    different normalized name keeps it off the decisive exact-key block path.)
    """
    store = InMemoryEntityStore()
    store.upsert(
        CanonicalEntity(canonical_id="store", name="Apple Store", type="ORG", vector=_unit(0.84))
    )
    llm = FakeLLMClient(completion="yes")

    stage = EntityLinkingStage(
        store,
        _FixedEmbedder([1.0, 0.0, 0.0, 0.0]),
        threshold=0.85,
        tiebreaker_margin=0.05,
        llm_client=llm,  # present but gated off
    )
    result = stage.link("", MENTIONS, [], CLUSTERS)

    assert llm.calls == 0  # gate off → never called
    assert result.links[0].is_new is True  # deterministic decision stands


def test_nil_off_creates_new_for_very_low_confidence() -> None:
    """Default (gate off): a very-low-confidence entity create-news, not NIL."""
    store = InMemoryEntityStore()  # empty → no candidates → score 0.0
    result = _link_once(store, FakeEmbedder(dim=64), nil_enabled=False)
    assert result.links[0].is_new is True
    assert store.count() == 1


def test_tiebreaker_on_is_wired_and_fires() -> None:
    """Sanity: the tie-breaker branch is wired — enabling it consults the LLM.

    Proves the gated path exists (not just absent); the OFF default above is the
    primary guarantee. The LLM says "yes" → the near-threshold (fuzzy) decision
    merges. A different normalized name keeps it off the decisive exact-key path.
    """
    store = InMemoryEntityStore()
    store.upsert(
        CanonicalEntity(canonical_id="store", name="Apple Store", type="ORG", vector=_unit(0.84))
    )
    llm = FakeLLMClient(completion="yes")

    stage = EntityLinkingStage(
        store,
        _FixedEmbedder([1.0, 0.0, 0.0, 0.0]),
        threshold=0.85,
        tiebreaker_margin=0.05,
        tiebreaker_enabled=True,
        llm_client=llm,
    )
    result = stage.link("", MENTIONS, [], CLUSTERS)

    assert llm.calls == 1
    assert result.links[0].is_new is False  # LLM "yes" flipped it to a merge
    assert result.links[0].canonical_id == "store"
