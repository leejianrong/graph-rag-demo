"""The entity-linking name guard (GRA-157, ADR-0004) — fast, offline.

Mention-in-context vectors carry the whole sentence, so two different organisations named
in one sentence score near 1.0 against each other. A similarity merge must therefore also
see related names: a shared distinctive token, or an acronym.
"""

from __future__ import annotations

import math

from graph_rag.fakes import FakeEmbedder, InMemoryEntityStore
from graph_rag.models import CanonicalEntity, CorefCluster, Mention, Sentence
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.stages.entity_linking import EntityLinkingStage

DOC_C = (
    "Acme Logistics acquired Northwind Shipping in 2021. Northwind Shipping operates out of "
    "Hamburg, and its CEO Klaus Weber moved to Acme Logistics after the deal."
)
_SECOND = DOC_C.index(". ") + 2  # start of the second sentence


class _FixedEmbedder:
    """Returns one canned unit vector for every text, so the cosine is known exactly."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dim(self) -> int:
        return len(self._vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def doc_c_inputs() -> tuple[list[Mention], list[Sentence], list[CorefCluster]]:
    """The doc-c NER mentions, sentences and (separate) coref clusters."""

    def at(text: str, nth: int) -> int:
        start = -1
        for _ in range(nth + 1):
            start = DOC_C.index(text, start + 1)
        return start

    spec: list[tuple[str, str, int]] = [
        ("Acme Logistics", "ORG", 0),
        ("Northwind Shipping", "ORG", 0),
        ("Northwind Shipping", "ORG", 1),
        ("Hamburg", "LOCATION", 0),
        ("Klaus Weber", "PERSON", 0),
        ("Acme Logistics", "ORG", 1),
    ]
    mentions = [
        Mention(
            text=t,
            type=ty,  # type: ignore[arg-type]
            char_start=at(t, n),
            char_end=at(t, n) + len(t),
        )
        for t, ty, n in spec
    ]
    sentences = [
        Sentence(text=DOC_C[: _SECOND - 1], char_start=0, char_end=_SECOND - 1, index=0),
        Sentence(text=DOC_C[_SECOND:], char_start=_SECOND, char_end=len(DOC_C), index=1),
    ]
    clusters = [
        CorefCluster(canonical="Acme Logistics", members=["Acme Logistics"]),
        CorefCluster(canonical="Northwind Shipping", members=["Northwind Shipping"]),
    ]
    return mentions, sentences, clusters


def test_two_orgs_in_one_sentence_stay_separate() -> None:
    """Acme and Northwind share a sentence (FakeEmbedder cosine ~0.95) yet are two entities."""
    mentions, sentences, clusters = doc_c_inputs()
    result = EntityLinkingStage(InMemoryEntityStore(), FakeEmbedder()).link(
        DOC_C, mentions, sentences, clusters
    )
    ids = {link.mention_text: link.canonical_id for link in result.links}
    assert ids["Acme Logistics"] != ids["Northwind Shipping"]
    assert all(link.is_new for link in result.links)


def _links_to_seed(seed_name: str, surface: str) -> bool:
    """Seed ``seed_name``, link ``surface`` at cosine 0.99 (above threshold); True if merged."""
    store = InMemoryEntityStore()
    store.upsert(
        DEFAULT_SCOPE,
        CanonicalEntity(
            canonical_id="seed",
            name=seed_name,
            type="ORG",
            aliases=[],
            vector=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    mentions = [Mention(text=surface, type="ORG", char_start=0, char_end=len(surface))]
    clusters = [CorefCluster(canonical=surface, members=[surface])]
    vector = [0.99, math.sqrt(1.0 - 0.99**2), 0.0, 0.0]
    stage = EntityLinkingStage(store, _FixedEmbedder(vector))
    return stage.link("", mentions, [], clusters).links[0].canonical_id == "seed"


def test_high_cosine_without_shared_token_does_not_merge() -> None:
    assert not _links_to_seed("Acme Logistics", "Northwind Shipping")


def test_shared_legal_suffix_alone_does_not_merge() -> None:
    assert not _links_to_seed("Acme Inc", "Globex Inc")


def test_shared_distinctive_token_still_merges() -> None:
    assert _links_to_seed("Acme", "Acme Logistics")
    assert _links_to_seed("Acme Logistics", "Acme")


def test_acronym_still_merges() -> None:
    assert _links_to_seed("International Business Machines", "IBM")
    assert _links_to_seed("IBM", "International Business Machines")
