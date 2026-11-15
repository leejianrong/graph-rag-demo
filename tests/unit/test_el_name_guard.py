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


def test_each_name_with_a_token_the_other_lacks_does_not_merge() -> None:
    """GRA-172: a shared token is not enough when both names add a distinctive one."""
    assert not _links_to_seed("Acme Logistics", "Northwind Logistics")
    assert not _links_to_seed("Northwind Logistics", "Acme Logistics")


def test_the_same_name_with_a_fuller_form_still_merges() -> None:
    assert _links_to_seed("Acme", "Acme Logistics Inc.")
    assert _links_to_seed("Acme Logistics Inc.", "Acme")
    assert _links_to_seed("Barack Obama", "President Barack Obama")


def test_people_who_share_only_a_given_name_or_surname_do_not_merge() -> None:
    """Merges found on the real 2Wiki run (docs/results/real-2wiki-medium-50.md)."""
    assert not _links_to_seed("Sir Henry Curwen", "Joan Curwen")
    assert not _links_to_seed("Rajaraja Chola I", "Arinjaya Chola")
    assert not _links_to_seed(
        "Frederick Calvert, 6th Baron Baltimore", "Frederick, Hereditary Prince of Baden-Durlach"
    )


def test_a_short_form_in_one_document_does_not_pull_in_another_person() -> None:
    """The merge found on the real 2Wiki run, with no model: two different Fredericks.

    Each document's coref clustered the bare "Frederick" with its own full name. The first
    document's "Frederick" then matches the second's exactly, and the exact-key block merges
    them without asking the guard or the cosine.
    """
    store = InMemoryEntityStore()
    stage = EntityLinkingStage(store, FakeEmbedder())

    def link(text: str, full: str) -> str:
        forms = [full, "Frederick"]
        mentions = [
            Mention(
                text=f, type="PERSON", char_start=text.index(f), char_end=text.index(f) + len(f)
            )
            for f in forms
        ]
        clusters = [CorefCluster(canonical=full, members=forms)]
        return stage.link(text, mentions, [], clusters).links[0].canonical_id

    calvert = link(
        "Frederick Calvert, 6th Baron Baltimore, was a British peer. Frederick sold the colony.",
        "Frederick Calvert, 6th Baron Baltimore",
    )
    prince = link(
        "Frederick, Hereditary Prince of Baden-Durlach, was a German noble. Frederick died young.",
        "Frederick, Hereditary Prince of Baden-Durlach",
    )

    assert calvert != prince


def _link_cluster(
    stage: EntityLinkingStage, text: str, full: str, short: str, type_: str = "PERSON"
) -> str:
    """Link one document whose coref cluster is ``full`` plus its bare short form ``short``."""
    forms = [full, short]
    mentions = [
        Mention(
            text=f,
            type=type_,  # type: ignore[arg-type]
            char_start=text.index(f),
            char_end=text.index(f) + len(f),
        )
        for f in forms
    ]
    clusters = [CorefCluster(canonical=full, members=forms)]
    return stage.link(text, mentions, [], clusters).links[0].canonical_id


def test_a_short_form_still_finds_the_fuller_name_it_belongs_to() -> None:
    """The alias case must survive: "Saba" in one document, "Saba Industries" in the next."""
    stage = EntityLinkingStage(InMemoryEntityStore(), FakeEmbedder())

    bare = _link_cluster(stage, "Saba opened a depot.", "Saba", "Saba", "ORG")
    fuller = _link_cluster(
        stage, "Saba Industries opened a depot. Saba hired staff.", "Saba Industries", "Saba", "ORG"
    )

    assert fuller == bare


def test_two_people_who_share_a_surname_stay_apart_through_the_short_form() -> None:
    stage = EntityLinkingStage(InMemoryEntityStore(), FakeEmbedder())

    henry = _link_cluster(
        stage,
        "Sir Henry Curwen was a landowner. Curwen died in 1778.",
        "Sir Henry Curwen",
        "Curwen",
    )
    joan = _link_cluster(
        stage, "Joan Curwen was his daughter. Curwen married late.", "Joan Curwen", "Curwen"
    )

    assert henry != joan


def test_acronym_still_merges() -> None:
    assert _links_to_seed("International Business Machines", "IBM")
    assert _links_to_seed("IBM", "International Business Machines")
