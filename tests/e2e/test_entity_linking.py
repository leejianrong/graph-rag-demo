"""Fast end-to-end suite for the V4 entity-linking path (TESTING §2, primary gate).

Drives :meth:`~graph_rag.orchestrator.Orchestrator.process_document` through the
port seam against in-memory fakes — no Docker, no sentence-transformer model, no
LLM, deterministic, ``$0``. The real
:class:`~graph_rag.stages.entity_linking.EntityLinkingStage` runs over
:class:`~graph_rag.fakes.InMemoryEntityStore` + :class:`~graph_rag.fakes.FakeEmbedder`,
so the actual block/score/merge logic + the EL checkpoint are exercised (only the
ports are faked, ADR-0010).

Fixtures **pin ingestion order** because EL is order-sensitive (ADR-0004): the
first document to mention an entity seeds its canonical record. The
``FakeEmbedder`` feature-hashes normalized tokens into a bag-of-tokens vector, so
two surface forms with the same token bag (e.g. ``"Barack Obama"`` /
``"Obama Barack"``) embed identically and score cosine 1.0 — a deterministic merge
without a real model.

Covers the V4 demo (merge across docs → one canonical; create-new for a distinct
entity; enriched ES-Documents record written at the checkpoint; order-sensitivity)
and re-asserts the V1–V3 guarantees under V4 (deterministic id + idempotent
overwrite; log-and-drop, including an EL failure).
"""

from __future__ import annotations

import pytest

from graph_rag.fakes import (
    FakeEmbedder,
    FakeNerStage,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryObjectStore,
)
from graph_rag.ids import document_id
from graph_rag.models import CorefCluster, IngestTrigger, Mention, Sentence
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.coref import FakeCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage

BUCKET = "documents"

# --- Fixtures pinning ingestion order (order-sensitive EL, ADR-0004) ---------
# Two docs naming the SAME person with token-bag-identical context: "Barack Obama"
# and its reorder "Obama Barack". Same bag -> identical FakeEmbedder vector ->
# cosine 1.0 >= el_threshold -> merge. Their normalized names differ, so the merge
# is driven by kNN (not blocking) and grows the canonical's aliases.
DOC1_KEY = "obama1.md"
DOC1_TEXT = "Barack Obama spoke today."
DOC1_SURFACE = "Barack Obama"

DOC2_KEY = "obama2.md"
DOC2_TEXT = "Obama Barack spoke today."
DOC2_SURFACE = "Obama Barack"

# A genuinely different entity: disjoint tokens -> cosine ~0 -> create-new.
DOC3_KEY = "globex.md"
DOC3_TEXT = "Globex Industries builds widgets."
DOC3_SURFACE = "Globex Industries"


def _person_stages(surface: str, text: str) -> tuple[FakeNerStage, FakeCorefStage]:
    """Canned NER + coref stages for a one-PERSON document (content-ignoring fakes)."""
    ner = FakeNerStage(
        mentions=[Mention(text=surface, type="PERSON", char_start=0, char_end=len(surface))],
        sentences=[Sentence(text=text, char_start=0, char_end=len(text), index=0)],
    )
    coref = FakeCorefStage(clusters=[CorefCluster(canonical=surface, members=[surface])])
    return ner, coref


def _org_stages(surface: str, text: str) -> tuple[FakeNerStage, FakeCorefStage]:
    """Canned NER + coref stages for a one-ORG document."""
    ner = FakeNerStage(
        mentions=[Mention(text=surface, type="ORG", char_start=0, char_end=len(surface))],
        sentences=[Sentence(text=text, char_start=0, char_end=len(text), index=0)],
    )
    coref = FakeCorefStage(clusters=[CorefCluster(canonical=surface, members=[surface])])
    return ner, coref


def _run(
    *,
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    el_stage: EntityLinkingStage,
    key: str,
    text: str,
    ner: FakeNerStage,
    coref: FakeCorefStage,
):
    """Ingest one doc through a fresh orchestrator sharing the stores + EL stage."""
    object_store.put(BUCKET, key, text.encode())
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner,
        coref_stage=coref,
        entity_linking_stage=el_stage,
    )
    return orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=key))


def test_merge_across_docs_yields_one_canonical(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """Two docs naming the same entity differently → ONE canonical (merge, R3.1/3.2).

    Order-sensitivity: doc1 seeds the canonical (name + vector); doc2 merges into
    the SAME ``canonical_id`` and contributes its surface as an alias.
    """
    el_stage = EntityLinkingStage(entity_store, embedder)

    ner1, coref1 = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    first = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner1,
        coref=coref1,
    )
    ner2, coref2 = _person_stages(DOC2_SURFACE, DOC2_TEXT)
    second = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC2_KEY,
        text=DOC2_TEXT,
        ner=ner2,
        coref=coref2,
    )

    assert first is not None and second is not None
    # Exactly ONE canonical entity across the two documents.
    assert entity_store.count() == 1

    (link1,) = first.el_result
    (link2,) = second.el_result
    # doc1 seeded it (create-new); doc2 merged into the same id (reuse).
    assert link1.is_new is True
    assert link2.is_new is False
    assert link1.canonical_id == link2.canonical_id
    assert link2.score >= 0.82  # cosine 1.0 for the identical token bag

    canonical = entity_store.get(link1.canonical_id)
    assert canonical is not None
    # First mention seeds the canonical name (order-sensitivity, ADR-0004).
    assert canonical.name == DOC1_SURFACE
    # The differently-normalized second surface accumulates as an alias.
    assert canonical.aliases == [DOC2_SURFACE]


def test_create_new_for_distinct_entity(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """A genuinely different entity mints a NEW canonical record (create-new, R3.3)."""
    el_stage = EntityLinkingStage(entity_store, embedder)

    ner1, coref1 = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    first = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner1,
        coref=coref1,
    )
    ner3, coref3 = _org_stages(DOC3_SURFACE, DOC3_TEXT)
    third = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC3_KEY,
        text=DOC3_TEXT,
        ner=ner3,
        coref=coref3,
    )

    assert first is not None and third is not None
    # Two distinct canonicals — the org did not merge into the person.
    assert entity_store.count() == 2
    (person_link,) = first.el_result
    (org_link,) = third.el_result
    assert org_link.is_new is True
    assert org_link.canonical_id != person_link.canonical_id
    assert org_link.score < 0.82  # disjoint tokens → far below threshold


def test_enriched_record_written_at_checkpoint(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """The EL checkpoint persists the enriched ES-Documents record (ADR-0001/0005).

    The stored record for the document — keyed by the same ``document_id`` — now
    carries mentions + coref clusters + the per-doc EL result + sentence vectors,
    not just raw text.
    """
    el_stage = EntityLinkingStage(entity_store, embedder)
    ner1, coref1 = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    result = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner1,
        coref=coref1,
    )

    assert result is not None
    doc_id = document_id(BUCKET, DOC1_KEY)
    stored = document_store.get(doc_id)
    assert stored is not None
    assert stored.document_id == doc_id
    # Raw text preserved, and every enrichment field now populated.
    assert stored.text == DOC1_TEXT
    assert [m.text for m in stored.mentions] == [DOC1_SURFACE]
    assert [c.canonical for c in stored.coref_clusters] == [DOC1_SURFACE]
    assert [link.mention_text for link in stored.el_result] == [DOC1_SURFACE]
    assert stored.sentence_vectors is not None
    assert len(stored.sentence_vectors) == 1
    assert len(stored.sentence_vectors[0]) == embedder.dim


def test_reingest_overwrites_no_duplicate(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """Re-ingesting the same doc is idempotent: one record, one canonical (R1.5).

    The checkpoint's second write overwrites in place (same ``document_id``), and
    the re-ingested entity merges into its own seeded canonical rather than
    duplicating it.
    """
    el_stage = EntityLinkingStage(entity_store, embedder)
    ner_a, coref_a = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    first = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner_a,
        coref=coref_a,
    )
    ner_b, coref_b = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    second = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner_b,
        coref=coref_b,
    )

    assert first is not None and second is not None
    assert first.record.document_id == second.record.document_id
    assert len(document_store._records) == 1  # noqa: SLF001 — asserting no duplication.
    assert entity_store.count() == 1  # merged into itself, not duplicated.
    # The re-ingest resolved to the same seeded canonical, now a merge.
    assert second.el_result[0].is_new is False
    assert second.el_result[0].canonical_id == first.el_result[0].canonical_id


def test_order_sensitivity_first_doc_seeds_canonical(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """Swapping ingestion order swaps which surface seeds the canonical (ADR-0004).

    The mirror of ``test_merge_across_docs`` with doc2 first: now ``"Obama Barack"``
    seeds the name and ``"Barack Obama"`` becomes the alias — proving the outcome
    is order-sensitive, not incidental.
    """
    el_stage = EntityLinkingStage(entity_store, embedder)
    ner2, coref2 = _person_stages(DOC2_SURFACE, DOC2_TEXT)
    _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC2_KEY,
        text=DOC2_TEXT,
        ner=ner2,
        coref=coref2,
    )
    ner1, coref1 = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    second = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner1,
        coref=coref1,
    )

    assert second is not None
    assert entity_store.count() == 1
    canonical = entity_store.get(second.el_result[0].canonical_id)
    assert canonical is not None
    assert canonical.name == DOC2_SURFACE  # ingested first → seeds the name
    assert canonical.aliases == [DOC1_SURFACE]


def test_el_failure_is_logged_and_dropped(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
) -> None:
    """A failing EL stage is dropped (returns None), not raised (ADR-0001)."""

    class ExplodingELStage:
        """EL stage whose link always fails, to exercise log-and-drop."""

        def link(self, text, mentions, sentences, coref_clusters):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated entity-linking failure")

    ner, coref = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    result = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=ExplodingELStage(),  # type: ignore[arg-type]
        key=DOC1_KEY,
        text=DOC1_TEXT,
        ner=ner,
        coref=coref,
    )
    assert result is None


def test_el_failure_does_not_wedge_the_loop(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
) -> None:
    """After an EL-dropped document, the next good trigger still processes (ADR-0001)."""

    class SometimesExplodingELStage:
        """Fails on the first link call, delegates to a real stage afterwards."""

        def __init__(self, delegate: EntityLinkingStage) -> None:
            self._delegate = delegate
            self._calls = 0

        def link(self, text, mentions, sentences, coref_clusters):  # type: ignore[no-untyped-def]
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("simulated entity-linking failure")
            return self._delegate.link(text, mentions, sentences, coref_clusters)

    el_stage = SometimesExplodingELStage(EntityLinkingStage(entity_store, embedder))

    ner_bad, coref_bad = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    bad = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,  # type: ignore[arg-type]
        key="bad.md",
        text=DOC1_TEXT,
        ner=ner_bad,
        coref=coref_bad,
    )
    assert bad is None

    ner_good, coref_good = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    good = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,  # type: ignore[arg-type]
        key="good.md",
        text=DOC1_TEXT,
        ner=ner_good,
        coref=coref_good,
    )
    assert good is not None
    assert good.el_result[0].is_new is True


@pytest.mark.parametrize("key", [DOC1_KEY])
def test_deterministic_id_under_v4(
    object_store: InMemoryObjectStore,
    document_store: InMemoryDocumentStore,
    entity_store: InMemoryEntityStore,
    embedder: FakeEmbedder,
    key: str,
) -> None:
    """V1 guarantee still holds under V4: the record uses the deterministic id."""
    el_stage = EntityLinkingStage(entity_store, embedder)
    ner, coref = _person_stages(DOC1_SURFACE, DOC1_TEXT)
    result = _run(
        object_store=object_store,
        document_store=document_store,
        el_stage=el_stage,
        key=key,
        text=DOC1_TEXT,
        ner=ner,
        coref=coref,
    )
    assert result is not None
    assert result.record.document_id == document_id(BUCKET, key)
