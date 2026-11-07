"""Scoped ids (ADR-0015, GRA-20): tenants get their own ids; the default scope keeps the old."""

from __future__ import annotations

import hashlib

import pytest

from graph_rag.fakes import FakeEmbedder, InMemoryEntityStore
from graph_rag.ids import document_id, entity_id
from graph_rag.models import CanonicalEntity, CorefCluster, Mention
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.stages.entity_linking import EntityLinkingStage

A, B = Scope("tenant-a"), Scope("tenant-b")
SURFACE = "Acme Corp"
MENTIONS = [Mention(text=SURFACE, type="ORG", char_start=0, char_end=len(SURFACE))]
CLUSTERS = [CorefCluster(canonical=SURFACE, members=[SURFACE])]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_default_scope_keeps_the_ids_minted_before_scopes_existed() -> None:
    """Pinned literals, hashed here from the old strings, so stored data stays valid."""
    assert document_id("documents", "a.txt") == sha("documents/a.txt")
    assert document_id("documents", "a.txt") == (
        "430622c741eb6e4d7cc78861e136c254be49d1d84877795ff3895319905d1015"
    )
    assert entity_id("ORG", "acme corp") == "e-081503071b2d1115"
    assert (
        entity_id("ORG", "acme corp", salt="1234abcd")
        == "e-" + sha("el:ORG:acme corp|1234abcd")[:16]
    )


def test_the_same_thing_in_two_scopes_gets_two_ids() -> None:
    scopes = [DEFAULT_SCOPE, A, B, Scope("tenant-a", "notes"), Scope("tenant-a", "work")]
    docs = {document_id("documents", "a.txt", s) for s in scopes}
    entities = {entity_id("ORG", "acme corp", s) for s in scopes}
    assert len(docs) == len(entities) == len(scopes)
    assert document_id("documents", "a.txt", A) == document_id(
        "documents", "a.txt", Scope("tenant-a")
    )


def test_a_tenant_named_default_with_a_namespace_is_not_the_default_scope() -> None:
    assert Scope("default").is_default and not Scope("default", "x").is_default
    assert entity_id("ORG", "x", Scope("default", "x")) != entity_id("ORG", "x")


@pytest.mark.parametrize("tenant", ["", "a/b", "a|b"])
def test_a_scope_part_that_would_make_the_key_ambiguous_is_rejected(tenant: str) -> None:
    with pytest.raises(ValueError):
        Scope(tenant)
    with pytest.raises(ValueError):
        Scope("t", tenant)


def link(store: InMemoryEntityStore, scope: Scope | None = None) -> str:
    stage = EntityLinkingStage(store, FakeEmbedder())
    result = stage.link("", MENTIONS, [], CLUSTERS, *([scope] if scope else []))
    return result.links[0].canonical_id


def test_the_linking_stage_mints_scoped_ids_and_stays_idempotent_inside_a_scope() -> None:
    store = InMemoryEntityStore()
    first = link(store, A)
    assert link(store, A) == first and store.count() == 1  # a re-ingest finds its entity

    assert first == entity_id("ORG", "acme corp", A)
    assert link(InMemoryEntityStore(), B) == entity_id("ORG", "acme corp", B) != first
    assert link(InMemoryEntityStore()) == "e-081503071b2d1115"  # default scope, pinned


def test_a_clash_inside_one_scope_still_gets_the_vector_salt() -> None:
    """Two distinct entities with one type and name keep distinct ids, as before."""
    store = InMemoryEntityStore()
    taken = entity_id("ORG", "acme corp", A)
    store.upsert(CanonicalEntity(canonical_id=taken, name="Other", type="ORG", vector=[1.0]))
    minted = link(store, A)
    assert minted != taken and minted.startswith("e-")
