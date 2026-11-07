"""A scope sees only its own records, on every in-memory store and through the whole pipeline."""

from __future__ import annotations

from graph_rag.fakes import (
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.models import IngestTrigger, QueryRequest
from graph_rag.scope import Scope
from graph_rag.wiring import build_offline_components
from tests.isolation import (
    check_document_store,
    check_entity_store,
    check_graph_store,
    check_object_store,
)

TEXT = b"Ada Lovelace works for Acme Corp. Acme Corp is based in London."


def test_the_object_store_fake_isolates_scopes() -> None:
    check_object_store(InMemoryObjectStore())


def test_the_document_store_fake_isolates_scopes() -> None:
    check_document_store(InMemoryDocumentStore(), dims=3)


def test_the_entity_store_fake_isolates_scopes() -> None:
    check_entity_store(InMemoryEntityStore(), dims=3)


def test_the_graph_store_fake_isolates_scopes() -> None:
    check_graph_store(InMemoryGraphStore())


def ingest(components, scope: Scope, key: str = "note.txt") -> None:
    bucket = components.bucket
    components.object_store.put(scope, bucket, key, TEXT)
    trigger = IngestTrigger(
        bucket=bucket, object_key=key, tenant=scope.tenant, namespace=scope.namespace
    )
    assert components.orchestrator.process_document(trigger)


def test_the_same_text_in_two_tenants_makes_two_entities_and_never_crosses() -> None:
    """The GRA-20 promise end to end: same name in two tenants, two ids, and no leakage."""
    c = build_offline_components()
    a, b = Scope("tenant-a"), Scope("tenant-b")
    ingest(c, a)
    ingest(c, b)

    ids_a = {e.canonical_id for e in c.entity_store.all(a)}
    ids_b = {e.canonical_id for e in c.entity_store.all(b)}
    assert ids_a and ids_b and not ids_a & ids_b  # same names, disjoint ids

    ask = QueryRequest(question="Where is Acme Corp based?")
    in_a = c.retriever.retrieve(ask, a)
    in_b = c.retriever.retrieve(ask, b)
    assert in_a.supporting_sentences and {n.canonical_id for n in in_a.subgraph.nodes} <= ids_a
    assert {n.canonical_id for n in in_b.subgraph.nodes} <= ids_b
    assert c.retriever.retrieve(ask, Scope("tenant-c")).supporting_sentences == []


def test_forgetting_in_one_tenant_leaves_the_other_alone() -> None:
    c = build_offline_components()
    a, b = Scope("tenant-a"), Scope("tenant-b")
    ingest(c, a)
    ingest(c, b)
    before = c.entity_store.count(b), c.graph_store.edge_count(b)

    from graph_rag.ids import document_id

    gone = c.orchestrator.delete_document(document_id(c.bucket, "note.txt", a), a)

    assert gone is not None and gone.entities_removed
    assert c.entity_store.count(a) == 0 and c.graph_store.edge_count(a) == 0
    assert (c.entity_store.count(b), c.graph_store.edge_count(b)) == before
    assert c.retriever.retrieve(QueryRequest(question="Acme Corp"), b).supporting_sentences


def test_a_trigger_names_its_scope_and_an_old_one_is_the_default() -> None:
    from graph_rag.scope import DEFAULT_SCOPE

    assert IngestTrigger.from_json('{"bucket": "b", "object_key": "k"}').scope == DEFAULT_SCOPE
    trigger = IngestTrigger(bucket="b", object_key="k", tenant="t", namespace="n")
    assert IngestTrigger.from_json(trigger.to_json()).scope == Scope("t", "n")


def test_object_keys_never_collide_across_scopes() -> None:
    import pytest

    from graph_rag.adapters.scoping import object_key

    cases = [
        (Scope(), "a/b.txt"),
        (Scope("a"), "b.txt"),
        (Scope("a"), "ns/n/b.txt"),
        (Scope("a", "n"), "b.txt"),
        (Scope("a", "root"), "b.txt"),
        (Scope("root"), "b.txt"),
        (Scope("a", "n"), "x/b.txt"),
    ]
    assert len({object_key(s, k) for s, k in cases}) == len(cases)
    assert object_key(Scope(), "a/b.txt") == "a/b.txt"  # the default scope keeps its keys
    with pytest.raises(ValueError):
        object_key(Scope(), "_scopes/a/root/b.txt")
