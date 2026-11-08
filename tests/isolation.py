"""One scope must never see another's records (ADR-0015, GRA-21).

Each ``check_*`` takes a store and asserts the same two-scope behaviour, so the fast suite runs
it on the in-memory fake and the contract suite runs it on the real adapter. Scopes get a fresh
random name each call because a contract store can be shared across tests, so a check never
assumes a scope it did not write to is empty (the default scope may hold other tests' records).
"""

from __future__ import annotations

import uuid

import pytest

from graph_rag.ids import document_id, entity_id
from graph_rag.models import (
    CanonicalEntity,
    DocumentRecord,
    EdgeProvenance,
    EntityLink,
    Sentence,
    Triple,
)
from graph_rag.ports import DocumentStore, EntityStore, GraphStore, ObjectStore
from graph_rag.scope import DEFAULT_SCOPE, Scope


def scopes() -> tuple[Scope, Scope, Scope]:
    """Two tenants and a namespace of the first, all new."""
    tag = uuid.uuid4().hex[:8]
    return Scope(f"a{tag}"), Scope(f"b{tag}"), Scope(f"a{tag}", "notes")


def check_object_store(store: ObjectStore) -> None:
    a, b, a_notes = scopes()
    bucket = f"documents-{uuid.uuid4().hex[:8]}"
    store.put(a, bucket, "k.txt", b"from a")

    for other in (b, a_notes, DEFAULT_SCOPE):
        with pytest.raises(FileNotFoundError):
            store.get(other, bucket, "k.txt")

    store.put(b, bucket, "k.txt", b"from b")
    store.delete(b, bucket, "k.txt")
    assert store.get(a, bucket, "k.txt") == b"from a"  # b's delete left a's object
    with pytest.raises(FileNotFoundError):
        store.get(b, bucket, "k.txt")


def check_document_store(store: DocumentStore, dims: int) -> None:
    a, b, a_notes = scopes()
    key = f"{uuid.uuid4().hex[:8]}.md"
    vector = [1.0] + [0.0] * (dims - 1)
    record = DocumentRecord(
        document_id=document_id("documents", key, a),
        bucket="documents",
        object_key=key,
        text="Acme Corp is in London.",
        sentences=[Sentence(text="Acme Corp is in London.", char_start=0, char_end=23, index=0)],
        sentence_vectors=[vector],
        el_result=[
            EntityLink(
                mention_text="Acme Corp",
                canonical_id="e-shared",
                entity_type="ORG",
                score=1.0,
                is_new=True,
            )
        ],
    )
    store.upsert(a, record)

    assert store.get(a, record.document_id) is not None
    assert len(store.search_sentences(a, vector=vector, top_k=5)) == 1
    assert store.entities_in_use(a, {"e-shared"}) == {"e-shared"}
    for other in (b, a_notes, DEFAULT_SCOPE):
        assert store.get(other, record.document_id) is None
        hits = store.search_sentences(other, vector=vector, top_k=50)
        assert record.document_id not in {h.document_id for h in hits}
        assert store.entities_in_use(other, {"e-shared"}) == set()
        assert store.delete(other, record.document_id) is False
    assert store.get(a, record.document_id) is not None  # nobody else's delete touched it
    assert store.delete(a, record.document_id) is True


def check_entity_store(store: EntityStore, dims: int) -> None:
    a, b, a_notes = scopes()
    vector = [1.0] + [0.0] * (dims - 1)

    def entity(scope: Scope) -> CanonicalEntity:
        return CanonicalEntity(
            canonical_id=entity_id("ORG", "acme corp", scope),
            name="Acme Corp",
            type="ORG",
            aliases=[],
            vector=vector,
        )

    in_a = entity(a)
    store.upsert(a, in_a)
    ask = {"entity_type": "ORG", "normalized_name": "acme corp"}

    assert [e.canonical_id for e in store.block_candidates(a, **ask)] == [in_a.canonical_id]  # type: ignore[arg-type]
    assert len(store.knn(a, vector=vector, top_k=5)) == 1
    assert (store.count(a), len(store.all(a))) == (1, 1)
    for other in (b, a_notes, DEFAULT_SCOPE):
        assert store.block_candidates(other, **ask) == []  # type: ignore[arg-type]
        assert store.knn(other, vector=vector, top_k=5) == []
        assert store.get(other, in_a.canonical_id) is None
        assert (store.count(other), store.all(other)) == (0, [])
        store.delete(other, in_a.canonical_id)
    assert store.get(a, in_a.canonical_id) is not None  # nobody else's delete touched it

    in_b = entity(b)  # the same name in another tenant is a different entity
    store.upsert(b, in_b)
    assert in_b.canonical_id != in_a.canonical_id
    assert [e.canonical_id for e in store.block_candidates(b, **ask)] == [in_b.canonical_id]  # type: ignore[arg-type]
    assert (store.count(a), store.count(b)) == (1, 1)


def check_graph_store(store: GraphStore) -> None:
    a, b, a_notes = scopes()

    def node(cid: str) -> CanonicalEntity:
        return CanonicalEntity(canonical_id=cid, name=cid, type="ORG", aliases=[], vector=None)

    def edge(subject: str, obj: str, doc: str) -> Triple:
        return Triple(
            subject_id=subject,
            predicate="PART_OF",
            object_id=obj,
            provenance=EdgeProvenance(
                source_doc_id=doc, sentence_index=0, source_sentence="x is part of y."
            ),
        )

    tag = uuid.uuid4().hex[:6]
    a1, a2, b1, b2 = (f"{n}-{tag}" for n in ("a1", "a2", "b1", "b2"))
    for scope, (one, two), doc in ((a, (a1, a2), "doc-a"), (b, (b1, b2), "doc-b")):
        store.upsert_entities(scope, [node(one), node(two)])
        store.write_triples(scope, [edge(one, two, doc)])
    assert (store.node_count(a), store.edge_count(a)) == (2, 1)
    assert (store.node_count(b), store.edge_count(b)) == (2, 1)

    assert store.get_node(a, a1) is not None
    assert len(store.khop(a, [a1], hops=1).nodes) == 2
    for other in (b, a_notes, DEFAULT_SCOPE):
        assert store.get_node(other, a1) is None
        assert store.get_node_edges(other, a1) == []
        assert store.khop(other, [a1], hops=2).nodes == []
        store.delete_document_edges(other, "doc-a")
        store.delete_nodes(other, [a1, a2])
    assert (store.node_count(a), store.edge_count(a)) == (2, 1)  # untouched by the others

    store.delete_document_edges(a, "doc-a")
    assert (store.edge_count(a), store.edge_count(b)) == (0, 1)


def check_tenants_cannot_reach_each_other(client, keys, drain) -> None:  # type: ignore[no-untyped-def]
    """Over the HTTP API with auth on: tenant B tries to read, query and delete tenant A's memory.

    ``client`` is a test client for an app with ``require_auth``; ``keys`` an ``ApiKeys`` it
    verifies against; ``drain`` runs the queued ingest triggers through the pipeline. Every
    attempt must return nothing or 404, and A must still see its own data afterwards.
    """
    question = "Where is Acme Corp based?"
    a, _ = keys.create(f"a{uuid.uuid4().hex[:8]}")
    b, _ = keys.create(f"b{uuid.uuid4().hex[:8]}")

    def auth(secret: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {secret}"}

    def recall(secret: str) -> list:
        found = client.post("/v1/recall", json={"question": question}, headers=auth(secret))
        assert found.status_code == 200
        return found.json()["supporting_sentences"]

    text = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."
    stored = client.post("/v1/remember", json={"text": text, "source_id": "n"}, headers=auth(a))
    assert stored.status_code == 202
    drain()
    doc_id = stored.json()["document_id"]

    assert recall(a)
    assert recall(b) == []
    legacy = client.post("/query", json={"question": question}, headers=auth(b)).json()
    assert legacy["supporting_sentences"] == [] and legacy["ranked_nodes"] == []

    assert client.post("/v1/forget", json={"source_id": "n"}, headers=auth(b)).status_code == 404
    forget_by_id = client.post("/v1/forget", json={"document_id": doc_id}, headers=auth(b))
    assert forget_by_id.status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=auth(b)).status_code == 404
    assert recall(a)  # none of B's attempts removed anything

    # B stores the same words under the same source id: a separate document, A's untouched.
    again = client.post("/v1/remember", json={"text": text, "source_id": "n"}, headers=auth(b))
    assert again.json()["document_id"] != doc_id and again.json()["status"] == "accepted"
    drain()
    assert recall(b) and recall(a)
    assert client.post("/v1/forget", json={"source_id": "n"}, headers=auth(b)).status_code == 200
    assert recall(b) == [] and recall(a)  # B's forget left A's memory

    assert client.delete(f"/documents/{doc_id}", headers=auth(a)).status_code == 200
    assert recall(a) == []
