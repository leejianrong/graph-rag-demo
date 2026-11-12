"""POST /v1/entities/{id}/merge: merging, the record, and a crash healed by a retry (GRA-144)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.curation_api import EntityCurator
from graph_rag.entities_api import EntityExplorer
from graph_rag.models import CanonicalEntity, EdgeProvenance, EntityLink, Triple
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_contract import SPEC_PATH, fits
from tests.unit.test_agent_api_v1 import Queue, drain


def _app(components, queue=None, **settings):  # type: ignore[no-untyped-def]
    return create_app(
        components.object_store,
        queue or Queue(),
        Settings(**settings),
        explorer=EntityExplorer(
            components.entity_store,
            components.graph_store,
            components.document_store,
            components.merge_store,
        ),
        curator=EntityCurator(
            components.entity_store,
            components.graph_store,
            components.document_store,
            components.merge_store,
        ),
        keys=ApiKeys(InMemoryKeyStore()) if settings.get("require_auth") else None,
    )


def _edge(subject: str, object_: str, doc: str, sentence: int = 0) -> Triple:
    return Triple(
        subject_id=subject,
        predicate="WORKS_FOR",
        object_id=object_,
        provenance=EdgeProvenance(
            source_doc_id=doc, sentence_index=sentence, source_sentence="Text."
        ),
    )


def _seed(components, scope: Scope = DEFAULT_SCOPE) -> None:  # type: ignore[no-untyped-def]
    """keep (Acme Corp) and dup (Acme Holdings, aka AH), each with an edge to a person."""
    from graph_rag.models import DocumentRecord

    entities = [
        CanonicalEntity(canonical_id="keep", name="Acme Corp", type="ORG", vector=[1.0, 0.0]),
        CanonicalEntity(
            canonical_id="dup", name="Acme Holdings", type="ORG", aliases=["AH"], vector=[0.0, 1.0]
        ),
        CanonicalEntity(canonical_id="bob", name="Bob Stone", type="PERSON"),
        CanonicalEntity(canonical_id="cat", name="Cat Lee", type="PERSON"),
    ]
    for entity in entities:
        components.entity_store.upsert(scope, entity)
    components.graph_store.upsert_entities(scope, entities)
    components.graph_store.write_triples(
        scope, [_edge("bob", "keep", "d1"), _edge("cat", "dup", "d2"), _edge("bob", "dup", "d2", 1)]
    )
    for doc, canonical in (("d1", "keep"), ("d2", "dup")):
        components.document_store.upsert(
            scope,
            DocumentRecord(
                document_id=doc,
                bucket="b",
                object_key=f"{doc}.txt",
                text="t",
                el_result=[
                    EntityLink(
                        mention_text="Acme",
                        canonical_id=canonical,
                        entity_type="ORG",
                        score=1.0,
                        is_new=True,
                    )
                ],
            ),
        )


@pytest.fixture
def stack():  # type: ignore[no-untyped-def]
    components = build_offline_components()
    _seed(components)
    return TestClient(_app(components)), components


def _merge(client: TestClient, keep: str, ids: list[str], **extra: Any):  # type: ignore[no-untyped-def]
    return client.post(f"/v1/entities/{keep}/merge", json={"merge_ids": ids} | extra)


def test_merging_leaves_one_entity_with_both_aliases_and_all_edges(stack) -> None:
    client, components = stack
    response = _merge(client, "keep", ["dup"])
    assert response.status_code == 200
    body = response.json()
    assert body["merged"] == ["dup"] and body["already_merged"] == [] and body["merge_id"]
    assert body["entity"]["aliases"] == ["Acme Holdings", "AH"]

    detail = client.get("/v1/entities/keep").json()
    assert detail["aliases"] == ["Acme Holdings", "AH"]
    assert detail["incoming_total"] == 3 and detail["outgoing_total"] == 0
    assert {n["entity"]["canonical_id"] for n in detail["neighbours"]} == {"bob", "cat"}
    assert [(s["document_id"], s["mentions"]) for s in detail["sources"]] == [("d1", 1), ("d2", 1)]
    assert client.get("/v1/entities/bob").json()["outgoing_total"] == 2  # bob's edges follow
    assert components.graph_store.get_node(DEFAULT_SCOPE, "dup") is None
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is None
    assert components.entity_store.first_seen(DEFAULT_SCOPE, "keep")  # not made new by the merge


def test_the_merge_record_is_returned_by_the_entity_detail(stack) -> None:
    client, _ = stack
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    [origin] = client.get("/v1/entities/keep").json()["merged_from"]
    assert origin["canonical_id"] == "dup" and origin["name"] == "Acme Holdings"
    assert (
        origin["type"] == "ORG" and origin["aliases"] == ["AH"] and origin["merge_id"] == merge_id
    )
    assert origin["merged_at"]
    assert client.get("/v1/entities/bob").json()["merged_from"] == []
    spec = yaml.safe_load(SPEC_PATH.read_text())
    assert fits(spec, "EntityNeighbourhood", client.get("/v1/entities/keep").json()) == []


def test_the_record_holds_all_a_split_needs(stack) -> None:
    _, components = stack
    client = TestClient(_app(components))
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    record = components.merge_store.get(DEFAULT_SCOPE, merge_id)
    assert record is not None and record.status == "done" and record.completed_at
    assert record.keep_id == "keep" and record.keep_aliases_before == []
    [dropped] = record.dropped
    assert (dropped.canonical_id, dropped.name, dropped.type) == ("dup", "Acme Holdings", "ORG")
    assert dropped.aliases == ["AH"] and dropped.vector == [0.0, 1.0] and dropped.first_seen
    assert {(e.subject_id, e.object_id, e.provenance.source_doc_id) for e in dropped.edges} == {
        ("cat", "dup", "d2"),
        ("bob", "dup", "d2"),
    }
    [link] = dropped.document_links
    assert link.document_id == "d2" and link.link.canonical_id == "dup"
    assert link.link.mention_text == "Acme"


def test_the_dropped_id_is_not_found_and_says_where_it_went(stack) -> None:
    client, _ = stack
    _merge(client, "keep", ["dup"])
    gone = client.get("/v1/entities/dup")
    assert gone.status_code == 404
    assert gone.json()["error"]["details"] == {"merged_into": "keep"}
    assert client.get("/v1/entities/nope").json()["error"].get("details") is None


def test_merging_again_is_a_no_op(stack) -> None:
    client, components = stack
    first = _merge(client, "keep", ["dup"]).json()
    again = _merge(client, "keep", ["dup"])
    assert again.status_code == 200
    assert again.json()["merged"] == [] and again.json()["already_merged"] == ["dup"]
    assert again.json()["merge_id"] == first["merge_id"]
    assert len(components.merge_store.all(DEFAULT_SCOPE)) == 1
    assert client.get("/v1/entities/keep").json()["incoming_total"] == 3


def test_a_request_can_mix_new_and_already_merged_ids(stack) -> None:
    client, components = stack
    components.entity_store.upsert(
        DEFAULT_SCOPE, CanonicalEntity(canonical_id="dup2", name="Acme Intl", type="ORG")
    )
    _merge(client, "keep", ["dup"])
    mixed = _merge(client, "keep", ["dup", "dup2"]).json()
    assert mixed["merged"] == ["dup2"] and mixed["already_merged"] == ["dup"]
    assert len(client.get("/v1/entities/keep").json()["merged_from"]) == 2


def test_bad_requests_are_invalid_and_change_nothing(stack) -> None:
    client, components = stack
    for ids in ([], ["keep"], ["dup", "dup"], [""], [f"x{i}" for i in range(11)]):
        response = _merge(client, "keep", ids)
        assert response.status_code == 400, ids
        assert response.json()["error"]["code"] == "invalid_request"
    assert client.post("/v1/entities/keep/merge", json={}).status_code == 400
    assert components.merge_store.all(DEFAULT_SCOPE) == []
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is not None


def test_unknown_ids_are_not_found_before_anything_is_written(stack) -> None:
    client, components = stack
    assert _merge(client, "nope", ["dup"]).status_code == 404
    response = _merge(client, "keep", ["dup", "nope"])
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert components.merge_store.all(DEFAULT_SCOPE) == []
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is not None


def test_entities_of_different_types_conflict(stack) -> None:
    client, components = stack
    response = _merge(client, "keep", ["dup", "bob"])
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "conflict" and error["details"]["entity_ids"] == ["bob"]
    assert components.merge_store.all(DEFAULT_SCOPE) == []
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is not None


def test_a_namespace_and_another_tenant_cannot_reach_the_entities() -> None:
    components = build_offline_components()
    _seed(components, Scope("tenant-a"))
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        Queue(),
        Settings(require_auth=True),
        explorer=EntityExplorer(
            components.entity_store,
            components.graph_store,
            components.document_store,
            components.merge_store,
        ),
        curator=EntityCurator(
            components.entity_store,
            components.graph_store,
            components.document_store,
            components.merge_store,
        ),
        keys=keys,
    )
    client = TestClient(app)
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    auth_a, auth_b = ({"Authorization": f"Bearer {s}"} for s in (a, b))

    other = client.post("/v1/entities/keep/merge", json={"merge_ids": ["dup"]}, headers=auth_b)
    assert other.status_code == 404  # tenant-b has no such entity: same as an unknown id
    in_ns = client.post(
        "/v1/entities/keep/merge",
        json={"merge_ids": ["dup"], "namespace": "notes"},
        headers=auth_a,
    )
    assert in_ns.status_code == 404
    assert components.merge_store.all(Scope("tenant-a")) == []
    assert client.post("/v1/entities/keep/merge", json={"merge_ids": ["dup"]}).status_code == 401

    mine = client.post("/v1/entities/keep/merge", json={"merge_ids": ["dup"]}, headers=auth_a)
    assert mine.status_code == 200
    detail_b = client.get("/v1/entities/keep", headers=auth_b)
    assert detail_b.status_code == 404  # tenant-b never sees tenant-a's merge record
    assert client.get("/v1/entities/dup", headers=auth_b).json()["error"].get("details") is None


def test_without_a_curator_the_route_says_unavailable() -> None:
    components = build_offline_components()
    client = TestClient(create_app(components.object_store, Queue(), Settings()))
    response = client.post("/v1/entities/x/merge", json={"merge_ids": ["y"]})
    assert response.status_code == 503 and response.json()["error"]["code"] == "unavailable"


class _Crash(Exception):
    pass


@pytest.mark.parametrize("step", ["upsert_entities", "merge_nodes", "documents", "delete"])
def test_a_crash_between_steps_is_healed_by_the_retry(stack, monkeypatch, step) -> None:  # type: ignore[no-untyped-def]
    _, components = stack
    client = TestClient(_app(components), raise_server_exceptions=False)
    graph, documents, entities = (
        components.graph_store,
        components.document_store,
        components.entity_store,
    )
    target, name = {
        "upsert_entities": (graph, "upsert_entities"),
        "merge_nodes": (graph, "merge_nodes"),
        "documents": (documents, "upsert"),
        "delete": (entities, "delete"),
    }[step]
    real = getattr(target, name)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise _Crash

    monkeypatch.setattr(target, name, boom)
    assert _merge(client, "keep", ["dup"]).status_code == 500
    [pending] = components.merge_store.all(DEFAULT_SCOPE)
    assert pending.status == "pending" and pending.dropped[0].edges  # written before any change
    assert client.get("/v1/entities/keep").json()["merged_from"] == []  # not yet a finished merge

    monkeypatch.setattr(target, name, real)
    retry = _merge(client, "keep", ["dup"])
    assert retry.status_code == 200
    assert retry.json()["already_merged"] == ["dup"]  # the retry finished the pending record
    [done] = components.merge_store.all(DEFAULT_SCOPE)
    assert done.merge_id == pending.merge_id and done.status == "done"

    detail = client.get("/v1/entities/keep").json()
    assert detail["aliases"] == ["Acme Holdings", "AH"] and detail["incoming_total"] == 3
    assert [s["document_id"] for s in detail["sources"]] == ["d1", "d2"]
    assert [m["canonical_id"] for m in detail["merged_from"]] == ["dup"]
    assert (
        entities.get(DEFAULT_SCOPE, "dup") is None and graph.get_node(DEFAULT_SCOPE, "dup") is None
    )
    assert client.get("/v1/entities/dup").json()["error"]["details"] == {"merged_into": "keep"}


def test_a_later_ingest_naming_the_dropped_entity_links_to_the_survivor() -> None:
    components = build_offline_components()
    queue = Queue()
    client = TestClient(_app(components, queue))
    for n, text in enumerate(["Acme Corp hired Bob Stone.", "Acme Holdings hired Carol Diaz."]):
        client.post("/v1/remember", json={"text": text, "source_id": f"n{n}"})
    drain(components, queue)
    ids = {e.name: e.canonical_id for e in components.entity_store.all(DEFAULT_SCOPE)}
    keep, dup = ids["Acme Corp"], ids["Acme Holdings"]
    assert _merge(client, keep, [dup]).status_code == 200

    client.post("/v1/remember", json={"text": "Acme Holdings hired Dave Kim.", "source_id": "n2"})
    drain(components, queue)

    names = [e.name for e in components.entity_store.all(DEFAULT_SCOPE)]
    assert names.count("Acme Holdings") == 0 and names.count("Acme Corp") == 1
    detail = client.get(f"/v1/entities/{keep}").json()
    assert len(detail["sources"]) == 3
    assert "Dave Kim" in {n["entity"]["name"] for n in detail["neighbours"]}
    assert client.get(f"/v1/entities/{dup}").status_code == 404


def test_the_fakes_meet_the_merge_checks_the_real_stores_must_meet() -> None:
    from graph_rag.fakes import InMemoryDocumentStore, InMemoryEntityStore, InMemoryGraphStore
    from graph_rag.merge_records import InMemoryMergeRecordStore
    from tests.merge_checks import check_entity_merge, check_merge_store

    check_merge_store(InMemoryMergeRecordStore())
    check_entity_merge(
        InMemoryEntityStore(),
        InMemoryGraphStore(),
        InMemoryDocumentStore(),
        InMemoryMergeRecordStore(),
        dims=4,
    )
