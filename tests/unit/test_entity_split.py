"""POST /v1/entities/{id}/split: undoing a recorded merge; a crash healed by a retry (GRA-145)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.curation_api import EntityCurator
from graph_rag.entities_api import EntityExplorer
from graph_rag.models import CanonicalEntity
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.wiring import build_offline_components
from tests.merge_checks import _snapshot, check_entity_split
from tests.unit.test_agent_api_v1 import Queue, drain
from tests.unit.test_entity_merge import _app, _merge, _seed


@pytest.fixture
def stack():  # type: ignore[no-untyped-def]
    components = build_offline_components()
    _seed(components)
    return TestClient(_app(components)), components


def _split(client: TestClient, keep: str, **body: Any):  # type: ignore[no-untyped-def]
    return client.post(f"/v1/entities/{keep}/split", json=body)


def _shot(components):  # type: ignore[no-untyped-def]
    return _snapshot(
        DEFAULT_SCOPE,
        ["keep", "dup", "bob", "cat"],
        ["d1", "d2"],
        components.entity_store,
        components.graph_store,
        components.document_store,
    )


def test_merge_then_split_gives_back_the_original_entities_and_edges(stack) -> None:
    client, components = stack
    before = _shot(components)
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    assert _shot(components) != before

    response = _split(client, "keep", merge_id=merge_id)
    assert response.status_code == 200
    body = response.json()
    assert body["merge_id"] == merge_id and body["already_split"] == []
    [restored] = body["split"]
    assert restored["canonical_id"] == "dup" and restored["aliases"] == ["AH"]
    assert restored["first_seen"] == before["entities"]["dup"][3]
    assert body["entity"]["aliases"] == []
    assert _shot(components) == before


def test_after_a_split_the_detail_and_the_ids_resolve_again(stack) -> None:
    client, components = stack
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    assert client.get("/v1/entities/dup").status_code == 404
    _split(client, "keep", merge_id=merge_id)

    keep = client.get("/v1/entities/keep").json()
    assert keep["merged_from"] == [] and keep["aliases"] == []
    assert keep["incoming_total"] == 1  # bob only; cat and bob's other edge went back to dup
    dup = client.get("/v1/entities/dup")
    assert dup.status_code == 200 and dup.json()["aliases"] == ["AH"]
    assert dup.json()["incoming_total"] == 2
    assert [s["document_id"] for s in dup.json()["sources"]] == ["d2"]
    record = components.merge_store.get(DEFAULT_SCOPE, merge_id)
    assert record is not None and record.status == "split" and record.split_ids == ["dup"]


def test_splitting_again_is_a_no_op(stack) -> None:
    client, _ = stack
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    _split(client, "keep", merge_id=merge_id)
    again = _split(client, "keep", merge_id=merge_id, entity_ids=["dup"])
    assert again.status_code == 200
    assert again.json()["split"] == [] and again.json()["already_split"] == ["dup"]


def test_merge_id_can_be_left_out_when_there_is_one_merge(stack) -> None:
    client, _ = stack
    _merge(client, "keep", ["dup"])
    assert _split(client, "keep").status_code == 200


def test_an_entity_with_no_merge_record_is_a_conflict(stack) -> None:
    client, components = stack
    response = _split(client, "keep")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "conflict" and error["details"]["reason"] == "no_merge_record"
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    wrong = _split(client, "keep", merge_id=merge_id, entity_ids=["bob"])
    assert wrong.status_code == 409 and wrong.json()["error"]["details"]["entity_ids"] == ["bob"]
    other = _split(client, "bob", merge_id=merge_id)  # a record of another survivor
    assert other.status_code == 409
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is None  # nothing was touched


def test_unknown_ids_are_not_found_and_bad_lists_are_invalid(stack) -> None:
    client, _ = stack
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    assert _split(client, "nope", merge_id=merge_id).status_code == 404
    assert _split(client, "keep", merge_id="nope").status_code == 404
    for ids in ([], ["dup", "dup"], [""], [f"x{i}" for i in range(11)]):
        response = _split(client, "keep", merge_id=merge_id, entity_ids=ids)
        assert response.status_code == 400, ids
        assert response.json()["error"]["code"] == "invalid_request"
    assert _split(client, "keep", merge_id="").status_code == 400


def test_several_merges_need_the_id_of_one(stack) -> None:
    client, components = stack
    components.entity_store.upsert(
        DEFAULT_SCOPE, CanonicalEntity(canonical_id="dup2", name="Acme Intl", type="ORG")
    )
    first = _merge(client, "keep", ["dup"]).json()["merge_id"]
    second = _merge(client, "keep", ["dup2"]).json()["merge_id"]
    response = _split(client, "keep")
    assert response.status_code == 400
    assert set(response.json()["error"]["details"]["merge_ids"]) == {first, second}
    assert _split(client, "keep", merge_id=second).status_code == 200


def test_a_pending_merge_cannot_be_split(stack, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, components = stack
    quiet = TestClient(_app(components), raise_server_exceptions=False)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError

    real = components.graph_store.merge_nodes
    monkeypatch.setattr(components.graph_store, "merge_nodes", boom)
    assert _merge(quiet, "keep", ["dup"]).status_code == 500
    [pending] = components.merge_store.all(DEFAULT_SCOPE)
    response = _split(client, "keep", merge_id=pending.merge_id)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["reason"] == "merge_pending"
    monkeypatch.setattr(components.graph_store, "merge_nodes", real)


def test_a_chain_is_split_in_dependency_order(stack) -> None:
    client, components = stack
    components.entity_store.upsert(
        DEFAULT_SCOPE, CanonicalEntity(canonical_id="top", name="Acme Group", type="ORG")
    )
    components.graph_store.upsert_entities(
        DEFAULT_SCOPE, [CanonicalEntity(canonical_id="top", name="Acme Group", type="ORG")]
    )
    before = _shot(components)
    a = _merge(client, "keep", ["dup"]).json()["merge_id"]
    b = _merge(client, "top", ["keep"]).json()["merge_id"]
    blocked = _split(client, "keep", merge_id=a)
    assert blocked.status_code == 409
    assert blocked.json()["error"]["details"] == {
        "reason": "survivor_merged",
        "merged_into": "top",
        "merge_id": b,
    }
    assert _split(client, "top", merge_id=b).status_code == 200
    assert _split(client, "keep", merge_id=a).status_code == 200
    assert _shot(components) == before


def test_a_dropped_id_that_is_taken_again_blocks_the_split(stack) -> None:
    client, components = stack
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    components.entity_store.upsert(
        DEFAULT_SCOPE, CanonicalEntity(canonical_id="dup", name="Else", type="ORG")
    )
    response = _split(client, "keep", merge_id=merge_id)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["reason"] == "entity_exists"


def test_a_later_edge_stays_with_the_survivor(stack) -> None:
    client, components = stack
    from tests.unit.test_entity_merge import _edge

    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    components.graph_store.write_triples(DEFAULT_SCOPE, [_edge("cat", "keep", "d9")])
    _split(client, "keep", merge_id=merge_id)
    keep_edges = {
        (e.subject_id, e.provenance.source_doc_id)
        for e in components.graph_store.get_node_edges(DEFAULT_SCOPE, "keep")
    }
    assert keep_edges == {("bob", "d1"), ("cat", "d9")}


def test_a_later_ingest_names_the_restored_entity_by_its_own_name() -> None:
    components = build_offline_components()
    queue = Queue()
    client = TestClient(_app(components, queue))
    for n, text in enumerate(["Acme Corp hired Bob Stone.", "Acme Holdings hired Carol Diaz."]):
        client.post("/v1/remember", json={"text": text, "source_id": f"n{n}"})
    drain(components, queue)
    ids = {e.name: e.canonical_id for e in components.entity_store.all(DEFAULT_SCOPE)}
    keep, dup = ids["Acme Corp"], ids["Acme Holdings"]
    merge_id = _merge(client, keep, [dup]).json()["merge_id"]
    assert _split(client, keep, merge_id=merge_id).status_code == 200

    client.post("/v1/remember", json={"text": "Acme Holdings hired Dave Kim.", "source_id": "n2"})
    drain(components, queue)

    names = [e.name for e in components.entity_store.all(DEFAULT_SCOPE)]
    assert names.count("Acme Holdings") == 1 and names.count("Acme Corp") == 1
    detail = client.get(f"/v1/entities/{dup}").json()
    assert len(detail["sources"]) == 2
    assert "Dave Kim" in {n["entity"]["name"] for n in detail["neighbours"]}
    assert len(client.get(f"/v1/entities/{keep}").json()["sources"]) == 1


def test_a_namespace_and_another_tenant_cannot_split() -> None:
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
    merge_id = client.post(
        "/v1/entities/keep/merge", json={"merge_ids": ["dup"]}, headers=auth_a
    ).json()["merge_id"]

    url = "/v1/entities/keep/split"
    assert client.post(url, json={"merge_id": merge_id}, headers=auth_b).status_code == 404
    assert client.post(url, json={"merge_id": merge_id}).status_code == 401
    in_ns = client.post(url, json={"merge_id": merge_id, "namespace": "notes"}, headers=auth_a)
    assert in_ns.status_code == 404
    assert components.entity_store.get(Scope("tenant-a"), "dup") is None
    assert client.post(url, json={"merge_id": merge_id}, headers=auth_a).status_code == 200
    assert components.entity_store.get(Scope("tenant-a"), "dup") is not None


def test_without_a_curator_the_route_says_unavailable() -> None:
    components = build_offline_components()
    client = TestClient(create_app(components.object_store, Queue(), Settings()))
    assert client.post("/v1/entities/x/split", json={}).status_code == 503


class _Crash(Exception):
    pass


@pytest.mark.parametrize(
    ("owner", "method", "nth"),
    [
        ("merge_store", "put", 1),  # before anything: the record is not yet marked
        ("graph_store", "split_nodes", 1),
        ("document_store", "upsert", 1),
        ("entity_store", "upsert", 1),  # the restored entity
        ("entity_store", "upsert", 2),  # the survivor's aliases
        ("merge_store", "put", 2),  # everything applied, the record not yet finished
    ],
)
def test_a_crash_between_steps_is_healed_by_the_retry(
    stack, monkeypatch, owner, method, nth
) -> None:  # type: ignore[no-untyped-def]
    client, components = stack
    quiet = TestClient(_app(components), raise_server_exceptions=False)
    before = _shot(components)
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    target = getattr(components, owner)
    real = getattr(target, method)
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == nth:
            raise _Crash
        return real(*args, **kwargs)

    monkeypatch.setattr(target, method, flaky)
    assert _split(quiet, "keep", merge_id=merge_id).status_code == 500
    record = components.merge_store.get(DEFAULT_SCOPE, merge_id)
    assert record is not None
    assert record.status == ("done" if (owner, nth) == ("merge_store", 1) else "splitting")
    assert record.split_ids == []

    retry = _split(client, "keep", merge_id=merge_id)
    assert retry.status_code == 200
    assert [e["canonical_id"] for e in retry.json()["split"]] == ["dup"]
    assert retry.json()["already_split"] == []
    assert _shot(components) == before
    finished = components.merge_store.get(DEFAULT_SCOPE, merge_id)
    assert finished is not None and finished.status == "split" and finished.splitting_ids == []
    again = _split(client, "keep", merge_id=merge_id, entity_ids=["dup"])
    assert again.json()["split"] == [] and again.json()["already_split"] == ["dup"]


def test_a_merge_touching_a_half_split_record_finishes_it(stack, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, components = stack
    quiet = TestClient(_app(components), raise_server_exceptions=False)
    before = _shot(components)
    merge_id = _merge(client, "keep", ["dup"]).json()["merge_id"]
    real = components.entity_store.upsert
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Crash
        return real(*args, **kwargs)

    monkeypatch.setattr(components.entity_store, "upsert", flaky)
    assert _split(quiet, "keep", merge_id=merge_id).status_code == 500
    components.entity_store.upsert(
        DEFAULT_SCOPE, CanonicalEntity(canonical_id="dup3", name="Acme X", type="ORG")
    )
    assert _merge(client, "keep", ["dup3"]).status_code == 200  # heals the split first
    [first] = [r for r in components.merge_store.all(DEFAULT_SCOPE) if r.merge_id == merge_id]
    assert first.status == "split"
    assert components.entity_store.get(DEFAULT_SCOPE, "dup") is not None
    assert _shot(components)["edges"] == before["edges"]


def test_the_fakes_meet_the_split_checks_the_real_stores_must_meet() -> None:
    from graph_rag.fakes import InMemoryDocumentStore, InMemoryEntityStore, InMemoryGraphStore
    from graph_rag.merge_records import InMemoryMergeRecordStore

    check_entity_split(
        InMemoryEntityStore(),
        InMemoryGraphStore(),
        InMemoryDocumentStore(),
        InMemoryMergeRecordStore(),
        dims=4,
    )


def test_a_split_record_is_exported_with_the_tenant(tmp_path: Path) -> None:
    from graph_rag.merge_records import DroppedEntity, InMemoryMergeRecordStore, MergeRecord
    from graph_rag.tenant_admin import main
    from tests.unit.test_tenant_admin import fakes

    stores = fakes()
    stores.merges = InMemoryMergeRecordStore()
    stores.merges.put(
        Scope("acme"),
        MergeRecord(
            merge_id="m1",
            keep_id="k",
            status="split",
            created_at="2026-01-01T00:00:00Z",
            split_ids=["d"],
            dropped=[DroppedEntity(canonical_id="d", name="D", type="ORG")],
        ),
    )
    assert main(["export", "acme", "--out", str(tmp_path)], stores=stores) == 0
    [line] = (tmp_path / "acme" / "_root" / "merges.jsonl").read_text().splitlines()
    assert json.loads(line)["status"] == "split" and json.loads(line)["split_ids"] == ["d"]
    assert main(["delete", "acme", "--yes"], stores=stores) == 0
    assert stores.merges.all(Scope("acme")) == []
