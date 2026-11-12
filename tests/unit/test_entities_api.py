"""GET /v1/entities and /v1/entities/{id}: search and neighbourhood for the explorer (GRA-142)."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.entities_api import EntityExplorer
from graph_rag.fakes import InMemoryEntityStore
from graph_rag.models import CanonicalEntity
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_contract import SPEC_PATH, fits
from tests.unit.test_agent_api_v1 import TEXT, Queue, drain


def _app(components, queue):  # type: ignore[no-untyped-def]
    return create_app(
        components.object_store,
        queue,
        Settings(),
        explorer=EntityExplorer(
            components.entity_store, components.graph_store, components.document_store
        ),
    )


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    client = TestClient(_app(components, queue))
    client.post("/v1/remember", json={"text": TEXT, "source_id": "note-1"})
    client.post("/v1/remember", json={"text": "Acme Corp hired Bob Stone.", "source_id": "note-2"})
    drain(components, queue)
    return client, components


def _find(client: TestClient, name: str) -> dict:
    found = client.get("/v1/entities", params={"q": name}).json()["entities"]
    return next(e for e in found if e["name"] == name)


def test_search_matches_a_name_ignoring_case_and_punctuation(stack) -> None:
    client, _ = stack
    found = client.get("/v1/entities", params={"q": "ACME, corp"})
    assert found.status_code == 200
    names = [e["name"] for e in found.json()["entities"]]
    assert "Acme Corp" in names and "Ada Lovelace" not in names
    acme = next(e for e in found.json()["entities"] if e["name"] == "Acme Corp")
    assert acme["type"] == "ORG" and acme["canonical_id"] and acme["first_seen"]
    assert "vector" not in acme
    assert found.json()["next_cursor"] is None


def test_search_matches_an_alias() -> None:
    components = build_offline_components()
    components.entity_store.upsert(
        DEFAULT_SCOPE,
        CanonicalEntity(
            canonical_id="e1", name="International Business Machines", type="ORG", aliases=["IBM"]
        ),
    )
    client = TestClient(_app(components, Queue()))
    [hit] = client.get("/v1/entities", params={"q": "ibm"}).json()["entities"]
    assert hit["name"] == "International Business Machines" and hit["aliases"] == ["IBM"]


def test_search_filters_by_type_and_pages_newest_first_without_repeats() -> None:
    ticks = iter(range(1, 100))
    components = build_offline_components()
    store = InMemoryEntityStore(clock=lambda: f"2026-01-01T00:00:{next(ticks):02d}.000000Z")
    for n in range(5):
        store.upsert(
            DEFAULT_SCOPE, CanonicalEntity(canonical_id=f"e{n}", name=f"Org {n}", type="ORG")
        )
    store.upsert(DEFAULT_SCOPE, CanonicalEntity(canonical_id="p", name="Org Person", type="PERSON"))
    client = TestClient(
        create_app(
            components.object_store,
            Queue(),
            Settings(),
            explorer=EntityExplorer(store, components.graph_store, components.document_store),
        )
    )
    seen: list[str] = []
    cursor = None
    while True:
        page = client.get(
            "/v1/entities",
            params={"q": "org", "type": "ORG", "limit": 2} | ({"cursor": cursor} if cursor else {}),
        ).json()
        assert len(page["entities"]) <= 2
        seen += [e["canonical_id"] for e in page["entities"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == ["e4", "e3", "e2", "e1", "e0"]  # newest first, each once, no PERSON


def test_limit_is_capped_and_a_bad_cursor_or_type_is_invalid_request(stack) -> None:
    client, _ = stack
    for params in ({"limit": 101}, {"limit": 0}, {"cursor": "not-a-cursor"}, {"type": "WIDGET"}):
        response = client.get("/v1/entities", params=params)
        assert response.status_code == 400, params
        assert response.json()["error"]["code"] == "invalid_request"
    assert client.get("/v1/entities", params={"namespace": "a/b"}).status_code == 400


def test_detail_returns_neighbours_by_direction_aliases_and_sources(stack) -> None:
    client, _ = stack
    acme = _find(client, "Acme Corp")
    body = client.get(f"/v1/entities/{acme['canonical_id']}").json()
    assert body["canonical_id"] == acme["canonical_id"] and body["first_seen"]
    assert {n["direction"] for n in body["neighbours"]} == {"outgoing"}  # Acme Corp points at both
    london = client.get(f"/v1/entities/{_find(client, 'London')['canonical_id']}").json()
    assert [(n["direction"], n["entity"]["name"]) for n in london["neighbours"]] == [
        ("incoming", "Acme Corp")
    ]
    assert (london["outgoing_total"], london["incoming_total"]) == (0, 1)
    assert body["outgoing_total"] + body["incoming_total"] >= len(body["neighbours"]) > 0
    for n in body["neighbours"]:
        assert n["predicate"] and n["entity"]["canonical_id"] and n["provenance"]["source_doc_id"]
    sources = {s["name"]: s["mentions"] for s in body["sources"]}
    assert set(sources) == {"note-1", "note-2"} and all(c >= 1 for c in sources.values())
    assert body["sources_total"] == 2
    assert all(set(s) == {"document_id", "name", "mentions"} for s in body["sources"])  # no text


def test_detail_caps_neighbours_and_sources_and_reports_the_totals(stack) -> None:
    client, _ = stack
    acme = _find(client, "Acme Corp")
    body = client.get(
        f"/v1/entities/{acme['canonical_id']}", params={"neighbours_limit": 1, "sources_limit": 1}
    ).json()
    assert len([n for n in body["neighbours"] if n["direction"] == "outgoing"]) <= 1
    assert len([n for n in body["neighbours"] if n["direction"] == "incoming"]) <= 1
    assert len(body["sources"]) == 1 and body["sources_total"] == 2
    assert (
        client.get(
            f"/v1/entities/{acme['canonical_id']}", params={"neighbours_limit": 101}
        ).status_code
        == 400
    )


def test_detail_of_an_unknown_id_is_not_found_in_the_contract_error_shape(stack) -> None:
    client, _ = stack
    response = client.get("/v1/entities/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_without_an_explorer_the_routes_say_unavailable() -> None:
    components = build_offline_components()
    client = TestClient(create_app(components.object_store, Queue(), Settings()))
    for path in ("/v1/entities", "/v1/entities/x"):
        response = client.get(path)
        assert response.status_code == 503 and response.json()["error"]["code"] == "unavailable"


def test_namespaces_and_tenants_see_only_their_own_entities() -> None:
    components = build_offline_components()
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True),
        explorer=EntityExplorer(
            components.entity_store, components.graph_store, components.document_store
        ),
        keys=keys,
    )
    client = TestClient(app)
    a, _ = keys.create("tenant-a")
    b, _ = keys.create("tenant-b")
    auth_a, auth_b = ({"Authorization": f"Bearer {s}"} for s in (a, b))

    client.post("/v1/remember", json={"text": TEXT, "source_id": "n"}, headers=auth_a)
    client.post(
        "/v1/remember", json={"text": TEXT, "source_id": "n", "namespace": "notes"}, headers=auth_a
    )
    drain(components, queue)

    mine = client.get("/v1/entities", params={"q": "acme"}, headers=auth_a).json()["entities"]
    assert mine
    acme_id = mine[0]["canonical_id"]
    assert client.get("/v1/entities", params={"q": "acme"}, headers=auth_b).json()["entities"] == []
    assert client.get(f"/v1/entities/{acme_id}", headers=auth_b).status_code == 404
    assert client.get(f"/v1/entities/{acme_id}", headers=auth_a).status_code == 200

    in_notes = client.get(
        "/v1/entities", params={"q": "acme", "namespace": "notes"}, headers=auth_a
    ).json()["entities"]
    assert in_notes and {e["canonical_id"] for e in in_notes}.isdisjoint(
        {e["canonical_id"] for e in mine}
    )
    other_ns = {"namespace": "elsewhere"}
    assert (
        client.get("/v1/entities", params={"q": "acme"} | other_ns, headers=auth_a).json()[
            "entities"
        ]
        == []
    )
    assert client.get(f"/v1/entities/{acme_id}", params=other_ns, headers=auth_a).status_code == 404

    assert client.get("/v1/entities").status_code == 401  # no key


def test_real_responses_fit_the_contract(stack) -> None:
    client, _ = stack
    spec = yaml.safe_load(SPEC_PATH.read_text())
    page = client.get("/v1/entities", params={"limit": 1}).json()
    assert page["next_cursor"] and fits(spec, "EntitySearchResponse", page) == []
    acme = _find(client, "Acme Corp")
    detail = client.get(f"/v1/entities/{acme['canonical_id']}").json()
    assert fits(spec, "EntityNeighbourhood", detail) == []
