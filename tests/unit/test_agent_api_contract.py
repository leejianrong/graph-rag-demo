"""The agent API contract (docs/api/agent-api.openapi.yaml, ADR-0014) against the code (GRA-35).

The contract is hand-written, so these tests keep it honest: it is valid OpenAPI, every operation
that claims to exist maps onto an endpoint the app really serves, real responses fit its schemas,
and a field it calls planned is not yet accepted (so the day one is built, this fails and the
status gets updated).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.fakes import InMemoryObjectStore, InMemoryTriggerPublisher
from graph_rag.models import DeletedDocument, QueryRequest, QueryResponse
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import build_offline_components

SPEC_PATH = Path(__file__).resolve().parents[2] / "docs" / "api" / "agent-api.openapi.yaml"
METHODS = ("get", "post", "put", "patch", "delete")


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())


@pytest.fixture(scope="module")
def served() -> dict:
    """The OpenAPI document FastAPI generates for the app as it is today."""
    app = create_app(InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings())
    return app.openapi()


def operations(spec: dict):
    for path, item in spec["paths"].items():
        for method in METHODS:
            if method in item:
                yield path, method, item[method]


def fits(spec: dict, schema_name: str, instance: object) -> list[str]:
    """Schema violations of ``instance`` against a named component schema."""
    root = {"$ref": f"#/components/schemas/{schema_name}", "components": spec["components"]}
    return [e.message for e in Draft202012Validator(root).iter_errors(instance)]


def test_the_contract_is_valid_openapi(spec: dict) -> None:
    validate(spec)  # raises on any structural error or unresolved $ref


def references(node: object):
    """Every ``$ref`` in the document."""
    if isinstance(node, dict):
        if "$ref" in node:
            yield node["$ref"]
        for value in node.values():
            yield from references(value)
    elif isinstance(node, list):
        for value in node:
            yield from references(value)


def test_every_reference_resolves(spec: dict) -> None:
    """The validator does not flag a $ref to nothing, so check them here."""
    refs = set(references(spec))
    assert len(refs) > 10
    for ref in refs:
        assert ref.startswith("#/"), ref
        target: object = spec
        for part in ref[2:].split("/"):
            assert isinstance(target, dict) and part in target, f"{ref} points at nothing"
            target = target[part]


def test_the_four_verbs_are_there_each_with_a_unique_operation_id(spec: dict) -> None:
    ids = [op["operationId"] for _, _, op in operations(spec)]
    assert len(ids) == len(set(ids))
    assert {"remember", "recall", "forget", "lookup"} <= set(ids)


def test_every_operation_says_whether_it_exists_and_where_to_find_it(
    spec: dict, served: dict
) -> None:
    for path, method, op in operations(spec):
        status = op["x-status"]
        assert status in {"existing", "planned"}, (path, method)
        if status == "existing":
            target = op["x-maps-to"]
            assert method_in(served, target["method"], target["path"]), (path, target)
            assert method_in(served, method, path), f"{path} is marked existing but not served"
        else:
            assert op["x-card"].startswith("GRA-"), (path, method)


def method_in(served: dict, method: str, path: str) -> bool:
    return method in served["paths"].get(path, {})


def test_every_error_response_uses_the_one_error_shape(spec: dict) -> None:
    for name, response in spec["components"]["responses"].items():
        ref = response["content"]["application/json"]["schema"]["$ref"]
        assert ref == "#/components/schemas/Error", name
    for path, _method, op in operations(spec):
        for code, response in op["responses"].items():
            if code.startswith(("4", "5")):
                assert response["$ref"].startswith("#/components/responses/"), (path, code)


def test_an_error_body_needs_a_known_code_and_a_message(spec: dict) -> None:
    ok = {"error": {"code": "not_found", "message": "No document x."}}
    assert fits(spec, "Error", ok) == []
    assert fits(spec, "Error", {"error": {"code": "teapot", "message": "x"}})
    assert fits(spec, "Error", {"error": {"code": "not_found"}})
    assert fits(spec, "Error", {"message": "bare"})


def test_a_real_recall_response_fits_the_schema() -> None:
    """Ingest through the offline pipeline, ask, and check the real answer against the contract."""
    spec = yaml.safe_load(SPEC_PATH.read_text())
    components = build_offline_components()
    bucket = components.bucket
    text = b"Ada Lovelace works for Acme Corp. Acme Corp is based in London."
    components.object_store.put(DEFAULT_SCOPE, bucket, "n.txt", text)
    from graph_rag.models import IngestTrigger

    assert components.orchestrator.process_document(
        IngestTrigger(bucket=bucket, object_key="n.txt")
    )
    response = components.retriever.retrieve(QueryRequest(question="Where is Acme Corp based?"))
    body = response.model_dump(mode="json")
    assert body["supporting_sentences"] and body["subgraph"]["edges"]  # a real, non-trivial answer
    assert fits(spec, "RecallResponse", body) == []
    assert (
        fits(
            spec,
            "RecallResponse",
            QueryResponse(
                answer=None, answer_entity=None, subgraph={"nodes": [], "edges": []}
            ).model_dump(mode="json"),
        )
        == []
    )  # nothing found is still a valid answer


def test_recall_request_fields_that_exist_today_are_in_the_contract(spec: dict) -> None:
    properties = spec["components"]["schemas"]["RecallRequest"]["properties"]
    assert set(QueryRequest.model_fields) <= set(properties)
    assert spec["components"]["schemas"]["RecallRequest"]["required"] == ["question"]


def test_a_real_forget_response_fits_the_schema(spec: dict) -> None:
    body = DeletedDocument(
        document_id="d", bucket="b", object_key="k", entities_removed=["e1"], entities_kept=["e2"]
    ).model_dump(mode="json")
    assert fits(spec, "ForgetResponse", body) == []


def test_remember_maps_onto_ingest_and_its_statuses_are_the_ones_ingest_returns(spec: dict) -> None:
    app = create_app(InMemoryObjectStore(), InMemoryTriggerPublisher(), Settings())
    client = TestClient(app)
    files = {"file": ("a.txt", b"same", "text/plain")}
    first = client.post("/ingest", files=files, data={"source_url": "src-1"}).json()
    again = client.post("/ingest", files=files, data={"source_url": "src-1"}).json()

    statuses = spec["components"]["schemas"]["RememberResponse"]["properties"]["status"]["enum"]
    assert {first["status"], again["status"]} == set(statuses)
    contract = spec["components"]["schemas"]["RememberResponse"]["properties"]
    assert "document_id" in contract and first["document_id"] and first["object_key"] == "src-1"
    assert fits(spec, "RememberResponse", {**first, "source_id": first["object_key"]}) == []


def planned_fields(spec: dict):
    for name in ("RememberRequest", "RecallRequest", "RecallResponse"):
        for field, schema in spec["components"]["schemas"][name]["properties"].items():
            if schema.get("x-status") == "planned":
                yield name, field


def ingest_fields(served: dict) -> set[str]:
    """The form fields and query parameters ``POST /ingest`` accepts today."""
    post = served["paths"]["/ingest"]["post"]
    ref = post["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    form = served["components"]["schemas"][ref.rsplit("/", 1)[1]]["properties"]
    return set(form) | {p["name"] for p in post.get("parameters", [])}


def test_a_field_the_contract_calls_planned_is_not_accepted_yet(spec: dict, served: dict) -> None:
    """When a card builds one of these, this fails: flip its x-status to existing."""
    planned = list(planned_fields(spec))
    assert {"RememberRequest", "RecallRequest", "RecallResponse"} == {n for n, _ in planned}
    accepted = {
        "RememberRequest": ingest_fields(served),
        "RecallRequest": set(QueryRequest.model_fields),
        "RecallResponse": set(QueryResponse.model_fields),
    }
    for name, field in planned:
        assert field not in accepted[name], f"{name}.{field} is built; mark it existing"


def test_the_existing_remember_fields_are_accepted_by_ingest(spec: dict, served: dict) -> None:
    properties = spec["components"]["schemas"]["RememberRequest"]["properties"]
    existing = {f for f, s in properties.items() if s.get("x-status") != "planned"}
    assert {"force", "source_id", "text"} == existing
    # text arrives as the uploaded file, source_id as source_url; force is a query parameter
    assert {"file", "source_url", "force"} <= ingest_fields(served)
