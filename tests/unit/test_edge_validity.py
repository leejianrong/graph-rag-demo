"""Validity intervals on edges (GRA-42): set from the document's date, null when unknown."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.models import EdgeProvenance, IngestTrigger, Triple
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    app = create_app(components.object_store, queue, Settings(), retriever=components.retriever)
    return TestClient(app), components, queue


def _edges(components) -> list[Triple]:
    graph = components.graph_store
    ids = {n.canonical_id for n in graph.khop(DEFAULT_SCOPE, _all_ids(components), 0).nodes}
    return [e for i in sorted(ids) for e in graph.get_node_edges(DEFAULT_SCOPE, i)]


def _all_ids(components) -> list[str]:
    return [e.canonical_id for e in components.entity_store.all(DEFAULT_SCOPE)]


def test_edges_start_when_the_document_says_it_was_written(stack) -> None:
    client, components, queue = stack
    sent = client.post(
        "/v1/remember", json={"text": TEXT, "source_id": "n", "document_date": "2019-05-01"}
    )
    assert sent.status_code == 202
    drain(components, queue)
    edges = _edges(components)
    assert edges, "the offline pipeline should find at least one edge"
    assert {e.valid_from for e in edges} == {"2019-05-01"}
    assert {e.valid_to for e in edges} == {None}  # nothing sets an end yet


def test_an_undated_document_leaves_the_interval_unknown(stack) -> None:
    client, components, queue = stack
    client.post("/v1/remember", json={"text": TEXT, "source_id": "n"})
    drain(components, queue)
    edges = _edges(components)
    assert edges and {(e.valid_from, e.valid_to) for e in edges} == {(None, None)}


def test_ingest_takes_the_date_as_a_form_field(stack) -> None:
    client, components, queue = stack
    sent = client.post(
        "/ingest",
        files={"file": ("n.txt", TEXT.encode())},
        data={"document_date": "2020-01-31"},
    )
    assert sent.status_code == 200
    assert queue.triggers[0].document_date == "2020-01-31"
    drain(components, queue)
    assert {e.valid_from for e in _edges(components)} == {"2020-01-31"}


@pytest.mark.parametrize("bad", ["2019", "2019-5-1", "20190501", "2019-02-30", "yesterday", ""])
def test_a_malformed_date_is_refused_and_nothing_is_queued(stack, bad: str) -> None:
    client, _, queue = stack
    v1 = client.post("/v1/remember", json={"text": TEXT, "document_date": bad})
    assert v1.status_code in (400, 422) and v1.json()["error"]["code"] == "invalid_request"
    assert queue.triggers == []
    if bad:  # an empty form field means "not sent" to /ingest
        legacy = client.post(
            "/ingest", files={"file": ("n.txt", TEXT.encode())}, data={"document_date": bad}
        )
        assert legacy.status_code in (400, 422)
        assert queue.triggers == []


def test_a_trigger_without_a_date_still_parses() -> None:
    # A Kafka message from before this change has no document_date.
    trigger = IngestTrigger.model_validate_json('{"bucket": "b", "object_key": "k"}')
    assert trigger.document_date is None


def test_the_model_refuses_a_bad_interval() -> None:
    provenance = EdgeProvenance(source_doc_id="d", sentence_index=0, source_sentence="s")
    ok = Triple(
        subject_id="a", predicate="P", object_id="b", provenance=provenance, valid_from="2019-05-01"
    )
    assert ok.valid_to is None
    with pytest.raises(ValidationError):
        Triple(subject_id="a", predicate="P", object_id="b", provenance=provenance, valid_to="soon")
