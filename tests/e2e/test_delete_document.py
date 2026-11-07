"""Fast E2E for deleting a document: its edges, its record, and entities only it used (GRA-28).

Runs the whole offline pipeline over the in-memory fakes, so it needs no Docker and no model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.ids import document_id
from graph_rag.models import IngestTrigger
from graph_rag.orchestrator import Orchestrator
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.wiring import HeuristicKgBuildStage, HeuristicNerStage

pytestmark = pytest.mark.e2e

BUCKET = Settings().minio_bucket
ADA = b"Ada Lovelace works for Acme Corp. Acme Corp is based in London."
GRACE = b"Grace Hopper works for Acme Corp."


class Stack:
    """The offline pipeline with every store kept in reach, ingesting synchronously."""

    def __init__(self) -> None:
        self.objects = InMemoryObjectStore()
        self.documents = InMemoryDocumentStore()
        self.entities = InMemoryEntityStore()
        self.graph = InMemoryGraphStore()
        self.embedder = FakeEmbedder()
        self.orchestrator = Orchestrator(
            object_store=self.objects,
            document_store=self.documents,
            ner_stage=HeuristicNerStage(),
            coref_stage=LLMCorefStage(FakeLLMClient()),
            entity_linking_stage=EntityLinkingStage(
                self.entities, self.embedder, threshold=-1.0, knn_top_k=0
            ),
            graph_store=self.graph,
            kg_build_stage=HeuristicKgBuildStage(),
            entity_store=self.entities,
        )

    def ingest(self, key: str, data: bytes) -> str:
        self.objects.put(DEFAULT_SCOPE, BUCKET, key, data)
        assert self.orchestrator.process_document(IngestTrigger(bucket=BUCKET, object_key=key))
        return document_id(BUCKET, key)

    def names(self) -> set[str]:
        return {e.name for e in self.entities.all(DEFAULT_SCOPE)}

    def sentences_of(self, doc_id: str) -> list:
        vector = self.embedder.embed(["Ada Lovelace Acme Corp London Grace Hopper"])[0]
        return [
            s
            for s in self.documents.search_sentences(DEFAULT_SCOPE, vector=vector, top_k=50)
            if s.document_id == doc_id
        ]


@pytest.fixture
def stack() -> Stack:
    return Stack()


def test_deleting_a_document_removes_its_record_edges_bytes_and_sentences(stack: Stack) -> None:
    ada = stack.ingest("ada.txt", ADA)
    grace = stack.ingest("grace.txt", GRACE)
    edges_before = stack.graph.edge_count(DEFAULT_SCOPE)
    assert stack.sentences_of(ada) and stack.documents.get(DEFAULT_SCOPE, ada) is not None

    result = stack.orchestrator.delete_document(ada)

    assert result is not None and result.document_id == ada and result.object_key == "ada.txt"
    assert stack.documents.get(DEFAULT_SCOPE, ada) is None
    assert stack.sentences_of(ada) == []  # recall can no longer return them
    assert stack.graph.edge_count(DEFAULT_SCOPE) < edges_before
    assert all(
        e.provenance.source_doc_id != ada
        for e in [
            t
            for n in stack.entities.all(DEFAULT_SCOPE)
            for t in stack.graph.get_node_edges(DEFAULT_SCOPE, n.canonical_id)
        ]
    )
    with pytest.raises(FileNotFoundError):
        stack.objects.get(DEFAULT_SCOPE, BUCKET, "ada.txt")
    assert stack.documents.get(DEFAULT_SCOPE, grace) is not None and stack.sentences_of(grace)


def test_an_entity_another_document_still_mentions_stays_and_the_rest_go(stack: Stack) -> None:
    ada = stack.ingest("ada.txt", ADA)
    stack.ingest("grace.txt", GRACE)
    assert {"Ada Lovelace", "Acme Corp", "London", "Grace Hopper"} <= stack.names()
    nodes_before = stack.graph.node_count(DEFAULT_SCOPE)

    result = stack.orchestrator.delete_document(ada)

    kept = {stack.entities.get(DEFAULT_SCOPE, i).name for i in result.entities_kept}
    removed_ids = set(result.entities_removed)
    assert kept == {"Acme Corp"}
    assert stack.names() >= {"Acme Corp", "Grace Hopper"}
    assert not stack.names() & {"Ada Lovelace", "London"}
    assert all(stack.graph.get_node(DEFAULT_SCOPE, i) is None for i in removed_ids) and removed_ids
    assert stack.graph.node_count(DEFAULT_SCOPE) == nodes_before - len(removed_ids)


def test_deleting_the_only_document_leaves_nothing_behind(stack: Stack) -> None:
    only = stack.ingest("only.txt", ADA)
    result = stack.orchestrator.delete_document(only)

    assert result.entities_kept == [] and result.entities_removed
    assert stack.entities.count(DEFAULT_SCOPE) == 0
    assert (stack.graph.node_count(DEFAULT_SCOPE), stack.graph.edge_count(DEFAULT_SCOPE)) == (0, 0)


def test_deleting_an_unknown_document_is_none_and_touches_nothing(stack: Stack) -> None:
    grace = stack.ingest("grace.txt", GRACE)
    before = (
        stack.entities.count(DEFAULT_SCOPE),
        stack.graph.node_count(DEFAULT_SCOPE),
        stack.graph.edge_count(DEFAULT_SCOPE),
    )

    assert stack.orchestrator.delete_document("0" * 64) is None
    assert (
        stack.entities.count(DEFAULT_SCOPE),
        stack.graph.node_count(DEFAULT_SCOPE),
        stack.graph.edge_count(DEFAULT_SCOPE),
    ) == before
    assert stack.documents.get(DEFAULT_SCOPE, grace) is not None


def test_a_delete_that_stopped_part_way_can_be_run_again(stack: Stack) -> None:
    """The record goes last, so a repeat finishes what a failed call left."""
    ada = stack.ingest("ada.txt", ADA)
    stack.ingest("grace.txt", GRACE)
    stack.graph.delete_document_edges(
        DEFAULT_SCOPE, ada
    )  # the first step of a delete that then failed

    result = stack.orchestrator.delete_document(ada)

    assert result is not None and "Ada Lovelace" not in stack.names()
    assert stack.orchestrator.delete_document(ada) is None


def _client(stack: Stack | None, *, wired: bool = True) -> TestClient:
    stack = stack or Stack()
    eraser = stack.orchestrator.delete_document if wired else None
    return TestClient(create_app(stack.objects, _NoPublish(), settings=Settings(), eraser=eraser))


class _NoPublish:
    def publish(self, trigger: IngestTrigger) -> None:
        return None


def test_the_delete_endpoint_reports_what_it_removed(stack: Stack) -> None:
    ada = stack.ingest("ada.txt", ADA)
    stack.ingest("grace.txt", GRACE)
    client = _client(stack)

    response = client.delete(f"/documents/{ada}")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == ada and body["bucket"] == BUCKET and body["entities_removed"]
    assert client.delete(f"/documents/{ada}").status_code == 404  # already gone


def test_the_delete_endpoint_404s_for_an_unknown_id_and_503s_when_not_wired() -> None:
    assert _client(None).delete("/documents/nothing").status_code == 404
    assert _client(None, wired=False).delete("/documents/nothing").status_code == 503


def test_a_deleted_source_can_be_ingested_again(stack: Stack) -> None:
    """Deleting also removes the raw bytes, so the same bytes are new again, not "unchanged"."""
    ada = stack.ingest("ada.txt", ADA)
    client = TestClient(
        create_app(
            stack.objects,
            _Runs(stack),
            settings=Settings(),
            eraser=stack.orchestrator.delete_document,
        )
    )
    files = {"file": ("ada.txt", ADA, "text/plain")}
    assert client.post("/ingest", files=files).json()["status"] == "unchanged"

    client.delete(f"/documents/{ada}")

    assert client.post("/ingest", files=files).json()["status"] == "accepted"
    assert stack.documents.get(DEFAULT_SCOPE, ada) is not None and "Ada Lovelace" in stack.names()


class _Runs:
    def __init__(self, stack: Stack) -> None:
        self._stack = stack

    def publish(self, trigger: IngestTrigger) -> None:
        self._stack.orchestrator.process_document(trigger)
