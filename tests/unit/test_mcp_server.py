"""The MCP server (GRA-36), driven through the real protocol over the offline pipeline."""

from __future__ import annotations

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.lookup import EntityLookup
from graph_rag.mcp_server import build_server
from graph_rag.models import IngestTrigger
from graph_rag.wiring import build_offline_components

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


class Queue:
    def __init__(self) -> None:
        self.triggers: list[IngestTrigger] = []

    def publish(self, trigger: IngestTrigger) -> None:
        self.triggers.append(trigger)


@pytest.fixture
def pipeline():
    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        lookup=EntityLookup(components.entity_store, components.graph_store),
    )
    return TestClient(app, base_url="http://graph-rag"), components, queue


def call(http: httpx.Client, tool: str, arguments: dict):
    async def run():
        async with Client(build_server(http)) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(run)


def test_every_tool_is_listed_with_a_description(pipeline) -> None:
    async def run():
        async with Client(build_server(pipeline[0])) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in anyio.run(run)}
    assert set(tools) == {"remember", "recall", "forget", "lookup"}
    assert all(t.description for t in tools.values())


def test_remember_then_recall_returns_the_evidence_with_provenance(pipeline) -> None:
    http, components, queue = pipeline
    stored = call(http, "remember", {"text": TEXT, "source_id": "note-1"})
    assert stored.structured_content["status"] == "accepted"
    for trigger in queue.triggers:
        assert components.orchestrator.process_document(trigger)

    found = call(http, "recall", {"question": "Where is Acme Corp based?"})
    sentences = found.structured_content["supporting_sentences"]
    assert not found.is_error and sentences
    assert {"text", "document_id", "char_start", "char_end"} <= set(sentences[0])


def test_a_repeated_remember_is_unchanged(pipeline) -> None:
    http, _, _ = pipeline
    call(http, "remember", {"text": "same", "source_id": "s"})
    again = call(http, "remember", {"text": "same", "source_id": "s"})
    assert again.structured_content["status"] == "unchanged"


def test_an_api_error_reaches_the_agent_as_a_tool_error(pipeline) -> None:
    result = call(pipeline[0], "remember", {"text": ""})
    assert result.is_error and "text" in result.content[0].text


def test_an_unreachable_service_is_a_tool_error_not_a_crash() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(base_url="http://nowhere", transport=httpx.MockTransport(refuse))
    result = call(http, "recall", {"question": "q"})
    assert result.is_error and "Could not reach" in result.content[0].text


def test_a_namespace_keeps_memory_apart_through_the_tools(pipeline) -> None:
    http, components, queue = pipeline
    call(http, "remember", {"text": TEXT, "source_id": "n", "namespace": "work"})
    for trigger in queue.triggers:
        assert components.orchestrator.process_document(trigger)

    ask = {"question": "Where is Acme Corp based?"}

    def found(extra: dict) -> list:
        return call(http, "recall", ask | extra).structured_content["supporting_sentences"]

    assert found({"namespace": "work"})
    assert found({}) == [] and found({"namespace": "home"}) == []


def test_forget_removes_what_recall_returned(pipeline) -> None:
    http, components, queue = pipeline
    call(http, "remember", {"text": TEXT, "source_id": "note-1"})
    for trigger in queue.triggers:
        assert components.orchestrator.process_document(trigger)
    question = {"question": "Where is Acme Corp based?"}
    assert call(http, "recall", question).structured_content["supporting_sentences"]

    gone = call(http, "forget", {"source_id": "note-1"})
    assert not gone.is_error and gone.structured_content["source_id"] == "note-1"
    assert call(http, "recall", question).structured_content["supporting_sentences"] == []
    again = call(http, "forget", {"source_id": "note-1"})
    assert again.is_error and "No source" in again.content[0].text


def test_lookup_returns_neighbours_and_reports_a_miss_as_a_tool_error(pipeline) -> None:
    http, components, queue = pipeline
    call(http, "remember", {"text": TEXT, "source_id": "note-1"})
    for trigger in queue.triggers:
        assert components.orchestrator.process_document(trigger)

    found = call(http, "lookup", {"name": "Acme Corp"})
    [acme] = found.structured_content["entities"]
    assert acme["neighbours"] and acme["neighbours"][0]["provenance"]["source_sentence"]
    assert call(http, "lookup", {"entity_id": acme["canonical_id"]}).is_error is False
    miss = call(http, "lookup", {"name": "Nobody Inc"})
    assert miss.is_error and "No entity" in miss.content[0].text
    assert call(http, "lookup", {}).is_error
