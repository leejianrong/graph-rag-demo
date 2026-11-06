"""An MCP server that gives an agent ``remember`` and ``recall`` over a running graph-rag service.

It holds no pipeline of its own: each tool is one call to the ``/v1`` agent API (ADR-0014), so
what an agent sees is what the HTTP contract says. Run it over stdio (``graph-rag-mcp``) and point
it at the service with ``GRAPH_RAG_URL`` (default ``http://localhost:8000``). ``forget`` and
``lookup`` are GRA-37.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

__all__ = ["build_server", "main"]

_INSTRUCTIONS = (
    "A knowledge-graph memory. `remember` stores text (it is read into the graph in the "
    "background, so a `recall` straight after may not see it yet). `recall` answers a question "
    "from what was remembered and returns the evidence: the sentences it rests on, each with the "
    "document and character offsets it came from, so cite those rather than guessing."
)


def _post(client: httpx.Client, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """One ``/v1`` call; an error body or an unreachable service becomes a tool error."""
    try:
        response = client.post(path, json=body)
    except httpx.HTTPError as error:
        raise ToolError(f"Could not reach the graph-rag service: {error}") from error
    payload = response.json()
    if response.is_error:
        raise ToolError(payload["error"]["message"])
    return payload  # type: ignore[no-any-return]


def build_server(client: httpx.Client) -> MCPServer:
    """The MCP server, calling the graph-rag service through ``client`` (its ``base_url`` set)."""
    server = MCPServer("graph-rag", instructions=_INSTRUCTIONS)

    @server.tool()
    def remember(text: str, source_id: str | None = None) -> dict[str, Any]:
        """Store text in memory.

        Give a stable `source_id` (a URL, a path, a note id) to make repeats safe: the same id
        with the same text does nothing and returns status `unchanged`. Without one, identical
        text is still stored once.
        """
        body = {"text": text} | ({"source_id": source_id} if source_id else {})
        return _post(client, "/v1/remember", body)

    @server.tool()
    def recall(question: str) -> dict[str, Any]:
        """Answer a question from memory, with its evidence.

        Returns the answer entity, the ranked entities, the connected subgraph and the
        supporting sentences, each with its source document and character offsets.
        """
        return _post(client, "/v1/recall", {"question": question})

    return server


def main() -> None:
    """Serve over stdio against the service named by ``GRAPH_RAG_URL``."""
    url = os.environ.get("GRAPH_RAG_URL", "http://localhost:8000")
    build_server(httpx.Client(base_url=url, timeout=60)).run()


if __name__ == "__main__":
    main()
