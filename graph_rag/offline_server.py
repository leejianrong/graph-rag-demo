"""Serve the whole service on one process, with no Docker, model or LLM key (GRA-41).

The same HTTP API as the real stack, over in-memory stores and the offline heuristic stages
(`build_offline_components`). It reads each document as soon as it is stored, so a ``recall``
straight after a ``remember`` already sees it. Nothing is kept: stop the process and the memory
is gone. The heuristic entity and fact extraction is much weaker than the real stack's, so use it
to try the API and the agent tools, not to judge answer quality.

``graph-rag-offline`` listens on ``127.0.0.1:8000``; ``HOST`` and ``PORT`` change that.
``REQUIRE_AUTH=true`` turns on API keys (create one with ``graph-rag-keys``).
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from graph_rag.api import create_app
from graph_rag.bind import enforce_bind
from graph_rag.config import Settings
from graph_rag.logging import get_logger
from graph_rag.lookup import EntityLookup
from graph_rag.models import IngestTrigger
from graph_rag.wiring import build_offline_components

__all__ = ["build_app", "main"]

_logger = get_logger(__name__)


def build_app(settings: Settings | None = None) -> FastAPI:
    """The service over in-memory stores, processing each document as it arrives."""
    settings = settings or Settings()
    components = build_offline_components(settings=settings)

    class Inline:
        def publish(self, trigger: IngestTrigger) -> None:
            try:
                components.orchestrator.process_document(trigger)
            except Exception:  # noqa: BLE001 - one bad document must not fail the request
                _logger.exception("could not process %s", trigger.object_key)

    return create_app(
        components.object_store,
        Inline(),
        settings,
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        lookup=EntityLookup(components.entity_store, components.graph_store),
        jobs=components.job_store,
    )


def main() -> None:
    """Run the offline service."""
    settings = Settings()
    host = os.environ.get("HOST", "127.0.0.1")
    enforce_bind(host, settings, _logger)
    uvicorn.run(build_app(settings), host=host, port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
