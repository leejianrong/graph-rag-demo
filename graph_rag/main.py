"""Composition root / service entrypoint (N19–N21).

Wires the real stack for local Docker Compose and runs the service:

* build :class:`~graph_rag.config.Settings` and configure the logging seam;
* construct the real adapters — ``MinioObjectStore`` and ``EsDocumentStore``
  (creating the documents index) — behind their ports;
* build the :class:`~graph_rag.orchestrator.Orchestrator` over those ports;
* build the ``KafkaTriggerPublisher`` and the FastAPI app via ``create_app``;
* start the ``KafkaTriggerConsumer`` in a background thread, injecting
  ``orchestrator.process_document`` as its handler (the thin driver, ADR-0001);
* serve the FastAPI app with uvicorn.

This module only *wires* — no business logic lives here. Adapters/messaging/API
modules marked below are owned by Agent B; this root imports them by their
contracted names.
"""

from __future__ import annotations

import threading

import uvicorn

from graph_rag.adapters.es_document_store import EsDocumentStore
from graph_rag.adapters.minio_object_store import MinioObjectStore
from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.logging import configure_logging, get_logger
from graph_rag.messaging.kafka_trigger import KafkaTriggerConsumer, KafkaTriggerPublisher
from graph_rag.orchestrator import Orchestrator
from graph_rag.stages.ner import SpacyNerStage

__all__ = ["main"]


def main() -> None:
    """Wire the real stack and run the ingest/query service.

    Blocks serving the FastAPI app; the Kafka consumer runs in a daemon thread.
    """
    settings = Settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    # --- Real adapters behind the V1-active ports ---------------------------
    object_store = MinioObjectStore.from_settings(settings)
    document_store = EsDocumentStore.from_settings(settings)
    document_store.ensure_index()

    # --- NER stage (N6): real spaCy pipeline from Settings.ner_model --------
    ner_stage = SpacyNerStage.from_settings(settings)

    # --- The pipeline shell over those ports + stages -----------------------
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner_stage,
    )

    # --- Messaging seam: publisher (for POST /ingest) + consumer driver -----
    publisher = KafkaTriggerPublisher.from_settings(settings)
    consumer = KafkaTriggerConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic=settings.ingest_trigger_topic,
        handler=orchestrator.process_document,
    )
    consumer_thread = threading.Thread(
        target=consumer.run_forever,
        name="kafka-trigger-consumer",
        daemon=True,
    )
    consumer_thread.start()
    logger.info("started Kafka trigger consumer on topic %r", settings.ingest_trigger_topic)

    # --- FastAPI app (POST /ingest publishes through the port) --------------
    app = create_app(object_store, publisher, settings)

    logger.info("serving FastAPI app on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 — local demo, bind all.


if __name__ == "__main__":
    main()
