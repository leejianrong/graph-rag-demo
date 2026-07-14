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

from graph_rag.adapters.embedder import SentenceTransformerEmbedder
from graph_rag.adapters.es_document_store import EsDocumentStore
from graph_rag.adapters.es_entity_store import EsEntityStore
from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.adapters.minio_object_store import MinioObjectStore
from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore
from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.logging import configure_logging, get_logger
from graph_rag.messaging.kafka_trigger import KafkaTriggerConsumer, KafkaTriggerPublisher
from graph_rag.orchestrator import Orchestrator
from graph_rag.query.retriever import QueryRetriever
from graph_rag.query.synthesis import AnswerSynthesizer
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.kg_build import KgBuildStage
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

    # --- Coref stage (N7): LiteLLM client + response cache (V3) -------------
    llm_client = LiteLLMClient.from_settings(settings)
    coref_stage = LLMCorefStage(llm_client)

    # --- Entity linking (N8, V4): local embedder + ES-Entities canonical store.
    #     Both behind their ports (ADR-0010); the stage blocks/scores/merges and
    #     drives the EL checkpoint. Gated tie-breaker/NIL stay off (Settings).
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    entity_store = EsEntityStore.from_settings(settings)
    entity_store.ensure_index()
    entity_linking_stage = EntityLinkingStage.from_settings(settings, entity_store, embedder)

    # --- KG-build (N9, V5): the graph store + LLM triple extractor ----------
    #     The Neo4j GraphStore holds multi-label nodes + provenance edges; the
    #     KG-build stage emits triples over canonical IDs (its own kg_build_model,
    #     sharing the LLM response cache). The orchestrator runs the graph
    #     checkpoint (delete-then-write) so re-ingest replaces a doc's edges.
    graph_store = Neo4jGraphStore.from_settings(settings)
    graph_store.init()
    kg_build_stage = KgBuildStage.from_settings(settings)

    # --- The pipeline shell over those ports + stages -----------------------
    orchestrator = Orchestrator(
        object_store=object_store,
        document_store=document_store,
        ner_stage=ner_stage,
        coref_stage=coref_stage,
        entity_linking_stage=entity_linking_stage,
        graph_store=graph_store,
        kg_build_stage=kg_build_stage,
    )

    # --- Prose synthesizer (N17, V7): the OPTIONAL gated LLM answer mode -----
    #     Pinned to its own Settings.synthesis_model (B6 — a fuller model reserved
    #     for synthesis), sharing the same LLM response cache/retry/key. Wired onto
    #     the retriever but GATED OFF: it runs only when a request sets
    #     synthesize=true (ADR-0009). The default /query path never calls it.
    synthesizer = AnswerSynthesizer.from_settings(settings)

    # --- Query retriever (N16, V6): the synchronous /query read path --------
    #     Reuses the SAME embedder + entity/document/graph stores built for
    #     ingestion so query-side kNN + k-hop read exactly what was written. No
    #     LLM in the default path (ADR-0007) — deterministic and $0; the optional
    #     V7 synthesizer above only fires on synthesize=true.
    retriever = QueryRetriever.from_settings(
        settings,
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
        synthesizer=synthesizer,
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

    # --- FastAPI app (POST /ingest publishes; POST /query retrieves) --------
    app = create_app(object_store, publisher, settings, retriever=retriever)

    logger.info("serving FastAPI app on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 — local demo, bind all.


if __name__ == "__main__":
    main()
