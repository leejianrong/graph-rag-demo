"""Tenant isolation through the whole stack on real services (GRA-24, ADR-0015).

The HTTP API with auth on, over real MinIO, Elasticsearch and Neo4j, with the offline pipeline
stages. Tenant B tries to recall, query, forget and delete tenant A's memory, and every attempt
must find nothing or a 404. The same attempts run on the fakes in the fast suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.isolation import check_tenants_cannot_reach_each_other
from tests.unit.test_agent_api_v1 import Queue

pytestmark = pytest.mark.contract

try:
    from elasticsearch import Elasticsearch
    from fastapi.testclient import TestClient
    from testcontainers.elasticsearch import ElasticSearchContainer
    from testcontainers.minio import MinioContainer
    from testcontainers.neo4j import Neo4jContainer

    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore
    from graph_rag.adapters.minio_object_store import MinioObjectStore
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"contract deps unavailable: {exc}", allow_module_level=True)

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_NEO4J_IMAGE = "neo4j:5.20"
_MINIO_IMAGE = "chainguard/minio:latest"  # see test_object_store_contract.py


@pytest.fixture(scope="module")
def stores() -> Iterator[tuple]:
    containers = []
    try:
        es = ElasticSearchContainer(_ES_IMAGE)
        neo = Neo4jContainer(_NEO4J_IMAGE)
        minio = MinioContainer(_MINIO_IMAGE)
        for container in (es, neo, minio):
            container.start()
            containers.append(container)
    except Exception as exc:  # noqa: BLE001 - Docker unavailable or an image cannot be pulled.
        for container in containers:
            container.stop()
        pytest.skip(f"Docker services unavailable: {exc}")

    try:
        settings = Settings()
        client = Elasticsearch(
            hosts=[f"http://{es.get_container_host_ip()}:{es.get_exposed_port(es.port)}"]
        )
        documents = EsDocumentStore(client=client, index="tenancy-documents")
        documents.ensure_index()
        entities = EsEntityStore(client=client, index="tenancy-entities", dims=settings.embed_dim)
        entities.ensure_index()
        driver = neo.get_driver()
        graph = Neo4jGraphStore(driver=driver)
        graph.init()
        config = minio.get_config()
        objects = MinioObjectStore(
            endpoint=config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=False,
        )
        yield objects, documents, entities, graph
        driver.close()
    finally:
        for container in containers:
            container.stop()


def test_tenant_b_cannot_reach_tenant_a_through_the_api_on_real_services(stores) -> None:
    components = build_offline_components(stores=stores)
    queue = Queue()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        queue,
        Settings(require_auth=True),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        keys=keys,
    )

    def drain() -> None:
        for trigger in queue.triggers:
            assert components.orchestrator.process_document(trigger)
        queue.triggers.clear()
        components.document_store._client.indices.refresh(index="tenancy-documents")  # noqa: SLF001
        components.entity_store._client.indices.refresh(index="tenancy-entities")  # noqa: SLF001

    check_tenants_cannot_reach_each_other(TestClient(app), keys, drain)
