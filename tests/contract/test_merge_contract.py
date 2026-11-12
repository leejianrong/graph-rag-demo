"""Contract test: real merge-record and merge behaviour matches the fakes (GRA-144).

``EsMergeRecordStore`` meets the ``MergeRecordStore`` port, and ``EntityCurator`` over real
Elasticsearch (entities, documents, merge records) and Neo4j (graph) merges, records and heals a
half-done merge exactly as it does over the in-memory fakes (``tests/merge_checks.py``).
Marked ``contract``; skips cleanly when Docker is unavailable.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from graph_rag.config import Settings
from tests.merge_checks import check_entity_merge, check_entity_split, check_merge_store

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_NEO4J_IMAGE = "neo4j:5.20"


@pytest.fixture(scope="module")
def es_url() -> Iterator[str]:
    try:
        from testcontainers.elasticsearch import ElasticSearchContainer
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"testcontainers/elasticsearch not importable: {exc}")
    try:
        container = ElasticSearchContainer(_ES_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001 - Docker not available / cannot pull image.
        pytest.skip(f"Docker/Elasticsearch container unavailable: {exc}")
    try:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
    finally:
        container.stop()


@pytest.fixture(scope="module")
def neo4j_driver() -> Iterator[object]:
    try:
        from testcontainers.neo4j import Neo4jContainer
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"testcontainers/neo4j not importable: {exc}")
    try:
        container = Neo4jContainer(_NEO4J_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001 - Docker not available / cannot pull image.
        pytest.skip(f"Docker/Neo4j container unavailable: {exc}")
    try:
        driver = container.get_driver()
        yield driver
        driver.close()
    finally:
        container.stop()


def test_the_es_merge_record_store_meets_the_contract(es_url: str) -> None:
    from elasticsearch import Elasticsearch

    from graph_rag.adapters.es_merge_store import EsMergeRecordStore

    store = EsMergeRecordStore(Elasticsearch(hosts=[es_url]), "merge-records")
    store.ensure_index()
    check_merge_store(store)


def test_a_merge_over_real_elasticsearch_and_neo4j_matches_the_fakes(
    es_url: str, neo4j_driver: object
) -> None:
    from elasticsearch import Elasticsearch

    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore
    from graph_rag.adapters.es_merge_store import EsMergeRecordStore
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore

    client = Elasticsearch(hosts=[es_url])
    dims = Settings().embed_dim
    entities = EsEntityStore(client=client, index="merge-entities", dims=dims)
    entities.ensure_index()
    documents = EsDocumentStore(client=client, index="merge-documents")
    documents.ensure_index()
    merges = EsMergeRecordStore(client, "merge-records-flow")
    merges.ensure_index()
    graph = Neo4jGraphStore(driver=neo4j_driver)  # type: ignore[arg-type]
    graph.init()
    check_entity_merge(entities, graph, documents, merges, dims)


def test_a_split_over_real_elasticsearch_and_neo4j_matches_the_fakes(
    es_url: str, neo4j_driver: object
) -> None:
    from elasticsearch import Elasticsearch

    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore
    from graph_rag.adapters.es_merge_store import EsMergeRecordStore
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore

    client = Elasticsearch(hosts=[es_url])
    dims = Settings().embed_dim
    entities = EsEntityStore(client=client, index="split-entities", dims=dims)
    entities.ensure_index()
    documents = EsDocumentStore(client=client, index="split-documents")
    documents.ensure_index()
    merges = EsMergeRecordStore(client, "split-merge-records")
    merges.ensure_index()
    graph = Neo4jGraphStore(driver=neo4j_driver)  # type: ignore[arg-type]
    graph.init()
    check_entity_split(entities, graph, documents, merges, dims)
