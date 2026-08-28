"""Contract test: the real ``EvalBenchExportStage`` writes the eval-bench shape.

Per TESTING §3, the contract layer proves a real adapter behaves as designed
against real services (here Elasticsearch + Neo4j via testcontainers, mirroring
``tests/contract/test_document_store_contract.py`` and
``tests/contract/test_graph_store_contract.py``). Marked ``contract`` and
excluded from the fast suite; skips cleanly when Docker is unavailable.

Asserts the exact external contract the separate ``eval-bench`` project's
existing client relies on (see ``EvalBenchExportStage``'s module docstring):

* the ES document lands at ``_id = sha256(url)`` and is findable by a bare
  ``query_string`` query with NO explicit ``fields:`` list;
* Neo4j gets a ``:Document {doc_id}`` node (``doc_id = sha256(url)``), one
  ``:Entity {name, description}`` node per mentioned entity, and a ``MENTIONS``
  edge from the document to each;
* the ``:Entity`` node is the SAME node this project's own
  ``Neo4jGraphStore.upsert_entities`` created (joined by ``canonical_id``, no
  second/parallel identity) — merging in ``name``/``description`` rather than
  creating a duplicate.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from graph_rag.models import CanonicalEntity, DocumentRecord, EntityLink, Mention, Sentence

if TYPE_CHECKING:
    from collections.abc import Iterator

    from elasticsearch import Elasticsearch
    from neo4j import Driver

    from graph_rag.adapters.eval_bench_export import EvalBenchExportStage

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_NEO4J_IMAGE = "neo4j:5.20"
_INDEX = "osint_reports"

URL = "https://example.com/news/acme-in-london"
DOC_ID = hashlib.sha256(URL.encode("utf-8")).hexdigest()

SENT0 = "Ada Lovelace works for Acme Corp."
SENT1 = "Acme Corp is based in London."
TEXT = f"{SENT0} {SENT1}"

ADA_ID = "p:ada"
ACME_ID = "o:acme"


@pytest.fixture(scope="module")
def es_client() -> Iterator[Elasticsearch]:
    """A real Elasticsearch client over a throwaway container (module-scoped)."""
    try:
        from elasticsearch import Elasticsearch
        from testcontainers.elasticsearch import ElasticSearchContainer
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"testcontainers/elasticsearch not importable: {exc}")

    try:
        container = ElasticSearchContainer(_ES_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001 - Docker not available / cannot pull image.
        pytest.skip(f"Docker/Elasticsearch container unavailable: {exc}")

    try:
        url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        client = Elasticsearch(hosts=[url])
        yield client
        client.close()
    finally:
        container.stop()


@pytest.fixture(scope="module")
def neo4j_driver() -> Iterator[Driver]:
    """A real Neo4j driver over a throwaway container (module-scoped)."""
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


@pytest.fixture()
def stage(es_client: Elasticsearch, neo4j_driver: Driver) -> EvalBenchExportStage:
    """A fresh :class:`EvalBenchExportStage` over an empty index + graph per test."""
    from graph_rag.adapters.eval_bench_export import EvalBenchExportStage
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore

    if es_client.indices.exists(index=_INDEX):
        es_client.indices.delete(index=_INDEX)
    neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")

    # Seed the pre-existing :Entity nodes this project's own graph checkpoint
    # would have created, so the test can prove the export MERGES onto them
    # (join by canonical_id) rather than minting a second identity.
    graph_store = Neo4jGraphStore(driver=neo4j_driver)
    graph_store.init()
    graph_store.upsert_entities(
        [
            CanonicalEntity(canonical_id=ADA_ID, name="Ada Lovelace", type="PERSON"),
            CanonicalEntity(canonical_id=ACME_ID, name="Acme Corp", type="ORG"),
        ]
    )

    export_stage = EvalBenchExportStage(es_client, _INDEX, neo4j_driver)
    export_stage.ensure_index()
    return export_stage


def _record() -> DocumentRecord:
    """A fully-enriched record (as the orchestrator's hook sees it at V5)."""
    return DocumentRecord(
        document_id="internal-doc-id",
        bucket="documents",
        object_key="acme.md",
        text=TEXT,
        mentions=[
            Mention(text="Ada Lovelace", type="PERSON", char_start=0, char_end=12),
            Mention(text="Acme Corp", type="ORG", char_start=23, char_end=32),
        ],
        sentences=[
            Sentence(text=SENT0, char_start=0, char_end=len(SENT0), index=0),
            Sentence(text=SENT1, char_start=len(SENT0) + 1, char_end=len(TEXT), index=1),
        ],
        el_result=[
            EntityLink(
                mention_text="Ada Lovelace",
                canonical_id=ADA_ID,
                entity_type="PERSON",
                score=1.0,
                is_new=True,
            ),
            EntityLink(
                mention_text="Acme Corp",
                canonical_id=ACME_ID,
                entity_type="ORG",
                score=1.0,
                is_new=True,
            ),
        ],
    )


def _entities() -> list[CanonicalEntity]:
    return [
        CanonicalEntity(canonical_id=ADA_ID, name="Ada Lovelace", type="PERSON"),
        CanonicalEntity(canonical_id=ACME_ID, name="Acme Corp", type="ORG"),
    ]


def test_es_document_lands_at_sha256_url_and_is_default_field_searchable(
    stage: EvalBenchExportStage, es_client: Elasticsearch
) -> None:
    """The doc is indexed at ``_id = sha256(url)`` and matches a bare ``query_string``."""
    stage.write(_record(), _entities(), [], source_url=URL)

    got = es_client.get(index=_INDEX, id=DOC_ID)
    assert got["_source"]["url"] == URL
    assert got["_source"]["text"] == TEXT

    # The load-bearing external contract: NO explicit `fields:` list.
    response = es_client.search(index=_INDEX, query={"query_string": {"query": "Lovelace"}})
    hits = response["hits"]["hits"]
    assert len(hits) == 1
    assert hits[0]["_id"] == DOC_ID


def test_neo4j_document_entity_mentions_shape(
    stage: EvalBenchExportStage, neo4j_driver: Driver
) -> None:
    """A ``:Document`` node, ``MENTIONS`` edges and ``:Entity.description`` all appear."""
    stage.write(_record(), _entities(), [], source_url=URL)

    doc_rows = neo4j_driver.execute_query(
        "MATCH (d:Document {doc_id: $doc_id}) RETURN d.url AS url", doc_id=DOC_ID
    ).records
    assert len(doc_rows) == 1
    assert doc_rows[0]["url"] == URL

    mention_rows = neo4j_driver.execute_query(
        "MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity) "
        "RETURN e.canonical_id AS cid, e.name AS name, e.description AS description "
        "ORDER BY e.canonical_id",
        doc_id=DOC_ID,
    ).records
    assert {r["cid"] for r in mention_rows} == {ADA_ID, ACME_ID}
    by_id = {r["cid"]: r for r in mention_rows}
    assert by_id[ADA_ID]["name"] == "Ada Lovelace"
    assert by_id[ADA_ID]["description"] == SENT0  # "first sentence mentioned in"
    assert by_id[ACME_ID]["description"] == SENT0  # Acme is also first mentioned in sentence 0


def test_entity_node_is_the_same_existing_node_not_a_duplicate(
    stage: EvalBenchExportStage, neo4j_driver: Driver
) -> None:
    """The export MERGEs onto the pre-existing ``:Entity {canonical_id}`` node.

    No second/parallel identity: still exactly one ``:Entity`` node per
    canonical_id after the export write, and it still carries the ``Person``/
    ``Organization`` type label the real graph checkpoint set.
    """
    stage.write(_record(), _entities(), [], source_url=URL)

    ada_nodes = neo4j_driver.execute_query(
        "MATCH (e:Entity {canonical_id: $cid}) RETURN labels(e) AS labels, e.type AS type",
        cid=ADA_ID,
    ).records
    assert len(ada_nodes) == 1  # exactly one node — merged, not duplicated
    assert "Person" in ada_nodes[0]["labels"]  # the graph checkpoint's own label survives
    assert ada_nodes[0]["type"] == "PERSON"


def test_write_skipped_without_source_url(
    stage: EvalBenchExportStage, es_client: Elasticsearch, neo4j_driver: Driver
) -> None:
    """No ``source_url`` -> the write is skipped (logged), not raised."""
    stage.write(_record(), _entities(), [], source_url=None)  # must not raise

    assert es_client.count(index=_INDEX)["count"] == 0
    doc_rows = neo4j_driver.execute_query("MATCH (d:Document) RETURN count(d) AS c").records
    assert doc_rows[0]["c"] == 0
