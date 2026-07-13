"""Contract test: real ``EsDocumentStore`` behaves like ``InMemoryDocumentStore``.

Per TESTING §3, the contract layer proves each real adapter behaves like its fake
against a real service (here Elasticsearch via testcontainers). It gates the
adapter, not pipeline logic, so it is marked ``contract`` and excluded from the
fast suite. Skips cleanly when Docker is unavailable.

Asserts the ``DocumentStore`` contract on the real adapter:

* ``upsert`` then ``get`` returns an equal record;
* ``get`` of a missing id returns ``None``;
* re-``upsert`` with the same ``document_id`` overwrites (one doc, updated text) —
  the R1.5 idempotency guarantee proven at real Elasticsearch (ADR-0001).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graph_rag.ids import document_id
from graph_rag.models import DocumentRecord

if TYPE_CHECKING:
    from collections.abc import Iterator

    from graph_rag.adapters.es_document_store import EsDocumentStore

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_INDEX = "documents"


@pytest.fixture(scope="module")
def es_document_store() -> Iterator[EsDocumentStore]:
    """A real :class:`EsDocumentStore` over a throwaway Elasticsearch container.

    Skips the whole module if Docker / testcontainers is unavailable.
    """
    try:
        from elasticsearch import Elasticsearch
        from testcontainers.elasticsearch import ElasticSearchContainer

        from graph_rag.adapters.es_document_store import EsDocumentStore
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
        store = EsDocumentStore(client=client, index=_INDEX)
        store.ensure_index()
        yield store
    finally:
        container.stop()


def _record(text: str, key: str = "a.md", bucket: str = "documents") -> DocumentRecord:
    """Build a raw V1 ``DocumentRecord`` for ``(bucket, key)`` with the given text."""
    return DocumentRecord(
        document_id=document_id(bucket, key),
        bucket=bucket,
        object_key=key,
        text=text,
    )


def test_upsert_then_get_returns_equal_record(es_document_store: EsDocumentStore) -> None:
    """A record round-trips: get after upsert returns an equal record."""
    record = _record("hello real elasticsearch", key="roundtrip.md")
    es_document_store.upsert(record)

    fetched = es_document_store.get(record.document_id)
    assert fetched == record


def test_get_missing_returns_none(es_document_store: EsDocumentStore) -> None:
    """get of an id that was never written returns None."""
    assert es_document_store.get(document_id("documents", "never-written.md")) is None


def test_reupsert_overwrites_single_doc(es_document_store: EsDocumentStore) -> None:
    """Re-upserting the same document_id overwrites in place (one doc, new text)."""
    first = _record("first version", key="overwrite.md")
    es_document_store.upsert(first)

    second = _record("second version", key="overwrite.md")
    es_document_store.upsert(second)

    # Same deterministic id -> overwrite, not duplicate.
    assert first.document_id == second.document_id
    fetched = es_document_store.get(second.document_id)
    assert fetched is not None
    assert fetched.text == "second version"
