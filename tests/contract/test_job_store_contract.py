"""Contract test: real ``EsJobStore`` behaves like ``InMemoryJobStore`` (GRA-31).

Two store instances over one Elasticsearch stand for the API process and a worker process.
Marked ``contract``; skips cleanly when Docker is unavailable.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from graph_rag.jobs import InMemoryJobStore, JobStore
from graph_rag.scope import DEFAULT_SCOPE, Scope

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"


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


def check_job_store(writer: JobStore, reader: JobStore) -> None:
    """What the API (``reader``) must see of what a worker (``writer``) records."""
    scope, other = Scope("acme"), Scope("globex")
    assert reader.get(scope, "d1") is None
    writer.queued(scope, "d1", "a.txt")
    queued = reader.get(scope, "d1")
    assert queued is not None and queued.state == "queued" and queued.stages == []
    writer.update(scope, "d1", "running", stage="read")
    writer.update(scope, "d1", "running", stage="coreference")
    writer.update(scope, "d1", "failed", "coreference failed: RuntimeError")
    job = reader.get(scope, "d1")
    assert job is not None and job.state == "failed" and job.object_key == "a.txt"
    assert job.stages == ["read", "coreference"] and job.error == "coreference failed: RuntimeError"
    assert reader.get(other, "d1") is None and reader.get(Scope("acme", "ns"), "d1") is None
    writer.update(DEFAULT_SCOPE, "unknown", "done")  # a trigger that skipped the API
    unknown = reader.get(DEFAULT_SCOPE, "unknown")
    assert unknown is not None and unknown.state == "done" and unknown.object_key == ""
    writer.queued(scope, "d1", "a.txt")  # resubmitted: starts over
    again = reader.get(scope, "d1")
    assert again is not None and again.state == "queued" and again.stages == []


def test_the_in_memory_store_meets_the_contract() -> None:
    store = InMemoryJobStore()
    check_job_store(store, store)


def test_the_es_store_meets_the_contract_across_two_instances(es_url: str) -> None:
    from elasticsearch import Elasticsearch

    from graph_rag.adapters.es_job_store import EsJobStore

    api, worker = (EsJobStore(Elasticsearch(hosts=[es_url]), "jobs") for _ in range(2))
    api.ensure_index()
    check_job_store(worker, api)
