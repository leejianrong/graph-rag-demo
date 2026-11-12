"""Contract test: entity search, first_seen and entity_sources on Elasticsearch match the fakes.

The explorer endpoints (GRA-142) rest on three store methods: ``EntityStore.search`` and
``first_seen`` and ``DocumentStore.entity_sources``. Each is loaded with the same records as its
in-memory fake and asked the same questions, and the answers must be equal. Marked ``contract``
(needs Docker); skips cleanly without it.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pytest

from graph_rag.fakes import InMemoryDocumentStore, InMemoryEntityStore
from graph_rag.models import CanonicalEntity, DocumentRecord, EntityLink
from graph_rag.scope import DEFAULT_SCOPE, Scope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_DIMS = 4


@pytest.fixture(scope="module")
def es_client() -> Iterator[object]:
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
        yield Elasticsearch(hosts=[url])
    finally:
        container.stop()


def _ticking_clock() -> Callable[[], str]:
    """A clock that returns a later time on each call, so "newest first" is unambiguous."""
    ticks = itertools.count(1)
    return lambda: f"2026-01-01T00:00:{next(ticks):02d}.000000Z"


def _entity(cid: str, name: str, type_: str = "ORG", aliases: list[str] | None = None):  # type: ignore[no-untyped-def]
    return CanonicalEntity(canonical_id=cid, name=name, type=type_, aliases=aliases or [])  # type: ignore[arg-type]


@pytest.fixture()
def entity_stores(es_client: object) -> tuple[EsEntityStore, InMemoryEntityStore]:
    from graph_rag.adapters.es_entity_store import EsEntityStore

    es_client.indices.delete(index="search-entities", ignore_unavailable=True)  # type: ignore[attr-defined]
    real = EsEntityStore(
        client=es_client,  # type: ignore[arg-type]
        index="search-entities",
        dims=_DIMS,
        clock=_ticking_clock(),
    )
    real.ensure_index()
    return real, InMemoryEntityStore(clock=_ticking_clock())


def test_search_and_first_seen_behave_like_the_fake(
    entity_stores: tuple[EsEntityStore, InMemoryEntityStore],
) -> None:
    real, fake = entity_stores
    entities = [
        _entity("e:1", "Acme Corp", aliases=["ACME"]),
        _entity("e:2", "Globex", aliases=["Acme Holdings"]),
        _entity("e:3", "Acme Labs"),
        _entity("e:4", "Ada Lovelace", "PERSON"),
        _entity("e:5", "Acme-Town", "LOCATION"),
    ]
    for entity in entities:
        real.upsert(DEFAULT_SCOPE, entity)
        fake.upsert(DEFAULT_SCOPE, entity)
    # A new alias on an existing entity must not make it new again.
    grown = _entity("e:1", "Acme Corp", aliases=["ACME", "Acme Inc"])
    real.upsert(DEFAULT_SCOPE, grown)
    fake.upsert(DEFAULT_SCOPE, grown)

    def ids(rows: list[tuple[CanonicalEntity, str]]) -> list[str]:
        return [e.canonical_id for e, _ in rows]

    for query, kind in (("acme", None), ("ACME,", None), ("holdings", None), ("acme", "ORG")):
        ask = {"query": query, "entity_type": kind, "limit": 10}
        assert real.search(DEFAULT_SCOPE, **ask) == fake.search(DEFAULT_SCOPE, **ask), ask  # type: ignore[arg-type]
    assert ids(real.search(DEFAULT_SCOPE, query="acme", limit=10)) == ["e:5", "e:3", "e:2", "e:1"]
    everything = ["e:5", "e:4", "e:3", "e:2", "e:1"]
    assert ids(real.search(DEFAULT_SCOPE, limit=10)) == everything
    assert ids(real.search(DEFAULT_SCOPE, query=" ,", limit=10)) == everything  # matches all

    first = real.search(DEFAULT_SCOPE, query="acme", limit=2)
    cursor = (first[-1][1], first[-1][0].canonical_id)
    second = real.search(DEFAULT_SCOPE, query="acme", limit=2, after=cursor)
    assert ids(first) + ids(second) == ["e:5", "e:3", "e:2", "e:1"]
    assert second == fake.search(DEFAULT_SCOPE, query="acme", limit=2, after=cursor)

    assert real.first_seen(DEFAULT_SCOPE, "e:1") == fake.first_seen(DEFAULT_SCOPE, "e:1")
    assert real.first_seen(DEFAULT_SCOPE, "e:1") == "2026-01-01T00:00:01.000000Z"
    assert real.first_seen(DEFAULT_SCOPE, "e:nope") is None
    assert real.search(Scope("someone-else"), query="acme", limit=10) == []
    assert real.first_seen(Scope("someone-else"), "e:1") is None


def _linked(key: str, *canonical_ids: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=f"doc-{key}",
        bucket="documents",
        object_key=key,
        text=f"text of {key}",
        el_result=[
            EntityLink(
                mention_text=cid, canonical_id=cid, entity_type="ORG", score=1.0, is_new=True
            )
            for cid in canonical_ids
        ],
    )


def test_entity_sources_behave_like_the_fake(es_client: object) -> None:
    from graph_rag.adapters.es_document_store import EsDocumentStore

    es_client.indices.delete(index="search-documents", ignore_unavailable=True)  # type: ignore[attr-defined]
    real: EsDocumentStore = EsDocumentStore(client=es_client, index="search-documents")  # type: ignore[arg-type]
    real.ensure_index()
    fake = InMemoryDocumentStore()
    records = [
        _linked("once.md", "e-src", "e-other"),
        _linked("twice.md", "e-src", "e-src"),
        _linked("none.md", "e-other"),
    ]
    for record in records:
        real.upsert(DEFAULT_SCOPE, record)
        fake.upsert(DEFAULT_SCOPE, record)

    found = real.entity_sources(DEFAULT_SCOPE, "e-src")
    assert found == fake.entity_sources(DEFAULT_SCOPE, "e-src")
    assert [(s.name, s.mentions) for s in found] == [("twice.md", 2), ("once.md", 1)]
    assert real.entity_sources(DEFAULT_SCOPE, "e-missing") == []
    assert real.entity_sources(Scope("someone-else"), "e-src") == []
