"""Contract test: real ``EsEntityStore`` behaves like ``InMemoryEntityStore``.

Per TESTING §3, the contract layer proves each real adapter behaves like its fake
against a real service (here Elasticsearch via testcontainers). It gates the
adapter, not pipeline logic, so it is marked ``contract`` and excluded from the
fast suite. Skips cleanly when Docker is unavailable.

Asserts the ``EntityStore`` contract (ADR-0004/0005) on the real adapter, mirrored
against :class:`~graph_rag.fakes.InMemoryEntityStore` loaded with the same
entities:

* ``upsert`` then ``get`` returns an equal entity;
* re-``upsert`` with the same ``canonical_id`` overwrites (``count`` stays 1);
* ``block_candidates`` returns only entities matching ``type`` AND the normalized
  name-or-alias key, using the shared ``normalize_name`` rule;
* ``knn`` ranks by cosine descending, in the same order as the fake;
* type-filtered ``knn`` restricts the candidate set.

Vectors are tiny hand-crafted unit vectors (``dims=4``) so the cosine ordering is
obvious and the test is fast — the dimension is adapter config, so the contract
holds regardless of the production 384-dim ``bge`` model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graph_rag.fakes import InMemoryEntityStore
from graph_rag.models import CanonicalEntity
from graph_rag.normalize import normalize_name

if TYPE_CHECKING:
    from collections.abc import Iterator

    from graph_rag.adapters.es_entity_store import EsEntityStore

pytestmark = pytest.mark.contract

_ES_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
_INDEX = "entities"
_DIMS = 4


@pytest.fixture(scope="module")
def es_client() -> Iterator[object]:
    """A real Elasticsearch client over a throwaway container (module-scoped).

    Skips the whole module if Docker / testcontainers is unavailable.
    """
    try:
        from elasticsearch import Elasticsearch
        from testcontainers.elasticsearch import ElasticSearchContainer

        import graph_rag.adapters.es_entity_store  # noqa: F401 - import hygiene guard
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


@pytest.fixture()
def store(es_client: object) -> EsEntityStore:
    """A fresh :class:`EsEntityStore` over a clean index for each test."""
    from graph_rag.adapters.es_entity_store import EsEntityStore

    # Delete any index left by a prior test so each test starts empty.
    es_client.indices.delete(index=_INDEX, ignore_unavailable=True)  # type: ignore[attr-defined]
    entity_store = EsEntityStore(client=es_client, index=_INDEX, dims=_DIMS)  # type: ignore[arg-type]
    entity_store.ensure_index()
    return entity_store


def _entity(
    canonical_id: str,
    name: str,
    type_: str = "ORG",
    aliases: list[str] | None = None,
    vector: list[float] | None = None,
) -> CanonicalEntity:
    """Build a ``CanonicalEntity`` for the given fields."""
    return CanonicalEntity(
        canonical_id=canonical_id,
        name=name,
        type=type_,  # type: ignore[arg-type]
        aliases=aliases or [],
        vector=vector,
    )


def test_upsert_then_get_returns_equal_entity(store: EsEntityStore) -> None:
    """An entity round-trips: get after upsert returns an equal entity (like the fake)."""
    entity = _entity("org:apple", "Apple, Inc.", aliases=["Apple"], vector=[1.0, 0.0, 0.0, 0.0])
    store.upsert(entity)

    fake = InMemoryEntityStore()
    fake.upsert(entity)

    assert store.get("org:apple") == entity == fake.get("org:apple")
    assert store.get("org:missing") is None
    assert fake.get("org:missing") is None


def test_reupsert_overwrites_single_entity(store: EsEntityStore) -> None:
    """Re-upserting the same canonical_id overwrites in place (count stays 1)."""
    store.upsert(_entity("org:apple", "Apple", aliases=[]))
    store.upsert(_entity("org:apple", "Apple", aliases=["Apple Inc"]))

    assert store.count() == 1
    fetched = store.get("org:apple")
    assert fetched is not None
    assert fetched.aliases == ["Apple Inc"]


def test_block_candidates_matches_type_and_name_or_alias(store: EsEntityStore) -> None:
    """Blocking returns only entities of the right type whose name/alias normalizes to the key."""
    apple = _entity("org:apple", "Apple, Inc.", type_="ORG", aliases=["Apple"])
    apple_person = _entity("person:apple", "Apple", type_="PERSON")
    google = _entity("org:google", "Google LLC", type_="ORG")
    for entity in (apple, apple_person, google):
        store.upsert(entity)

    fake = InMemoryEntityStore()
    for entity in (apple, apple_person, google):
        fake.upsert(entity)

    # Match on the name's normalized form.
    key = normalize_name("apple inc")
    hits = store.block_candidates(entity_type="ORG", normalized_name=key)
    assert [e.canonical_id for e in hits] == ["org:apple"]
    fake_hits = fake.block_candidates(entity_type="ORG", normalized_name=key)
    assert {e.canonical_id for e in fake_hits} == {"org:apple"}

    # Match on an alias's normalized form (Apple -> apple), still type-scoped to ORG.
    alias_key = normalize_name("Apple")
    org_hits = store.block_candidates(entity_type="ORG", normalized_name=alias_key)
    assert [e.canonical_id for e in org_hits] == ["org:apple"]

    # Same key, PERSON type -> only the person entity (blocking is type-scoped).
    person_hits = store.block_candidates(entity_type="PERSON", normalized_name=alias_key)
    assert [e.canonical_id for e in person_hits] == ["person:apple"]

    # No match for an unknown key.
    assert store.block_candidates(entity_type="ORG", normalized_name=normalize_name("nope")) == []


def test_knn_ranks_by_cosine_like_the_fake(store: EsEntityStore) -> None:
    """kNN returns nearest-by-cosine descending, in the same order as the fake."""
    a = _entity("e:a", "A", vector=[1.0, 0.0, 0.0, 0.0])
    b = _entity("e:b", "B", vector=[0.0, 1.0, 0.0, 0.0])
    c = _entity("e:c", "C", vector=[0.0, 0.0, 1.0, 0.0])
    novec = _entity("e:novec", "NoVec", vector=None)  # never ranked (no vector)

    fake = InMemoryEntityStore()
    for entity in (a, b, c, novec):
        store.upsert(entity)
        fake.upsert(entity)

    query = [0.9, 0.1, 0.0, 0.0]  # closest to A, then B, then C
    real = store.knn(vector=query, top_k=3)
    ranked = InMemoryEntityStore.knn(fake, vector=query, top_k=3)

    real_order = [e.canonical_id for e, _ in real]
    fake_order = [e.canonical_id for e, _ in ranked]
    assert real_order == fake_order == ["e:a", "e:b", "e:c"]

    # Scores match the fake's raw cosine (ES (1+cos)/2 converted back) within FP tol.
    for (re_ent, re_score), (fk_ent, fk_score) in zip(real, ranked, strict=True):
        assert re_ent.canonical_id == fk_ent.canonical_id
        assert re_score == pytest.approx(fk_score, abs=1e-5)

    # The vector-less entity is never a candidate.
    assert "e:novec" not in real_order


def test_knn_type_filter_restricts_candidates(store: EsEntityStore) -> None:
    """A type-filtered kNN only ranks entities of that type."""
    org = _entity("org:x", "X", type_="ORG", vector=[1.0, 0.0, 0.0, 0.0])
    person = _entity("person:y", "Y", type_="PERSON", vector=[0.99, 0.14, 0.0, 0.0])
    store.upsert(org)
    store.upsert(person)

    query = [1.0, 0.0, 0.0, 0.0]
    person_hits = store.knn(vector=query, entity_type="PERSON", top_k=5)
    assert [e.canonical_id for e, _ in person_hits] == ["person:y"]

    org_hits = store.knn(vector=query, entity_type="ORG", top_k=5)
    assert [e.canonical_id for e, _ in org_hits] == ["org:x"]
