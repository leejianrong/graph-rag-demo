# ruff: noqa: E501
"""Run the existing port contract tests against the Postgres stores.

The ES/Neo4j/MinIO contract modules in ``tests/contract`` take a ``store`` (or ``es_document_store``,
``object_store``) fixture. This module re-exports every test of those modules that needs only that
one fixture and supplies the fixture from Postgres, so the same assertions run unchanged. Tests
that reach into the backend (Neo4j labels via ``driver``, raw ES writes via ``es_client``) cannot
run here; ``test_what_was_not_run`` lists them.

    PG_DSN=postgresql://postgres:spike@127.0.0.1:55432/spike \\
      uv run --with "psycopg[binary]" pytest spikes/pg_stores -p no:cacheprovider
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from graph_rag.fakes import InMemoryEntityStore
from spikes.pg_stores.pg_stores import (
    PgDocumentStore,
    PgEntityStore,
    PgGraphStore,
    PgObjectStore,
    ensure_schema,
)

_SOURCES = {
    "tests.contract.test_entity_store_contract": {"store"},
    "tests.contract.test_entity_search_contract": {"entity_stores"},
    "tests.contract.test_graph_store_contract": {"store"},
    "tests.contract.test_document_store_contract": {"es_document_store"},
    "tests.contract.test_object_store_contract": {"object_store"},
}


def _conn(dsn: str, dims: int) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn, autocommit=True)
    ensure_schema(conn, dims, reset=True)
    yield conn
    conn.close()


@pytest.fixture()
def store(request: pytest.FixtureRequest, pg_dsn: str) -> Iterator[Any]:
    """An entity store (4-dim vectors, as the ES contract) or a graph store, by test module."""
    for conn in _conn(pg_dsn, 4):
        entity = "entity" in request.function.__module__
        yield PgEntityStore(conn) if entity else PgGraphStore(conn)


@pytest.fixture()
def entity_stores(pg_dsn: str) -> Iterator[tuple[PgEntityStore, InMemoryEntityStore]]:
    """The Postgres store and the fake, each with the same ticking clock."""
    from tests.contract.test_entity_search_contract import _ticking_clock

    for conn in _conn(pg_dsn, 4):
        yield (
            PgEntityStore(conn, clock=_ticking_clock()),
            InMemoryEntityStore(clock=_ticking_clock()),
        )


@pytest.fixture()
def es_document_store(pg_dsn: str) -> Iterator[PgDocumentStore]:
    for conn in _conn(pg_dsn, 384):
        yield PgDocumentStore(conn)


@pytest.fixture()
def object_store(pg_dsn: str) -> Iterator[PgObjectStore]:
    for conn in _conn(pg_dsn, 4):
        yield PgObjectStore(conn)


_not_run: list[str] = []


def _install() -> None:
    for module_name, allowed in _SOURCES.items():
        try:
            module = importlib.import_module(module_name)
        except BaseException:  # noqa: BLE001 - a module may skip at import (no testcontainers)
            continue
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != module_name:
                continue
            params = set(inspect.signature(fn).parameters)
            backend_only = (
                "before_scopes_existed" in name
                or "index_made_before" in name
                or "reserved_key" in name
            )
            if params <= allowed and not backend_only:
                globals()[f"{name}__{module_name.rsplit('.', 1)[1]}"] = fn
            else:
                _not_run.append(f"{module_name.rsplit('.', 1)[1]}::{name} needs {sorted(params)}")


_install()


def test_what_was_not_run() -> None:
    """Print the contract tests that are tied to a backend and so were not reused."""
    print("\nnot reused:", *_not_run, sep="\n  ")
