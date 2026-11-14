# ruff: noqa: E501
"""Spike fixtures: a Postgres DSN from ``PG_DSN``; every test skips when it is unset."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg")


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        pytest.skip("PG_DSN not set (start the grsmoke6-pg container, see README.md here)")
    return dsn
