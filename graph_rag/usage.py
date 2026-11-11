"""Per-tenant usage: documents ingested, queries answered and LLM tokens spent (GRA-58).

Counts live behind :class:`UsageStore`; :class:`JsonFileUsageStore` is one file that survives a
restart, fine for a single service instance like the API key file (``graph_rag/auth.py``).
Metering is off unless ``METER_USAGE`` is on; off, nothing is recorded or written. Operators read
the totals with ``graph-rag-admin usage``; there is no tenant-facing endpoint, so no key can read
another tenant's counts. This is metering, not invoicing.

What is counted, and when:

* **documents**: a document the API accepted and queued (``/ingest``, ``/v1/remember``). An
  ``unchanged`` re-ingest is not counted, since nothing runs for it. ``force=true`` is a new
  ingest and counts. It is counted at the door, so a Kafka redelivery or a stage retry never adds
  to it, and a document that later fails still counts.
* **queries**: a successful ``/query`` or ``/v1/recall``, whether or not it synthesised prose.
* **llm_tokens**: tokens the provider reported for a call, kept in ``tenant``'s name by the
  acting tenant (``graph_rag/scope.py``). A response-cache hit makes no call and costs none. A
  retry after a malformed answer is a second provider call and does spend tokens, so it counts.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

try:
    import fcntl
except ImportError:  # not on Windows: there, only one process may write the file
    fcntl = None  # type: ignore[assignment]

from graph_rag.scope import DEFAULT_TENANT

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response

    from graph_rag.config import Settings

__all__ = [
    "InMemoryUsageStore",
    "JsonFileUsageStore",
    "Usage",
    "UsageMeter",
    "UsageStore",
    "add_query_metering",
]

# Every file-backed store in the process shares this, so two components writing one file
# (the API and the LLM client) cannot interleave a read-modify-write. Another process (the ingest
# worker, GRA-31) is kept out by an ``flock`` on a sidecar file, taken inside this lock.
_FILE_LOCK = threading.Lock()
_QUERY_PATHS = frozenset({"/query", "/v1/recall"})


@dataclass(frozen=True)
class Usage:
    """What one tenant has used since metering began."""

    documents: int = 0
    queries: int = 0
    llm_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.documents + other.documents,
            self.queries + other.queries,
            self.llm_tokens + other.llm_tokens,
        )


class UsageStore(Protocol):
    """Where per-tenant totals are kept."""

    def add(self, tenant: str, delta: Usage) -> None: ...

    def get(self, tenant: str) -> Usage: ...

    def all(self) -> dict[str, Usage]: ...


class InMemoryUsageStore:
    """A store for tests; the totals go with the process."""

    def __init__(self) -> None:
        self._totals: dict[str, Usage] = {}
        self._lock = threading.Lock()

    def add(self, tenant: str, delta: Usage) -> None:
        with self._lock:
            self._totals[tenant] = self._totals.get(tenant, Usage()) + delta

    def get(self, tenant: str) -> Usage:
        return self._totals.get(tenant, Usage())

    def all(self) -> dict[str, Usage]:
        return dict(self._totals)


class JsonFileUsageStore:
    """Totals in one JSON file, written atomically on every change and read on every call.

    Safe for several processes on one file (the API and ingest workers sharing a volume): each
    change is a read-modify-write under a file lock.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with _FILE_LOCK:
            if fcntl is None:
                yield
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(f"{self._path}.lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Usage]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        return {tenant: Usage(**fields) for tenant, fields in raw.items()}

    def add(self, tenant: str, delta: Usage) -> None:
        with self._locked():
            totals = self._read()
            totals[tenant] = totals.get(tenant, Usage()) + delta
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self._path.parent, prefix=".usage-")
            with os.fdopen(fd, "w") as handle:
                json.dump({t: asdict(u) for t, u in totals.items()}, handle, indent=2)
            os.replace(temp, self._path)

    def get(self, tenant: str) -> Usage:
        return self.all().get(tenant, Usage())

    def all(self) -> dict[str, Usage]:
        with self._locked():
            return self._read()


class UsageMeter:
    """Record usage for the tenant a call acts for."""

    def __init__(self, store: UsageStore) -> None:
        self._store = store

    @classmethod
    def from_settings(cls, settings: Settings) -> UsageMeter | None:
        """The meter the service writes to, or ``None`` when ``meter_usage`` is off."""
        return cls(JsonFileUsageStore(settings.usage_path)) if settings.meter_usage else None

    def document(self, tenant: str) -> None:
        self._store.add(tenant, Usage(documents=1))

    def query(self, tenant: str) -> None:
        self._store.add(tenant, Usage(queries=1))

    def llm_tokens(self, tenant: str, tokens: int) -> None:
        if tokens > 0:
            self._store.add(tenant, Usage(llm_tokens=tokens))

    def totals(self, tenant: str | None = None) -> dict[str, Usage]:
        """Totals for ``tenant``, or for every tenant that has used anything."""
        if tenant is not None:
            return {tenant: self._store.get(tenant)}
        return self._store.all()


def add_query_metering(app: FastAPI, meter: UsageMeter) -> None:
    """Count each successful query. ``tenant_of`` leaves the tenant on ``request.state``."""

    @app.middleware("http")
    async def count_queries(request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        response: Response = await call_next(request)
        if (
            request.method == "POST"
            and request.url.path in _QUERY_PATHS
            and response.status_code < 300
        ):
            meter.query(getattr(request.state, "tenant", DEFAULT_TENANT))
        return response
