"""Whose memory a call is about: a tenant and an optional namespace inside it (ADR-0015).

The tenant comes from the API key (GRA-22, 23) and the namespace from the request (GRA-45).
Until keys exist everything runs as :data:`DEFAULT_SCOPE`, whose ids are the ones minted before
scopes existed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = [
    "DEFAULT_SCOPE",
    "RESERVED_KEY_PREFIX",
    "Scope",
    "acting_for",
    "current_tenant",
    "tenant_dir",
]

DEFAULT_TENANT = "default"
# Object keys under this prefix hold non-default scopes' objects; no caller may name one directly.
RESERVED_KEY_PREFIX = "_scopes/"


@dataclass(frozen=True)
class Scope:
    """A tenant, and optionally a namespace that partitions memory inside it."""

    tenant: str = DEFAULT_TENANT
    namespace: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("tenant", self.tenant), ("namespace", self.namespace)):
            if value is not None and (not value or "/" in value or "|" in value):
                raise ValueError(f"{name} must be non-empty and contain no '/' or '|': {value!r}")

    @property
    def is_default(self) -> bool:
        return self == DEFAULT_SCOPE

    @property
    def key(self) -> str:
        """``tenant`` or ``tenant/namespace``: unambiguous because neither part holds a ``/``."""
        return self.tenant if self.namespace is None else f"{self.tenant}/{self.namespace}"


DEFAULT_SCOPE = Scope()


_acting_tenant: ContextVar[str] = ContextVar("acting_tenant", default=DEFAULT_TENANT)


def current_tenant() -> str:
    """The tenant the running call acts for; the default tenant outside :func:`acting_for`."""
    return _acting_tenant.get()


@contextmanager
def acting_for(tenant: str) -> Iterator[None]:
    """Mark the code inside as working for ``tenant``.

    The four store ports take a scope, but the LLM client does not: it is shared by every tenant
    and has no scope argument. Work that calls it (ingest, synthesis) runs inside this so the
    client can keep a tenant's cached responses apart from everyone else's (GRA-139).
    """
    token = _acting_tenant.set(tenant)
    try:
        yield
    finally:
        _acting_tenant.reset(token)


def tenant_dir(tenant: str) -> str:
    """A directory name for ``tenant``: a hash, so a tenant name cannot name a path."""
    return hashlib.sha256(tenant.encode()).hexdigest()[:16]
