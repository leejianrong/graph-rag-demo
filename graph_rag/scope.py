"""Whose memory a call is about: a tenant and an optional namespace inside it (ADR-0015).

The tenant comes from the API key (GRA-22, 23) and the namespace from the request (GRA-45).
Until keys exist everything runs as :data:`DEFAULT_SCOPE`, whose ids are the ones minted before
scopes existed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_SCOPE", "Scope"]

DEFAULT_TENANT = "default"


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
