"""How the real stores keep one scope's records apart from another's (ADR-0015).

Every record written for a scope carries ``tenant`` and ``scope`` (``tenant`` or
``tenant/namespace``) fields, and every read filters on ``scope``. A record written before scopes
existed has neither field and belongs to the default scope. Object keys are different: they are
prefixed, and the prefix is reserved so a default-scope key can never spell another scope's
location.
"""

from __future__ import annotations

from typing import Any

from graph_rag.scope import DEFAULT_SCOPE, RESERVED_KEY_PREFIX, Scope

__all__ = ["es_in_scope", "es_scope_filter", "object_key", "scope_fields"]


def scope_fields(scope: Scope) -> dict[str, str]:
    """The fields stamped on every record written for ``scope``."""
    return {"tenant": scope.tenant, "scope": scope.key}


def es_scope_filter(scope: Scope) -> dict[str, Any]:
    """An Elasticsearch query clause matching only ``scope``'s records."""
    if scope == DEFAULT_SCOPE:
        return {
            "bool": {
                "should": [
                    {"term": {"scope": scope.key}},
                    {"bool": {"must_not": [{"exists": {"field": "scope"}}]}},
                ],
                "minimum_should_match": 1,
            }
        }
    return {"term": {"scope": scope.key}}


def es_in_scope(source: dict[str, Any], scope: Scope) -> bool:
    """Whether a fetched ``_source`` is in ``scope``; a record with no field is the default's."""
    return source.get("scope", DEFAULT_SCOPE.key) == scope.key


def object_key(scope: Scope, key: str) -> str:
    """Where ``key`` lives in the bucket for ``scope``.

    The default scope keeps its key as given, so objects stored before scopes existed stay where
    they are. Any other scope gets ``_scopes/<tenant>/root/<key>`` or
    ``_scopes/<tenant>/ns/<namespace>/<key>``, unambiguous because a tenant and a namespace hold
    no ``/``.

    Raises:
        ValueError: If a default-scope key starts with the reserved prefix.
    """
    if scope == DEFAULT_SCOPE:
        if key.startswith(RESERVED_KEY_PREFIX):
            raise ValueError(f"object keys may not start with {RESERVED_KEY_PREFIX!r}")
        return key
    where = "root" if scope.namespace is None else f"ns/{scope.namespace}"
    return f"{RESERVED_KEY_PREFIX}{scope.tenant}/{where}/{key}"
