"""``/v1/keys``: list, create and revoke the calling tenant's API keys (GRA-143, ADR-0014).

Any live key of a tenant may manage that tenant's keys; the limits are a cap on active keys, a
stricter rate on creating them, and a refusal to revoke the key the request used (so a tenant
cannot lock itself out). The secret is in the create response only, with ``Cache-Control:
no-store``; it is stored as a hash, so nothing can show it again. Nothing here logs a body, a
label or a secret: only the opaque ``key_id``. With ``REQUIRE_AUTH`` off there are no keys to
manage and every route answers 400, never acting on the default tenant.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from pydantic import BaseModel, Field, field_validator

from graph_rag.agent_api import AgentError
from graph_rag.auth import ApiKey, ApiKeys
from graph_rag.config import Settings
from graph_rag.logging import get_logger
from graph_rag.ratelimit import RateLimited, RateLimiter

__all__ = ["add_key_routes"]

_logger = get_logger(__name__)
_MAX_LABEL = 64


class KeyInfo(BaseModel):
    """A key as listed: never the secret, only its last four characters."""

    key_id: str
    label: str
    created_at: str
    key_suffix: str | None = Field(description="The last four characters of the secret.")
    status: Literal["active", "revoked"]
    revoked_at: str | None = None
    current: bool = Field(description="True for the key that made this request.")


class CreateKeyRequest(BaseModel):
    label: str = Field(default="", description=f"A name, up to {_MAX_LABEL} characters.")

    @field_validator("label")
    @classmethod
    def _clean(cls, value: str) -> str:
        value = " ".join(value.split())  # trims and collapses whitespace, newlines included
        if len(value) > _MAX_LABEL:
            raise ValueError(f"must be at most {_MAX_LABEL} characters")
        if not value.isprintable():
            raise ValueError("must not contain control characters")
        return value


class CreatedKey(KeyInfo):
    secret: str = Field(description="The whole key. Shown here once and never again.")


class KeyList(BaseModel):
    keys: list[KeyInfo]


def _info(record: ApiKey, current: str | None) -> KeyInfo:
    return KeyInfo(
        key_id=record.key_id,
        label=record.label,
        created_at=record.created_at,
        key_suffix=record.suffix or None,
        status="active" if record.active else "revoked",
        revoked_at=record.revoked_at,
        current=record.key_id == current,
    )


def add_key_routes(
    app: FastAPI,
    keys: ApiKeys | None,
    tenant_of: Callable[[Request], str],
    settings: Settings,
) -> None:
    """Serve ``/v1/keys`` on ``app``. ``keys`` is ``None`` when auth is off."""
    router = APIRouter(prefix="/v1/keys")
    create_limit = RateLimiter(settings.key_create_limit_per_minute)
    create_lock = threading.Lock()  # makes "count active, then add" one step in this process

    def store() -> ApiKeys:
        if keys is None:  # auth off: the caller is the default tenant, which has no keys
            raise AgentError(
                "invalid_request",
                "API keys exist only when the service runs with REQUIRE_AUTH on.",
                {"reason": "auth_disabled"},
            )
        return keys

    def current(request: Request) -> str | None:
        return getattr(request.state, "key_id", None)  # type: ignore[no-any-return]

    @router.get("")
    def list_keys(request: Request, tenant: str = Depends(tenant_of)) -> KeyList:
        """The calling tenant's keys, oldest first, revoked ones included."""
        mine = store().list(tenant)
        return KeyList(keys=[_info(k, current(request)) for k in mine])

    @router.post("", status_code=201)
    def create_key(
        body: CreateKeyRequest,
        request: Request,
        response: Response,
        tenant: str = Depends(tenant_of),
    ) -> CreatedKey:
        """Make a key for the calling tenant. The secret is in this response only."""
        api_keys = store()
        if (wait := create_limit.spend(tenant)) is not None:
            raise RateLimited("Too many keys created.", wait)
        with create_lock:
            active = sum(1 for k in api_keys.list(tenant) if k.active)
            if active >= settings.max_keys_per_tenant:
                raise AgentError(
                    "invalid_request",
                    f"This tenant has {active} active keys, the most allowed. Revoke one first.",
                    {"reason": "key_limit", "limit": settings.max_keys_per_tenant},
                )
            secret, record = api_keys.create(tenant, body.label)
        response.headers["Cache-Control"] = "no-store"
        _logger.info("api key created key_id=%s", record.key_id)
        return CreatedKey(**_info(record, current(request)).model_dump(), secret=secret)

    @router.delete("/{key_id}")
    def revoke_key(key_id: str, request: Request, tenant: str = Depends(tenant_of)) -> KeyInfo:
        """Revoke one of the calling tenant's keys; it gets 401 from its next request."""
        api_keys = store()
        # Another tenant's key answers exactly like one that does not exist.
        if not any(k.key_id == key_id for k in api_keys.list(tenant)):
            raise AgentError("not_found", f"No key {key_id}.")
        if key_id == current(request):
            raise AgentError(
                "invalid_request",
                "This request used that key. Revoke it with another key, so you keep access.",
                {"reason": "own_key"},
            )
        api_keys.revoke(key_id)
        _logger.info("api key revoked key_id=%s", key_id)
        revoked = next(k for k in api_keys.list(tenant) if k.key_id == key_id)
        return _info(revoked, current(request))

    app.include_router(router)
