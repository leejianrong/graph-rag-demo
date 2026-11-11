"""Per-tenant LLM provider keys, stored encrypted (GRA-25).

With ``require_tenant_llm_key`` on, every named tenant's LLM calls (ingest, synthesis) use the
provider key that tenant brought, never the deployment's own, and ingest for a tenant with no key
is refused with :class:`LlmKeyRequired`. The default tenant is not asked for one: it keeps the
deployment key, or the no-LLM path. Off, none of this runs.

A key is encrypted with Fernet (``cryptography``; AES-128-CBC with an HMAC) under a master key from
``TENANT_LLM_KEY_SECRET`` before it reaches the store, so a copy of the store file holds no usable
key. The master key itself lives in the environment, like the other secrets, and is the one thing
to guard; losing it makes the stored keys unreadable, and they are set again.

Records live behind :class:`TenantKeyStore`, keyed by tenant, so one tenant cannot read another's.
:class:`JsonFileTenantKeyStore` is one file, fine for a single service instance, like the API key
file (``graph_rag/auth.py``). The ``graph-rag-keys llm-key`` commands set and remove keys.

A key is never logged: nothing here logs one, and the commands read it from a prompt or stdin so it
is not in the shell history or the process list (``tests/unit/test_tenant_llm_keys.py``).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from cryptography.fernet import Fernet, InvalidToken

from graph_rag.scope import DEFAULT_TENANT, Scope

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = [
    "InMemoryTenantKeyStore",
    "JsonFileTenantKeyStore",
    "LlmKeyRequired",
    "TenantKeyStore",
    "TenantLlmKeys",
]


class LlmKeyRequired(Exception):
    """A tenant with no LLM key of its own asked for work that calls the provider."""


class TenantKeyStore(Protocol):
    """Where encrypted keys are kept, one per tenant."""

    def get(self, tenant: str) -> str | None: ...

    def put(self, tenant: str, token: str) -> None: ...

    def delete(self, tenant: str) -> bool: ...

    def tenants(self) -> list[str]: ...


class InMemoryTenantKeyStore:
    """A store for tests."""

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}

    def get(self, tenant: str) -> str | None:
        return self._tokens.get(tenant)

    def put(self, tenant: str, token: str) -> None:
        self._tokens[tenant] = token

    def delete(self, tenant: str) -> bool:
        return self._tokens.pop(tenant, None) is not None

    def tenants(self) -> list[str]:
        return sorted(self._tokens)


class JsonFileTenantKeyStore:
    """Encrypted keys in one JSON file (mode 0600), read on every call so a removal is at once."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        try:
            return dict(json.loads(self._path.read_text()))
        except FileNotFoundError:
            return {}

    def _write(self, tokens: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self._path.parent, prefix=".llm-keys-")
        with os.fdopen(fd, "w") as handle:
            json.dump(tokens, handle, indent=2)
        os.chmod(temp, 0o600)
        os.replace(temp, self._path)

    def get(self, tenant: str) -> str | None:
        with self._lock:
            return self._read().get(tenant)

    def put(self, tenant: str, token: str) -> None:
        with self._lock:
            tokens = self._read()
            tokens[tenant] = token
            self._write(tokens)

    def delete(self, tenant: str) -> bool:
        with self._lock:
            tokens = self._read()
            if tenant not in tokens:
                return False
            del tokens[tenant]
            self._write(tokens)
            return True

    def tenants(self) -> list[str]:
        with self._lock:
            return sorted(self._read())


class TenantLlmKeys:
    """Set, read and remove each tenant's provider key over a :class:`TenantKeyStore`."""

    def __init__(self, store: TenantKeyStore, secret: str) -> None:
        """``secret`` is a Fernet key; anything else raises ``ValueError``."""
        try:
            self._fernet = Fernet(secret.encode())
        except ValueError as error:
            raise ValueError(
                "TENANT_LLM_KEY_SECRET must be a Fernet key: 32 url-safe base64-encoded bytes "
                "(make one with cryptography.fernet.Fernet.generate_key())"
            ) from error
        self._store = store

    @classmethod
    def from_settings(cls, settings: Settings, *, always: bool = False) -> TenantLlmKeys | None:
        """The keys the service reads, or ``None`` when ``require_tenant_llm_key`` is off.

        ``always`` builds them regardless of the flag, for the operator commands: keys are set
        before the flag is turned on.

        Raises:
            ValueError: If they are wanted and ``tenant_llm_key_secret`` is not a Fernet key, so a
                misconfigured deployment fails at start, not at the first ingest.
        """
        if not (always or settings.require_tenant_llm_key):
            return None
        path = settings.tenant_llm_keys_path
        return cls(JsonFileTenantKeyStore(path), settings.tenant_llm_key_secret)

    def set(self, tenant: str, key: str) -> None:
        """Store ``key`` for ``tenant``, replacing any it had. The default tenant has none."""
        Scope(tenant=tenant)  # rejects a tenant that cannot be one
        if tenant == DEFAULT_TENANT:
            raise ValueError(
                f"the {DEFAULT_TENANT!r} tenant uses the deployment's key, not its own"
            )
        if not key.strip():
            raise ValueError("the key is empty")
        self._store.put(tenant, self._fernet.encrypt(key.strip().encode()).decode())

    def get(self, tenant: str) -> str | None:
        """``tenant``'s key, or ``None`` when it has none (or it cannot be decrypted)."""
        token = self._store.get(tenant)
        if token is None:
            return None
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken:  # stored under another master key: as good as not set
            return None

    def has(self, tenant: str) -> bool:
        return self.get(tenant) is not None

    def remove(self, tenant: str) -> bool:
        """Remove ``tenant``'s key. False when it had none."""
        return self._store.delete(tenant)

    def tenants(self) -> list[str]:
        """Tenants that have a key stored, never the keys."""
        return self._store.tenants()

    def require(self, tenant: str) -> str | None:
        """The key a call for ``tenant`` must use; ``None`` for the default tenant.

        Raises:
            LlmKeyRequired: For a named tenant with no key.
        """
        if tenant == DEFAULT_TENANT:
            return None
        key = self.get(tenant)
        if key is None:
            raise LlmKeyRequired(
                f"Tenant {tenant!r} has no LLM provider key. Set one with "
                f"'graph-rag-keys llm-key set {tenant}' before ingesting."
            )
        return key
