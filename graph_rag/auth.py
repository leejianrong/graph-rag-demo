"""API keys: issue, verify, list and revoke, per tenant (GRA-22, ADR-0014 and ADR-0015).

A key looks like ``grk_<key_id>_<token>``. The whole string is shown once, when it is created;
only its SHA-256 is stored, next to the tenant it opens. SHA-256 with no salt is enough because a
key is 256 random bits, so there is nothing to guess; a password would need a slow hash.

Records live behind :class:`KeyStore`. :class:`JsonFileKeyStore` is one file, fine for a single
service instance; a deployment with several instances needs a shared store, which is not built.

The ``graph-rag-keys`` command manages the file: ``create TENANT``, ``list`` and ``revoke KEY_ID``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from graph_rag.scope import DEFAULT_TENANT, Scope

__all__ = ["ApiKey", "ApiKeys", "AuthError", "InMemoryKeyStore", "JsonFileKeyStore", "KeyStore"]

_PREFIX = "grk"


class AuthError(Exception):
    """A request carried no key, or a key that is unknown or revoked."""


@dataclass(frozen=True)
class ApiKey:
    """What is stored about a key: never the key itself."""

    key_id: str
    tenant: str
    label: str
    key_hash: str
    created_at: str
    revoked_at: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class KeyStore(Protocol):
    """Where key records are kept."""

    def add(self, key: ApiKey) -> None: ...

    def get(self, key_id: str) -> ApiKey | None: ...

    def all(self) -> list[ApiKey]: ...

    def replace(self, key: ApiKey) -> None: ...


class InMemoryKeyStore:
    """A store for tests and for a service that issues no keys."""

    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {}

    def add(self, key: ApiKey) -> None:
        self._keys[key.key_id] = key

    def get(self, key_id: str) -> ApiKey | None:
        return self._keys.get(key_id)

    def all(self) -> list[ApiKey]:
        return list(self._keys.values())

    def replace(self, key: ApiKey) -> None:
        self._keys[key.key_id] = key


class JsonFileKeyStore:
    """Key records in one JSON file, read on every call so a revoke takes effect at once."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, ApiKey]:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        return {key_id: ApiKey(**fields) for key_id, fields in raw.items()}

    def _write(self, keys: dict[str, ApiKey]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self._path.parent, prefix=".keys-")
        with os.fdopen(fd, "w") as handle:
            json.dump({k: asdict(v) for k, v in keys.items()}, handle, indent=2)
        os.chmod(temp, 0o600)
        os.replace(temp, self._path)

    def add(self, key: ApiKey) -> None:
        with self._lock:
            keys = self._read()
            keys[key.key_id] = key
            self._write(keys)

    def get(self, key_id: str) -> ApiKey | None:
        with self._lock:
            return self._read().get(key_id)

    def all(self) -> list[ApiKey]:
        with self._lock:
            return list(self._read().values())

    def replace(self, key: ApiKey) -> None:
        self.add(key)


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class ApiKeys:
    """Issue, verify, list and revoke keys over a :class:`KeyStore`."""

    def __init__(self, store: KeyStore) -> None:
        self._store = store

    def create(self, tenant: str, label: str = "") -> tuple[str, ApiKey]:
        """Make a key for ``tenant``. Returns the secret, which is not kept, and its record."""
        Scope(tenant=tenant)  # rejects a tenant that cannot be one
        if tenant == DEFAULT_TENANT:
            # The default scope holds everything written before tenancy and with auth off, so a
            # key naming it would read data that belongs to no tenant (THREAT-MODEL.md, section 1).
            raise ValueError(
                f"no key for the {DEFAULT_TENANT!r} tenant: it holds data that predates tenancy"
            )
        key_id = secrets.token_hex(4)
        secret = f"{_PREFIX}_{key_id}_{secrets.token_urlsafe(32)}"
        record = ApiKey(
            key_id=key_id,
            tenant=tenant,
            label=label,
            key_hash=_hash(secret),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._store.add(record)
        return secret, record

    def verify(self, secret: str) -> ApiKey:
        """The record for a live key; :class:`AuthError` for anything else."""
        parts = secret.split("_", 2)
        if len(parts) != 3 or parts[0] != _PREFIX:
            raise AuthError("The API key is not in the expected format.")
        record = self._store.get(parts[1])
        # Compare even when the id is unknown, so a miss and a wrong token cost the same.
        expected = record.key_hash if record else _hash("")
        matches = hmac.compare_digest(expected, _hash(secret))
        if record is None or not matches:
            raise AuthError("The API key is not recognised.")
        if not record.active:
            raise AuthError("The API key has been revoked.")
        if record.tenant == DEFAULT_TENANT:  # a key file from before create refused it
            raise AuthError("The API key is not recognised.")
        return record

    def revoke(self, key_id: str) -> bool:
        """Revoke a key. False when there is no such key; revoking twice keeps the first time."""
        record = self._store.get(key_id)
        if record is None:
            return False
        if record.active:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            self._store.replace(ApiKey(**{**asdict(record), "revoked_at": now}))
        return True

    def list(self, tenant: str | None = None) -> list[ApiKey]:
        """Records, oldest first, optionally for one tenant."""
        keys = [k for k in self._store.all() if tenant is None or k.tenant == tenant]
        return sorted(keys, key=lambda k: k.created_at)


def _llm_key_command(args: argparse.Namespace) -> int:
    """``graph-rag-keys llm-key``: set, remove and list tenants' own LLM keys, encrypted."""
    import sys
    from getpass import getpass

    from graph_rag.config import get_settings
    from graph_rag.tenant_keys import TenantLlmKeys

    try:
        stored = TenantLlmKeys.from_settings(get_settings(), always=True)
        assert stored is not None
        if args.llm_command == "list":
            print("\n".join(stored.tenants()))
        elif args.llm_command == "remove":
            if not stored.remove(args.tenant):
                print(f"error: {args.tenant} has no LLM key")
                return 1
            print(f"removed the LLM key of {args.tenant}")
        else:  # the key comes from stdin or a prompt, never an argument, so it stays out of history
            key = sys.stdin.read() if not sys.stdin.isatty() else getpass("LLM provider key: ")
            stored.set(args.tenant, key)
            print(f"stored an encrypted LLM key for {args.tenant}")
    except ValueError as error:
        print(f"error: {error}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """The ``graph-rag-keys`` command."""
    from graph_rag.config import get_settings

    parser = argparse.ArgumentParser(prog="graph-rag-keys", description="Manage API keys.")
    parser.add_argument("--file", default=None, help="the key file (default: API_KEYS_PATH)")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="make a key; it is shown once")
    create.add_argument("tenant")
    create.add_argument("--label", default="")
    listing = sub.add_parser("list", help="list keys, never their secrets")
    listing.add_argument("--tenant")
    revoke = sub.add_parser("revoke", help="stop a key working")
    revoke.add_argument("key_id")
    llm = sub.add_parser("llm-key", help="a tenant's own LLM provider key (GRA-25)")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    llm_sub.add_parser("set", help="store it, read from stdin or a prompt").add_argument("tenant")
    llm_sub.add_parser("remove", help="remove it").add_argument("tenant")
    llm_sub.add_parser("list", help="tenants that have one, never the keys")
    args = parser.parse_args(argv)

    if args.command == "llm-key":
        return _llm_key_command(args)
    keys = ApiKeys(JsonFileKeyStore(args.file or get_settings().api_keys_path))
    if args.command == "create":
        try:
            secret, record = keys.create(args.tenant, args.label)
        except ValueError as error:
            print(f"error: {error}")
            return 2
        print(f"key_id: {record.key_id}\ntenant: {record.tenant}\nkey:    {secret}")
        print("Copy the key now. It is not stored and cannot be shown again.")
    elif args.command == "list":
        for k in keys.list(args.tenant):
            state = "active" if k.active else f"revoked {k.revoked_at}"
            print(f"{k.key_id}\t{k.tenant}\t{state}\t{k.created_at}\t{k.label}")
    else:
        if not keys.revoke(args.key_id):
            print(f"error: no key {args.key_id}")
            return 1
        print(f"revoked {args.key_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
