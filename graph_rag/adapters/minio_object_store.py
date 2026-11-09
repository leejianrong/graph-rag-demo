"""MinIO-backed :class:`~graph_rag.ports.ObjectStore` adapter (N10).

Wraps the ``minio`` SDK behind the ``ObjectStore`` port so the pipeline reads and
writes a document's raw bytes without knowing it's talking to S3-compatible
storage. The in-memory fake (:class:`graph_rag.fakes.InMemoryObjectStore`) mirrors
this behaviour exactly, and ``tests/contract/test_object_store_contract.py`` pins
the two to the same contract — most importantly that a missing object raises
``FileNotFoundError`` (ADR-0010).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from minio import Minio
from minio.error import S3Error

from graph_rag.adapters.scoping import object_key as scoped_key
from graph_rag.logging import get_logger
from graph_rag.scope import RESERVED_KEY_PREFIX, Scope

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["MinioObjectStore"]

_logger = get_logger(__name__)

# S3 error codes that mean "the requested object does not exist" — mapped to the
# port's ``FileNotFoundError`` contract so callers (and the fake) behave alike.
_MISSING_CODES = frozenset({"NoSuchKey", "NoSuchBucket"})


class MinioObjectStore:
    """Store/fetch document bytes in MinIO, implementing the ``ObjectStore`` port.

    Construct directly from connection parameters, or via :meth:`from_settings` so
    a composition root can build it straight from a :class:`~graph_rag.config.Settings`.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ) -> None:
        """Build a MinIO-backed object store.

        Args:
            endpoint: MinIO host:port (no scheme), e.g. ``"minio:9000"``.
            access_key: MinIO access key.
            secret_key: MinIO secret key.
            secure: Whether to use TLS (``https``). Defaults to ``False`` for the
                local docker-compose stack.
        """
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        _logger.debug("MinioObjectStore initialised for endpoint=%s secure=%s", endpoint, secure)

    @classmethod
    def from_settings(cls, settings: Settings) -> MinioObjectStore:
        """Build a :class:`MinioObjectStore` from a :class:`~graph_rag.config.Settings`.

        Reads ``minio_endpoint`` / ``minio_access_key`` / ``minio_secret_key`` /
        ``minio_secure`` off ``settings``.
        """
        return cls(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def put(self, scope: Scope, bucket: str, object_key: str, data: bytes) -> None:
        """Store ``data`` under ``(bucket, object_key)``, overwriting any prior object.

        Ensures the bucket exists first (creating it if missing), matching the
        fake's overwrite-idempotent contract.
        """
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            _logger.info("created bucket %s", bucket)
        self._client.put_object(
            bucket,
            scoped_key(scope, object_key),
            io.BytesIO(data),
            length=len(data),
        )
        _logger.info("put an object (%d bytes)", len(data))

    def delete(self, scope: Scope, bucket: str, object_key: str) -> None:
        """Remove the object; a no-op if it (or its bucket) is not there."""
        try:
            self._client.remove_object(bucket, scoped_key(scope, object_key))
        except S3Error as exc:
            if exc.code not in _MISSING_CODES:
                raise
            return
        _logger.info("deleted an object")

    def scopes(self, tenant: str) -> list[Scope]:
        """The scopes of ``tenant`` that hold objects (only non-default scopes are prefixed)."""
        prefix = f"{RESERVED_KEY_PREFIX}{tenant}/"
        found: set[Scope] = set()
        for bucket in self._client.list_buckets():
            for item in self._client.list_objects(bucket.name, prefix=prefix, recursive=True):
                rest = item.object_name.removeprefix(prefix)
                if rest.startswith("root/"):
                    found.add(Scope(tenant))
                elif rest.startswith("ns/"):
                    found.add(Scope(tenant, rest.split("/")[1]))
        return sorted(found, key=lambda s: s.key)

    def keys(self, scope: Scope) -> list[tuple[str, str]]:
        """Every ``(bucket, object_key)`` in a non-default ``scope``."""
        prefix = scoped_key(scope, "")
        return [
            (bucket.name, item.object_name.removeprefix(prefix))
            for bucket in self._client.list_buckets()
            for item in self._client.list_objects(bucket.name, prefix=prefix, recursive=True)
        ]

    def get(self, scope: Scope, bucket: str, object_key: str) -> bytes:
        """Return the bytes at ``(bucket, object_key)``.

        Raises:
            FileNotFoundError: If no object exists at that location (mapped from the
                MinIO ``NoSuchKey`` / ``NoSuchBucket`` errors, matching the fake).
        """
        response = None
        try:
            response = self._client.get_object(bucket, scoped_key(scope, object_key))
            return response.read()
        except S3Error as exc:
            if exc.code in _MISSING_CODES:
                raise FileNotFoundError(f"No object at {bucket}/{object_key}") from exc
            raise
        finally:
            if response is not None:
                response.close()
                response.release_conn()
