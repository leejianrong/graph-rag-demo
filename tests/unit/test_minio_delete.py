"""``MinioObjectStore.delete`` against a stub client (GRA-28).

The MinIO contract tests skip where the pinned image cannot be pulled (they do in CI), so this
keeps the adapter's own rules covered without a server: delete asks MinIO to remove the object,
a missing key or bucket is a no-op like the fake, and any other S3 error is raised.
"""

from __future__ import annotations

import pytest
from minio.error import S3Error

from graph_rag.adapters.minio_object_store import MinioObjectStore


class _Client:
    def __init__(self, error: S3Error | None = None) -> None:
        self.removed: list[tuple[str, str]] = []
        self._error = error

    def remove_object(self, bucket: str, key: str) -> None:
        self.removed.append((bucket, key))
        if self._error:
            raise self._error


def _store(client: _Client) -> MinioObjectStore:
    store = MinioObjectStore("localhost:9", "access", "secret")
    store._client = client  # type: ignore[assignment]  # noqa: SLF001
    return store


def _error(code: str) -> S3Error:
    return S3Error(None, code, code, "/resource", "request-id", "host-id")  # type: ignore[arg-type]


def test_delete_asks_minio_to_remove_the_object() -> None:
    client = _Client()
    _store(client).delete("documents", "a.md")
    assert client.removed == [("documents", "a.md")]


@pytest.mark.parametrize("code", ["NoSuchKey", "NoSuchBucket"])
def test_a_missing_object_or_bucket_is_not_an_error(code: str) -> None:
    client = _Client(_error(code))
    _store(client).delete("documents", "a.md")  # does not raise
    assert client.removed == [("documents", "a.md")]


def test_any_other_s3_error_is_raised() -> None:
    with pytest.raises(S3Error):
        _store(_Client(_error("AccessDenied"))).delete("documents", "a.md")
