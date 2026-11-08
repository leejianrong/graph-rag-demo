"""Contract test: real :class:`MinioObjectStore` behaves like the in-memory fake.

Runs the real MinIO adapter against a testcontainers MinIO and pins it to the same
observable contract as :class:`~graph_rag.fakes.InMemoryObjectStore` (ADR-0010):
byte round-trip, ``FileNotFoundError`` on a missing key, and overwrite-idempotent
``put``. Skips cleanly when Docker is unavailable.

Marked ``contract`` — excluded from the fast ($0) suite.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from graph_rag.scope import DEFAULT_SCOPE
from tests.isolation import check_object_store

pytestmark = pytest.mark.contract

# minio/minio is no longer published to Docker Hub or Quay, so the testcontainers
# default (a 2022 tag of it) cannot be pulled. This is the same server binary.
_MINIO_IMAGE = "chainguard/minio:latest"

# Import guard: testcontainers/docker deps or a missing daemon must skip, not error.
try:
    from testcontainers.minio import MinioContainer

    from graph_rag.adapters.minio_object_store import MinioObjectStore
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"MinIO contract deps unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def object_store() -> Iterator[MinioObjectStore]:
    """A real :class:`MinioObjectStore` backed by a throwaway MinIO container."""
    try:
        container = MinioContainer(_MINIO_IMAGE)
        container.start()
    except Exception as exc:  # noqa: BLE001 — Docker not available on this host.
        pytest.skip(f"Docker unavailable for MinIO contract test: {exc}")

    try:
        config = container.get_config()
        store = MinioObjectStore(
            endpoint=config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=False,
        )
        yield store
    finally:
        container.stop()


def test_round_trip_returns_identical_bytes(object_store: MinioObjectStore) -> None:
    """put then get returns the exact same bytes."""
    bucket = f"documents-{uuid.uuid4().hex[:8]}"
    data = b"\x00\x01 the quick brown fox \xff\xfe"

    object_store.put(DEFAULT_SCOPE, bucket, "roundtrip.bin", data)

    assert object_store.get(DEFAULT_SCOPE, bucket, "roundtrip.bin") == data


def test_get_missing_key_raises_file_not_found(object_store: MinioObjectStore) -> None:
    """get of an absent object raises FileNotFoundError (matching the fake)."""
    bucket = f"documents-{uuid.uuid4().hex[:8]}"
    # Bucket exists but key does not.
    object_store.put(DEFAULT_SCOPE, bucket, "present.txt", b"present")

    with pytest.raises(FileNotFoundError):
        object_store.get(DEFAULT_SCOPE, bucket, "does-not-exist.txt")


def test_get_missing_bucket_raises_file_not_found(object_store: MinioObjectStore) -> None:
    """get against a bucket that was never created also raises FileNotFoundError."""
    bucket = f"never-created-{uuid.uuid4().hex[:8]}"

    with pytest.raises(FileNotFoundError):
        object_store.get(DEFAULT_SCOPE, bucket, "whatever.txt")


def test_put_is_overwrite_idempotent(object_store: MinioObjectStore) -> None:
    """Re-putting the same key overwrites; get returns the latest bytes."""
    bucket = f"documents-{uuid.uuid4().hex[:8]}"

    object_store.put(DEFAULT_SCOPE, bucket, "doc.txt", b"first")
    object_store.put(DEFAULT_SCOPE, bucket, "doc.txt", b"second")

    assert object_store.get(DEFAULT_SCOPE, bucket, "doc.txt") == b"second"


def test_delete_removes_the_object_and_is_a_no_op_when_it_is_missing(
    object_store: MinioObjectStore,
) -> None:
    """delete drops the bytes (get then raises); deleting again, or in a bucket that was
    never made, does nothing, like the fake."""
    from graph_rag.fakes import InMemoryObjectStore

    bucket = f"documents-{uuid.uuid4().hex[:8]}"
    fake = InMemoryObjectStore()
    for store in (object_store, fake):
        store.put(DEFAULT_SCOPE, bucket, "gone.txt", b"bytes")
        store.put(DEFAULT_SCOPE, bucket, "kept.txt", b"other")
        store.delete(DEFAULT_SCOPE, bucket, "gone.txt")
        with pytest.raises(FileNotFoundError):
            store.get(DEFAULT_SCOPE, bucket, "gone.txt")
        assert store.get(DEFAULT_SCOPE, bucket, "kept.txt") == b"other"
        store.delete(DEFAULT_SCOPE, bucket, "gone.txt")  # already gone
        store.delete(
            DEFAULT_SCOPE, f"never-created-{uuid.uuid4().hex[:8]}", "x.txt"
        )  # no such bucket


def test_scopes_are_isolated(object_store: MinioObjectStore) -> None:
    """One scope cannot read, overwrite or delete another's object (ADR-0015)."""
    check_object_store(object_store)


def test_the_default_scope_cannot_name_a_reserved_key(object_store: MinioObjectStore) -> None:
    bucket = f"documents-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError):
        object_store.put(DEFAULT_SCOPE, bucket, "_scopes/other/root/k.txt", b"x")
