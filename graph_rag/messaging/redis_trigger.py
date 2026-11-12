"""Redis Streams messaging seam for ingest triggers (GRA-146, ADR-0016).

Valkey and Redis speak the same protocol, so this is one adapter for both. It mirrors the Kafka
classes: :class:`RedisTriggerPublisher` implements ``TriggerPublisher``;
:class:`RedisTriggerConsumer` is the thin driver that hands each trigger to an injected handler.

Delivery is at-least-once. ``XADD`` publishes; a worker ``XREADGROUP``s as its own consumer and
``XACK``s (then ``XDEL``s, so the stream does not grow) only after the handler returns. Nothing
is trimmed, so an unacknowledged trigger is never lost. Two mechanisms replace what Kafka's keyed
partitions gave:

* **Crash recovery**: a trigger held longer than ``reclaim_idle_ms`` by a worker that is not
  renewing it is taken over with ``XAUTOCLAIM`` by whichever worker polls next.
* **One worker per document**: a lock per ``scope|object_key`` (``SET NX PX``, renewed while the
  handler runs, released after the ack). A trigger that finds its document locked is re-enqueued
  at the tail and the worker backs off briefly; it is never dropped, never acknowledged unhandled.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import redis

from graph_rag.logging import get_logger
from graph_rag.models import IngestTrigger

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["RedisTriggerPublisher", "RedisTriggerConsumer"]

_logger = get_logger(__name__)

TriggerHandler = Callable[[IngestTrigger], object]  # a return value, if any, is ignored

_FIELD = "trigger"


def _document_key(trigger: IngestTrigger) -> str:
    return f"{trigger.scope.key}|{trigger.object_key}"


def _client(settings: Settings) -> Any:
    return redis.Redis.from_url(
        settings.redis_url, password=settings.redis_password or None, decode_responses=True
    )


def _ensure_group(client: Any, stream: str, group: str) -> None:
    """Create the group at the start of the stream (and the stream) if absent."""
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as error:
        if "BUSYGROUP" not in str(error):
            raise


class RedisTriggerPublisher:
    """Publish triggers to a Redis stream (implements ``TriggerPublisher``)."""

    def __init__(self, client: Any, stream: str, group: str = "graph-rag-ingest") -> None:
        self._r = client
        self._stream = stream
        self._group = group

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisTriggerPublisher:
        """Build a publisher from ``redis_url``, ``redis_password``, ``redis_stream``."""
        return cls(_client(settings), settings.redis_stream, settings.redis_group)

    def publish(self, trigger: IngestTrigger) -> None:
        """Append ``trigger`` (JSON, with its traceparent) to the stream."""
        _ensure_group(self._r, self._stream, self._group)  # so nothing is lost before a worker runs
        self._r.xadd(self._stream, {_FIELD: trigger.to_json()})
        _logger.info("published trigger tenant=%s stream=%s", trigger.tenant, self._stream)

    def close(self) -> None:
        """Close the connection."""
        self._r.close()


class RedisTriggerConsumer:
    """Thin driver: take triggers from the stream and hand each to ``handler``.

    A malformed message, or a handler that raises, is logged and acknowledged (ADR-0001: a failing
    document is the orchestrator's to record; the loop must keep going), the same as the Kafka
    consumer. A *crash* is the case that leaves a trigger unacknowledged, and it is reclaimed.
    """

    def __init__(
        self,
        client: Any,
        stream: str,
        handler: TriggerHandler,
        *,
        group: str = "graph-rag-ingest",
        consumer_name: str | None = None,
        reclaim_idle_ms: int = 900_000,
        lock_ms: int = 300_000,
        busy_backoff_s: float = 1.0,
    ) -> None:
        """Build a consumer; ``consumer_name`` defaults to a name unique to this process."""
        self._r = client
        self._stream = stream
        self._handler = handler
        self._group = group
        self._name = consumer_name or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._reclaim_idle_ms = reclaim_idle_ms
        self._lock_ms = lock_ms
        self._backoff = busy_backoff_s
        self._stopping = threading.Event()
        _ensure_group(self._r, stream, group)

    @classmethod
    def from_settings(cls, settings: Settings, handler: TriggerHandler) -> RedisTriggerConsumer:
        """Build a consumer from the ``redis_*`` settings."""
        return cls(
            _client(settings),
            settings.redis_stream,
            handler,
            group=settings.redis_group,
            reclaim_idle_ms=settings.redis_reclaim_idle_seconds * 1000,
            lock_ms=settings.redis_lock_seconds * 1000,
        )

    # --- the per-document lock ---------------------------------------------------------

    def _lock_key(self, trigger: IngestTrigger) -> str:
        digest = hashlib.sha256(_document_key(trigger).encode()).hexdigest()
        return f"{self._stream}:lock:{digest}"

    def _if_owner(self, key: str, command: str, *args: Any) -> None:
        """Run ``command`` on ``key`` only while this consumer still owns the lock."""
        with self._r.pipeline() as pipe:
            try:
                pipe.watch(key)
                if pipe.get(key) == self._name:
                    pipe.multi()
                    getattr(pipe, command)(key, *args)
                    pipe.execute()
            except redis.WatchError:
                pass  # the lock changed hands under us: not ours to touch

    # --- handling one entry ------------------------------------------------------------

    def _finish(self, entry_id: str) -> None:
        self._r.xack(self._stream, self._group, entry_id)
        self._r.xdel(
            self._stream, entry_id
        )  # acknowledged, so safe to drop; keeps the stream small

    def _heartbeat(self, lock_key: str, entry_id: str, done: threading.Event) -> None:
        """While the handler runs: keep the lock, and keep the entry from looking abandoned."""
        interval = max(self._lock_ms / 3000, 0.05)
        while not done.wait(interval):
            try:
                self._if_owner(lock_key, "pexpire", self._lock_ms)
                self._r.xclaim(self._stream, self._group, self._name, 0, [entry_id])
            except redis.RedisError:
                _logger.warning("could not renew a claim; will retry", exc_info=True)

    def _handle(self, entry_id: str, fields: dict[str, str]) -> bool:
        """Process one delivered entry; True if the handler ran."""
        try:
            trigger = IngestTrigger.from_json(fields.get(_FIELD, ""))
        except Exception:  # noqa: BLE001 - a bad message must not wedge the loop.
            _logger.exception("dropping a malformed trigger message")
            self._finish(entry_id)
            return False
        lock_key = self._lock_key(trigger)
        if not self._r.set(lock_key, self._name, nx=True, px=self._lock_ms):
            # Another worker has this document. Put the trigger back at the tail *before*
            # acknowledging this copy, so a crash in between duplicates it rather than losing it.
            self._r.xadd(self._stream, {_FIELD: trigger.to_json()})
            self._finish(entry_id)
            _logger.info("document busy on another worker; trigger re-queued")
            self._stopping.wait(self._backoff)
            return False
        done = threading.Event()
        beat = threading.Thread(
            target=self._heartbeat, args=(lock_key, entry_id, done), name="trigger-heartbeat"
        )
        beat.start()
        _logger.info("received trigger tenant=%s", trigger.tenant)
        try:
            self._handler(trigger)
        except Exception:  # noqa: BLE001 - handler failure must not wedge the loop.
            _logger.exception("handler failed for a trigger of tenant %s", trigger.tenant)
        finally:
            done.set()
            beat.join()
        self._finish(entry_id)
        self._if_owner(lock_key, "delete")
        return True

    # --- the Kafka-shaped surface ------------------------------------------------------

    def poll_once(self, timeout_ms: int = 1000, max_records: int | None = None) -> int:
        """Handle up to ``max_records`` triggers (abandoned ones first); return how many ran."""
        count = max_records or 10
        _next, claimed, *_ = self._r.xautoclaim(
            self._stream, self._group, self._name, self._reclaim_idle_ms, "0-0", count=count
        )
        entries = list(claimed)
        if entries:
            _logger.warning(
                "reclaimed %d trigger(s) from a worker that stopped renewing", len(entries)
            )
        else:
            batches = self._r.xreadgroup(
                self._group, self._name, {self._stream: ">"}, count=count, block=timeout_ms
            )
            entries = [e for _stream, items in batches or [] for e in items]
        handled = 0
        for entry_id, fields in entries:
            if fields is None:  # deleted while pending
                self._r.xack(self._stream, self._group, entry_id)
            elif self._handle(entry_id, fields):
                handled += 1
        return handled

    def run_forever(self, poll_timeout_ms: int = 1000, max_records: int | None = None) -> None:
        """Consume until :meth:`stop`.

        A worker passes ``max_records=1``: stopping then waits for one document and leaves the
        unread triggers unread.
        """
        _logger.info("consumer loop started stream=%s", self._stream)
        try:
            while not self._stopping.is_set():
                try:
                    self.poll_once(timeout_ms=poll_timeout_ms, max_records=max_records)
                except (redis.ConnectionError, redis.TimeoutError):
                    _logger.warning("trigger store unreachable; retrying", exc_info=True)
                    self._stopping.wait(1.0)
        finally:
            _logger.info("consumer loop stopped stream=%s", self._stream)

    run = run_forever

    def stop(self) -> None:
        """Make :meth:`run_forever` exit after the document in hand."""
        self._stopping.set()

    def close(self) -> None:
        """Stop, leave the group if this consumer holds nothing, close the connection."""
        self.stop()
        try:
            for info in self._r.xinfo_consumers(self._stream, self._group):
                if info["name"] == self._name and info["pending"] == 0:
                    self._r.xgroup_delconsumer(self._stream, self._group, self._name)
        except redis.RedisError:
            _logger.debug("could not remove the consumer from its group", exc_info=True)
        self._r.close()
