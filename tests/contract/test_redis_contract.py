"""Contract test: the Redis Streams trigger backend against a real Valkey (GRA-146).

Proves the real adapter behaves like the fakeredis the unit tests use: a trigger round-trips,
a dead worker's trigger is reclaimed, one document is never run by two workers at once, and
queued triggers survive a server restart (append-only file). Skips without Docker.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.contract

try:
    import redis
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    from graph_rag.messaging.redis_trigger import RedisTriggerConsumer, RedisTriggerPublisher
    from graph_rag.models import IngestTrigger
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"Valkey contract deps unavailable: {exc}", allow_module_level=True)

IMAGE = "valkey/valkey:8.1.10"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def valkey() -> Iterator[tuple[Any, Any]]:
    """A throwaway Valkey on a fixed host port (so it keeps it across a restart)."""
    port = _free_port()
    container = (
        DockerContainer(IMAGE)
        .with_bind_ports(6379, port)
        .with_command(
            "valkey-server --appendonly yes --appendfsync always --maxmemory-policy noeviction"
        )
    )
    try:
        container.start()
        wait_for_logs(container, "Ready to accept connections", timeout=30)
    except Exception as exc:  # noqa: BLE001 - Docker not available on this host.
        pytest.skip(f"Docker unavailable for the Valkey contract test: {exc}")
    try:
        yield container, port
    finally:
        container.stop()


def _client(port: int) -> Any:
    return redis.Redis(port=port, decode_responses=True)


def _wait_up(port: int) -> None:
    for _ in range(100):
        try:
            _client(port).ping()
            return
        except redis.RedisError:
            time.sleep(0.2)
    pytest.fail("Valkey did not come back")


def test_trigger_round_trips_and_a_dead_workers_trigger_is_reclaimed(valkey) -> None:
    _, port = valkey
    stream, group = f"t-{uuid.uuid4().hex[:8]}", "g"
    sent = IngestTrigger(bucket="documents", object_key="c.md", tenant="acme")
    RedisTriggerPublisher(_client(port), stream, group).publish(sent)
    assert _client(port).xreadgroup(group, "dead", {stream: ">"}, count=1)  # taken, then died
    got: list[IngestTrigger] = []
    rescuer = RedisTriggerConsumer(
        _client(port), stream, got.append, group=group, consumer_name="w2", reclaim_idle_ms=300
    )
    time.sleep(0.5)
    assert rescuer.poll_once(timeout_ms=100, max_records=1) == 1
    assert got == [sent]
    assert _client(port).xpending(stream, group)["pending"] == 0


def test_one_document_is_never_run_by_two_workers_at_once(valkey) -> None:
    _, port = valkey
    stream, group = f"t-{uuid.uuid4().hex[:8]}", "g"
    pub = RedisTriggerPublisher(_client(port), stream, group)
    pub.publish(IngestTrigger(bucket="b", object_key="same.md"))
    pub.publish(IngestTrigger(bucket="b", object_key="same.md"))
    inside, gate, log = threading.Event(), threading.Event(), []

    def handler(t: IngestTrigger) -> None:
        log.append("start")
        inside.set()
        gate.wait(5)
        log.append("end")

    mk = lambda name: RedisTriggerConsumer(  # noqa: E731
        _client(port), stream, handler, group=group, consumer_name=name, busy_backoff_s=0.05
    )
    a, b = mk("w1"), mk("w2")
    first = threading.Thread(target=a.poll_once, kwargs={"timeout_ms": 100, "max_records": 1})
    first.start()
    assert inside.wait(5)
    assert b.poll_once(timeout_ms=100, max_records=1) == 0  # busy, re-queued
    gate.set()
    first.join(5)
    assert b.poll_once(timeout_ms=100, max_records=1) == 1
    assert log == ["start", "end", "start", "end"]


def test_queued_triggers_survive_a_valkey_restart(valkey) -> None:
    container, port = valkey
    stream, group = f"t-{uuid.uuid4().hex[:8]}", "g"
    pub = RedisTriggerPublisher(_client(port), stream, group)
    pub.publish(IngestTrigger(bucket="b", object_key="kept-1.md"))
    pub.publish(IngestTrigger(bucket="b", object_key="kept-2.md"))
    container.get_wrapped_container().restart()
    _wait_up(port)
    got: list[IngestTrigger] = []
    c = RedisTriggerConsumer(_client(port), stream, got.append, group=group, consumer_name="w1")
    assert [c.poll_once(timeout_ms=100, max_records=1) for _ in range(2)] == [1, 1]
    assert [t.object_key for t in got] == ["kept-1.md", "kept-2.md"]
