"""The Redis Streams trigger backend (GRA-146), over fakeredis (which implements Streams).

The real Valkey is covered by tests/contract/test_redis_contract.py. These prove the delivery
rules at the seam: at-least-once, reclaim of a dead worker's trigger, one worker per document,
and a stop that leaves unread triggers unread.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import fakeredis
import pytest
import redis

from graph_rag.messaging.redis_trigger import RedisTriggerConsumer, RedisTriggerPublisher
from graph_rag.models import IngestTrigger

STREAM, GROUP = "triggers", "g"
TRACE = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


def client(server: fakeredis.FakeServer) -> Any:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def consumer(
    server: fakeredis.FakeServer, handler: Any, name: str, **kw: Any
) -> RedisTriggerConsumer:
    kw.setdefault("busy_backoff_s", 0.01)
    return RedisTriggerConsumer(
        client(server), STREAM, handler, group=GROUP, consumer_name=name, **kw
    )


def publisher(server: fakeredis.FakeServer) -> RedisTriggerPublisher:
    return RedisTriggerPublisher(client(server), STREAM, GROUP)


def trigger(key: str = "a.md", tenant: str = "acme") -> IngestTrigger:
    return IngestTrigger(bucket="b", object_key=key, tenant=tenant, traceparent=TRACE)


def pending(server: fakeredis.FakeServer) -> int:
    return int(client(server).xpending(STREAM, GROUP)["pending"])


def run_in_parallel(*consumers: RedisTriggerConsumer) -> list[int]:
    results: list[int] = []
    threads = [
        threading.Thread(
            target=lambda c=c: results.append(c.poll_once(timeout_ms=10, max_records=1))
        )
        for c in consumers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    return results


def test_a_published_trigger_reaches_the_handler_intact_and_is_acknowledged(server) -> None:
    got: list[IngestTrigger] = []
    sent = trigger()
    publisher(server).publish(sent)
    c = consumer(server, got.append, "w1")
    assert c.poll_once(timeout_ms=10, max_records=1) == 1
    assert got == [sent] and got[0].traceparent == TRACE
    assert pending(server) == 0 and client(server).xlen(STREAM) == 0  # acked, then dropped


def test_triggers_published_before_any_worker_runs_are_kept_in_order(server) -> None:
    publisher(server).publish(trigger("a"))
    publisher(server).publish(trigger("b"))
    got: list[IngestTrigger] = []
    c = consumer(server, got.append, "w1")
    assert c.poll_once(timeout_ms=10, max_records=1) == 1
    assert c.poll_once(timeout_ms=10, max_records=1) == 1
    assert [t.object_key for t in got] == ["a", "b"]


def test_a_message_is_acknowledged_only_after_the_handler_returns(server) -> None:
    publisher(server).publish(trigger())
    seen: list[int] = []
    c = consumer(server, lambda t: seen.append(pending(server)), "w1")
    c.poll_once(timeout_ms=10, max_records=1)
    assert seen == [1] and pending(server) == 0  # still pending while it ran


def test_a_malformed_message_or_failing_handler_is_dropped_and_the_loop_goes_on(server) -> None:
    client(server).xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    client(server).xadd(STREAM, {"trigger": "not json"})
    publisher(server).publish(trigger("boom"))
    publisher(server).publish(trigger("ok"))
    done: list[str] = []

    def handler(t: IngestTrigger) -> None:
        if t.object_key == "boom":
            raise RuntimeError("document failed")
        done.append(t.object_key)

    c = consumer(server, handler, "w1")
    handled = [c.poll_once(timeout_ms=10, max_records=1) for _ in range(3)]
    assert handled == [0, 1, 1] and done == ["ok"]  # the failing one still counts as handled
    assert pending(server) == 0


def test_a_dead_workers_trigger_is_reclaimed_after_the_idle_time_and_not_before(server) -> None:
    publisher(server).publish(trigger())
    c = client(server)
    assert c.xreadgroup(GROUP, "dead", {STREAM: ">"}, count=1)  # taken, never finished
    got: list[IngestTrigger] = []
    rescuer = consumer(server, got.append, "w2", reclaim_idle_ms=300)
    assert rescuer.poll_once(timeout_ms=10, max_records=1) == 0 and got == []  # too soon
    time.sleep(0.4)
    assert rescuer.poll_once(timeout_ms=10, max_records=1) == 1
    assert [t.object_key for t in got] == ["a.md"] and pending(server) == 0


def test_a_dead_workers_document_lock_lapses_so_its_reclaimed_trigger_can_run(server) -> None:
    publisher(server).publish(trigger())
    c = client(server)
    c.xreadgroup(GROUP, "dead", {STREAM: ">"}, count=1)
    probe = consumer(server, lambda t: None, "probe", lock_ms=200)
    c.set(probe._lock_key(trigger()), "dead", px=200)  # noqa: SLF001 - the dead worker's lock
    got: list[IngestTrigger] = []
    rescuer = consumer(server, got.append, "w2", lock_ms=200, reclaim_idle_ms=300)
    time.sleep(0.4)  # past both the lock's expiry and the idle time
    assert rescuer.poll_once(timeout_ms=10, max_records=1) == 1 and len(got) == 1


def test_one_document_is_never_run_by_two_workers_at_once_and_nothing_is_lost(server) -> None:
    publisher(server).publish(trigger())
    publisher(server).publish(trigger())  # a second trigger for the same document
    running, peak, log = 0, 0, []
    gate, inside = threading.Event(), threading.Event()
    guard = threading.Lock()

    def handler(t: IngestTrigger) -> None:
        nonlocal running, peak
        with guard:
            running += 1
            peak = max(peak, running)
        log.append("start")
        inside.set()
        gate.wait(5)
        with guard:
            running -= 1
        log.append("end")

    # lock and reclaim time are both 150ms: each would lapse mid-handler if not renewed
    a = consumer(server, handler, "w1", lock_ms=150, reclaim_idle_ms=150)
    b = consumer(server, handler, "w2", lock_ms=150, reclaim_idle_ms=150)
    first = threading.Thread(target=a.poll_once, kwargs={"timeout_ms": 10, "max_records": 1})
    first.start()
    assert inside.wait(2)
    time.sleep(0.5)  # well past both 150ms limits
    assert b.poll_once(timeout_ms=10, max_records=1) == 0  # busy: re-queued, not run, not lost
    assert pending(server) == 1 and client(server).xlen(STREAM) == 2  # A's, plus the re-queued one
    gate.set()
    first.join(5)
    assert b.poll_once(timeout_ms=10, max_records=1) == 1  # the document is free again
    assert peak == 1 and log == ["start", "end", "start", "end"]
    assert pending(server) == 0 and client(server).xlen(STREAM) == 0


def test_different_documents_run_in_parallel(server) -> None:
    publisher(server).publish(trigger("a"))
    publisher(server).publish(trigger("b"))
    barrier = threading.Barrier(2, timeout=3)  # both handlers must be inside at once
    a = consumer(server, lambda t: barrier.wait(), "w1")
    b = consumer(server, lambda t: barrier.wait(), "w2")
    assert run_in_parallel(a, b) == [1, 1]


def test_the_same_source_in_another_tenant_is_a_different_document(server) -> None:
    publisher(server).publish(trigger("a.md", "one"))
    publisher(server).publish(trigger("a.md", "two"))
    barrier = threading.Barrier(2, timeout=3)
    a = consumer(server, lambda t: barrier.wait(), "w1")
    b = consumer(server, lambda t: barrier.wait(), "w2")
    assert run_in_parallel(a, b) == [1, 1]


def test_stop_finishes_the_document_in_hand_and_leaves_unread_triggers_unread(server) -> None:
    for key in ("a", "b", "c"):
        publisher(server).publish(trigger(key))
    done: list[str] = []
    holder: dict[str, RedisTriggerConsumer] = {}

    def handler(t: IngestTrigger) -> None:
        done.append(t.object_key)
        holder["c"].stop()  # SIGTERM arrives mid-document

    holder["c"] = consumer(server, handler, "w1")
    holder["c"].run_forever(poll_timeout_ms=10, max_records=1)
    assert done == ["a"] and pending(server) == 0
    assert client(server).xlen(STREAM) == 2  # b and c were never read
    assert client(server).xinfo_groups(STREAM)[0]["lag"] == 2
    holder["c"].close()
    assert client(server).xinfo_consumers(STREAM, GROUP) == []  # a clean worker leaves no ghost


def test_the_loop_rides_out_the_server_being_unreachable(server, monkeypatch) -> None:
    publisher(server).publish(trigger())
    c = consumer(server, lambda t: c.stop(), "w1")
    real, calls = c._r.xreadgroup, []  # noqa: SLF001

    def flaky(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise redis.ConnectionError("restarting")
        return real(*a, **k)

    monkeypatch.setattr(c._r, "xreadgroup", flaky)  # noqa: SLF001
    monkeypatch.setattr(c._stopping, "wait", lambda t=None: False)  # noqa: SLF001 - no real sleep
    c.run_forever(poll_timeout_ms=10, max_records=1)
    assert len(calls) == 2 and pending(server) == 0  # retried, then handled the trigger
