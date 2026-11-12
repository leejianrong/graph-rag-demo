"""The ingest worker as its own process (GRA-31): entrypoint, flag, health, shutdown, shared jobs.

Real Kafka and compose are not run here; the contract job covers the real adapters. These tests
drive the entrypoint and the flag at the fakes seam.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from fastapi.testclient import TestClient

from graph_rag import main as main_module
from graph_rag import worker
from graph_rag.adapters.es_job_store import EsJobStore
from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.messaging import kafka_trigger
from graph_rag.models import IngestTrigger
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain
from tests.unit.test_jobs import _events


class FakeConsumer:
    """Loops until stopped, like ``KafkaTriggerConsumer.run_forever``."""

    def __init__(self) -> None:
        self.running = threading.Event()
        self._stop = threading.Event()
        self.closed = False
        self.max_records: int | None = None

    def run_forever(self, poll_timeout_ms: int = 1000, max_records: int | None = None) -> None:
        self.max_records = max_records
        self.running.set()
        self._stop.wait(5)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.closed = True


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _status(port: int, path: str = "/healthz") -> int:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2).status  # noqa: S310
    except urllib.error.HTTPError as error:
        return error.code


def test_the_worker_reports_healthy_while_consuming_and_closes_on_stop() -> None:
    consumer, port = FakeConsumer(), _free_port()
    thread = threading.Thread(
        target=worker.run_worker, args=(consumer, port), kwargs={"install_signals": False}
    )
    thread.start()
    assert consumer.running.wait(2)
    assert _status(port) == 200 and _status(port, "/other") == 404
    consumer.stop()
    thread.join(5)
    assert not thread.is_alive() and consumer.closed
    assert consumer.max_records == 1  # one message at a time, so a stop never strands a batch
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)  # noqa: S310


def test_sigterm_stops_the_loop_and_closes_the_consumer() -> None:
    consumer = FakeConsumer()
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        started = time.monotonic()
        worker.run_worker(consumer, None)
        assert time.monotonic() - started < 4 and consumer.closed
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


# --- the flag in the composition root -------------------------------------------------------


class _Patched:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.consumers: list[dict[str, Any]] = []
        self.threads: list[str] = []
        self.app_kwargs: dict[str, Any] = {}
        components = build_offline_components()
        pipeline = main_module.Pipeline(
            object_store=components.object_store,  # type: ignore[arg-type]
            document_store=components.document_store,  # type: ignore[arg-type]
            entity_store=components.entity_store,  # type: ignore[arg-type]
            graph_store=components.graph_store,  # type: ignore[arg-type]
            job_store=components.job_store,  # type: ignore[arg-type]
            orchestrator=components.orchestrator,
            retriever=components.retriever,
        )
        outer = self

        class FakeThread:
            def __init__(self, target: Any, name: str, daemon: bool) -> None:
                outer.threads.append(name)

            def start(self) -> None:
                pass

        monkeypatch.setattr(main_module, "build_pipeline", lambda settings: pipeline)
        monkeypatch.setattr(main_module, "enforce_bind", lambda *a, **k: None)
        monkeypatch.setattr(main_module, "enforce_credentials", lambda *a, **k: None)
        monkeypatch.setattr(main_module, "configure_logging", lambda *a, **k: None)
        monkeypatch.setattr(main_module, "build_publisher", lambda s: Queue())
        monkeypatch.setattr(
            main_module,
            "build_consumer",
            lambda s, handler: outer.consumers.append({"handler": handler}) or FakeConsumer(),
        )
        monkeypatch.setattr(main_module.threading, "Thread", FakeThread)
        monkeypatch.setattr(main_module.uvicorn, "run", lambda app, **kw: None)


def test_inline_is_the_default_and_the_api_starts_a_consumer(monkeypatch) -> None:
    assert Settings().ingest_mode == "inline"
    patched = _Patched(monkeypatch)
    monkeypatch.delenv("INGEST_MODE", raising=False)
    main_module.main()
    assert len(patched.consumers) == 1 and patched.threads == ["trigger-consumer"]


def test_publish_mode_starts_no_consumer_in_the_api(monkeypatch) -> None:
    patched = _Patched(monkeypatch)
    monkeypatch.setenv("INGEST_MODE", "publish")
    main_module.main()
    assert patched.consumers == [] and patched.threads == []


def test_the_worker_entrypoint_needs_publish_mode(monkeypatch) -> None:
    monkeypatch.setenv("INGEST_MODE", "inline")
    monkeypatch.setattr(main_module, "build_pipeline", lambda s: pytest.fail("built"))
    with pytest.raises(SystemExit, match="INGEST_MODE=publish"):
        worker.main()


def test_the_worker_entrypoint_runs_the_orchestrator_as_the_handler(monkeypatch) -> None:
    patched = _Patched(monkeypatch)
    monkeypatch.setenv("INGEST_MODE", "publish")
    monkeypatch.setattr("graph_rag.bind.enforce_credentials", lambda s: None)
    monkeypatch.setattr("graph_rag.logging.configure_logging", lambda *a: None)
    ran: list[Any] = []
    monkeypatch.setattr(
        worker, "run_worker", lambda consumer, port, **kw: ran.append((consumer, port))
    )
    monkeypatch.setattr(
        "graph_rag.messaging.factory.build_consumer",
        lambda s, handler: patched.consumers.append({"handler": handler}) or "c",
    )
    worker.main()
    assert ran == [("c", Settings().worker_health_port)]
    assert callable(patched.consumers[0]["handler"])


# --- jobs shared between the API and a worker in another process ----------------------------


class FakeEs:
    """Just enough of the Elasticsearch client for :class:`EsJobStore`, over a dict."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.indices = self

    def exists(self, index: str) -> bool:
        return True

    def get(self, index: str, id: str) -> dict[str, Any]:
        from elasticsearch import NotFoundError

        if id not in self.docs:
            raise NotFoundError("not_found", meta=None, body={})  # type: ignore[arg-type]
        return {"_source": json.loads(json.dumps(self.docs[id]))}

    def index(self, index: str, id: str, document: dict[str, Any]) -> None:
        self.docs[id] = json.loads(json.dumps(document))


def test_a_job_written_by_a_worker_is_read_and_streamed_by_the_api() -> None:
    shared = FakeEs()
    api_jobs, worker_jobs = EsJobStore(shared, "jobs"), EsJobStore(shared, "jobs")  # type: ignore[arg-type]
    components = build_offline_components()
    components.orchestrator._job_store = worker_jobs  # noqa: SLF001
    queue = Queue()
    client = TestClient(create_app(components.object_store, queue, Settings(), jobs=api_jobs))

    doc = client.post("/ingest", files={"file": ("a.txt", b"Ada Lovelace works for Acme Corp.")})
    doc_id = doc.json()["document_id"]
    assert client.get(f"/jobs/{doc_id}").json()["state"] == "queued"
    drain(components, queue)  # "the worker"
    assert client.get(f"/jobs/{doc_id}").json()["state"] == "done"
    with client.stream("GET", f"/jobs/{doc_id}/events") as response:
        events = _events("".join(response.iter_text()))
    assert events[0][0] == "stage" and events[-1][0] == "done"

    # a job is only visible to its own scope
    assert api_jobs.get(Scope("other"), doc_id) is None
    assert api_jobs.get(DEFAULT_SCOPE, "nope") is None
    assert api_jobs.get(Scope("t", "ns"), doc_id) is None


def test_the_publisher_keys_a_trigger_by_its_source() -> None:
    sent: list[dict[str, Any]] = []

    class Producer:
        def __init__(self, **kw: Any) -> None:
            pass

        def send(self, topic: str, **kw: Any) -> None:
            sent.append(kw)

        def flush(self) -> None:
            pass

    original = kafka_trigger.KafkaProducer
    kafka_trigger.KafkaProducer = Producer  # type: ignore[misc,assignment]
    try:
        publisher = kafka_trigger.KafkaTriggerPublisher("k:9092", "t")
        trigger = IngestTrigger(bucket="b", object_key="a.txt", tenant="acme")
        publisher.publish(trigger)
        publisher.publish(trigger)
    finally:
        kafka_trigger.KafkaProducer = original  # type: ignore[misc]
    assert sent[0]["key"] == sent[1]["key"] == "acme|a.txt"


def test_the_es_job_store_logic_meets_the_shared_contract_over_a_fake_client() -> None:
    from tests.contract.test_job_store_contract import check_job_store

    shared = FakeEs()
    check_job_store(EsJobStore(shared, "jobs"), EsJobStore(shared, "jobs"))  # type: ignore[arg-type]


def test_the_worker_pipeline_meters_llm_tokens_into_the_shared_usage_file(tmp_path) -> None:
    from graph_rag.adapters.llm_client import LiteLLMClient

    settings = Settings(meter_usage=True, usage_path=str(tmp_path / "usage.json"))
    # build_pipeline makes its coref and KG-build clients this way
    assert LiteLLMClient.from_settings(settings)._meter is not None  # noqa: SLF001
    assert LiteLLMClient.from_settings(Settings())._meter is None  # noqa: SLF001


def test_the_factory_picks_the_backend_from_the_setting(monkeypatch) -> None:
    import fakeredis

    from graph_rag.messaging import factory, redis_trigger

    monkeypatch.setattr(
        redis_trigger, "_client", lambda s: fakeredis.FakeRedis(decode_responses=True)
    )
    monkeypatch.setattr(kafka_trigger, "KafkaProducer", lambda **kw: object())
    monkeypatch.setattr(kafka_trigger, "KafkaConsumer", lambda *a, **kw: object())
    assert Settings().trigger_backend == "redis"
    assert isinstance(factory.build_publisher(Settings()), redis_trigger.RedisTriggerPublisher)
    assert isinstance(factory.build_consumer(Settings(), print), redis_trigger.RedisTriggerConsumer)
    kafka = Settings(trigger_backend="kafka")
    assert isinstance(factory.build_publisher(kafka), kafka_trigger.KafkaTriggerPublisher)
    assert isinstance(factory.build_consumer(kafka, print), kafka_trigger.KafkaTriggerConsumer)
