"""OTLP export starts only when an endpoint is set (GRA-153). No network, no Docker."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator

import pytest

from graph_rag import observability
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue
from tests.unit.test_observability import TEXT, app_over

OTEL_ENV = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "FASTAPI_OTEL_AUTO_CONFIGURE",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_LOGS_EXPORTER",
)
EXPORT_THREADS = ("OtelBatch", "OtelPeriodic")


@pytest.fixture
def clean_otel_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in OTEL_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    observability.use_tracer_provider(None)


def export_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith(EXPORT_THREADS)]


def test_with_no_endpoint_no_exporter_starts_and_nothing_is_logged(
    clean_otel_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    from opentelemetry import metrics

    caplog.set_level(logging.WARNING)
    settings = Settings()
    assert observability.configure_tracing(settings, "graph-rag") is False
    assert observability._provider is None
    # Libraries that read OTEL_* are told to export nothing, so none starts a pipeline of its own.
    assert os.environ["FASTAPI_OTEL_AUTO_CONFIGURE"] == "false"
    assert os.environ["OTEL_METRICS_EXPORTER"] == "none"

    client = app_over(build_offline_components(settings=settings), Queue(), settings)
    with client:
        assert client.get("/health").status_code == 200
        assert client.post("/ingest", files={"file": ("a.txt", TEXT)}).status_code < 500
    assert export_threads() == []
    assert "Proxy" in type(metrics.get_meter_provider()).__name__  # no SDK meter provider
    prefixes = ("opentelemetry", "litellm", "LiteLLM")
    assert [r for r in caplog.records if r.name.startswith(prefixes)] == []


def test_the_operator_can_still_choose_their_own_otel_environment(
    clean_otel_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    observability.configure_tracing(Settings(), "graph-rag")
    assert os.environ["OTEL_METRICS_EXPORTER"] == "otlp"


def test_with_an_endpoint_spans_go_to_the_otlp_trace_exporter_and_nothing_else(
    clean_otel_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    made: list[InMemorySpanExporter] = []

    class Recording(InMemorySpanExporter):
        def __init__(self, endpoint: str) -> None:
            super().__init__()
            self.endpoint = endpoint
            made.append(self)

    # Stand in for the whole exporter package, so this runs without the ``otel`` extra and never
    # opens a connection.
    parts = "opentelemetry.exporter.otlp.proto.http.trace_exporter".split(".")
    for i in range(3, len(parts) + 1):
        fake = types.ModuleType(".".join(parts[:i]))
        fake.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    fake.OTLPSpanExporter = Recording  # type: ignore[attr-defined]
    settings = Settings(otel_exporter_otlp_endpoint="http://collector:4318/")
    assert observability.configure_tracing(settings, "graph-rag-worker") is True
    provider = observability._provider
    try:
        assert [e.endpoint for e in made] == ["http://collector:4318/v1/traces"]
        with observability.ingest_run(None):
            observability.enter_stage("read")
            observability.finish_ingest("done")
        provider.force_flush()
        names = {s.name for s in made[0].get_finished_spans()}
        assert {"ingest.document", "ingest.read"} <= names
        assert os.environ["OTEL_METRICS_EXPORTER"] == "none"  # still no OTLP metrics
    finally:
        provider.shutdown()


def test_the_worker_and_the_offline_server_configure_tracing_at_startup(
    clean_otel_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from graph_rag import main as main_module
    from graph_rag import offline_server, worker

    seen: list[str] = []
    real = observability.configure_tracing

    def spy(settings: Settings, service: str) -> bool:
        seen.append(service)
        return real(settings, service)

    monkeypatch.setattr(observability, "configure_tracing", spy)
    monkeypatch.setattr(offline_server, "configure_tracing", spy)
    monkeypatch.setenv("INGEST_MODE", "publish")

    def stop(settings: Settings) -> None:
        raise RuntimeError("stop before any store is touched")

    monkeypatch.setattr(main_module, "build_pipeline", stop)
    with pytest.raises(RuntimeError, match="stop"):
        worker.main()
    offline_server.build_app(Settings())
    assert seen == ["graph-rag-worker", "graph-rag-offline"]
    assert export_threads() == []
