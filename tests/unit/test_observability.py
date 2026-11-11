"""Metrics, traces and trace ids in logs (GRA-56), at the fakes seam. Nothing here needs a stack."""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from graph_rag import observability
from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.fakes import InMemoryDocumentStore
from graph_rag.logging import _JsonFormatter, _TraceFilter
from graph_rag.models import IngestTrigger
from graph_rag.observability import METRICS, metrics_response
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.wiring import build_offline_components
from graph_rag.worker import _health_server
from tests.unit.test_agent_api_v1 import Queue

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Zebulon Quillfeather works for Quillco. Quillco is based in Lisbon."
SECRETS = ["Zebulon", "Quillfeather", "Quillco", "Lisbon", "ledger.txt"]
SCRAPE_TOKEN = "scrape-credential-for-tests"


def sample(name: str, **labels: str) -> float:
    return METRICS.registry.get_sample_value(name, labels) or 0.0


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observability.use_tracer_provider(provider)
    yield exporter
    observability.use_tracer_provider(None)


def app_over(components, queue, settings: Settings, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            components.object_store,
            queue,
            settings,
            retriever=components.retriever,
            jobs=components.job_store,
            **kwargs,
        )
    )


# --- metrics ---------------------------------------------------------------


def test_requests_are_counted_by_route_template_and_status_never_by_raw_path() -> None:
    http = app_over(build_offline_components(), Queue(), Settings())
    before = sample("http_requests_total", method="GET", route="/jobs/{doc_id}", status="404")
    for doc in ("doc-aaa", "doc-bbb"):
        assert http.get(f"/jobs/{doc}").status_code == 404
    http.get("/nowhere-at-all/ledger.txt")
    assert (
        sample("http_requests_total", method="GET", route="/jobs/{doc_id}", status="404")
        == before + 2
    )
    assert sample("http_requests_total", method="GET", route="unmatched", status="404") >= 1
    exposed = metrics_response(Settings(metrics_enabled=True), None)[1].decode()
    assert "doc-aaa" not in exposed and "ledger.txt" not in exposed
    assert "http_request_duration_seconds_bucket" in exposed


def test_metrics_is_off_unless_enabled_and_open_only_when_auth_is_off() -> None:
    assert (
        app_over(build_offline_components(), Queue(), Settings()).get("/metrics").status_code == 404
    )
    on = app_over(build_offline_components(), Queue(), Settings(metrics_enabled=True))
    response = on.get("/metrics")
    assert response.status_code == 200 and "http_requests_total" in response.text


def test_with_auth_on_metrics_needs_its_own_token_and_no_tenant_key_reads_it() -> None:
    api_keys = ApiKeys(InMemoryKeyStore())
    tenant_key, _ = api_keys.create("acme")
    settings = Settings(require_auth=True, metrics_enabled=True, metrics_token=SCRAPE_TOKEN)
    http = app_over(build_offline_components(), Queue(), settings, keys=api_keys)
    assert http.get("/metrics").status_code == 401
    assert (
        http.get("/metrics", headers={"Authorization": f"Bearer {tenant_key}"}).status_code == 401
    )
    ok = http.get("/metrics", headers={"Authorization": f"Bearer {SCRAPE_TOKEN}"})
    assert ok.status_code == 200
    # the token does not open the API either
    assert (
        http.get("/jobs/x", headers={"Authorization": f"Bearer {SCRAPE_TOKEN}"}).status_code == 401
    )


def test_auth_with_metrics_but_no_token_does_not_start() -> None:
    with pytest.raises(ValueError, match="METRICS_TOKEN"):
        create_app(
            build_offline_components().object_store,
            Queue(),
            Settings(require_auth=True, metrics_enabled=True),
        )


def test_the_worker_serves_the_same_text_behind_the_same_check() -> None:
    import threading

    settings = Settings(require_auth=True, metrics_enabled=True, metrics_token=SCRAPE_TOKEN)
    server = _health_server(0, threading.Event(), lambda auth: metrics_response(settings, auth))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3)  # noqa: S310
        assert refused.value.code == 401
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {SCRAPE_TOKEN}"}
        )
        with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
            assert b"ingest_stage_duration_seconds" in response.read()
    finally:
        server.shutdown()
        server.server_close()


# --- one trace from the API to the graph write ------------------------------


class Backend:
    def __call__(self, *, model: str, prompt: str, **params: object) -> str:
        return '{"clusters": []}'


def test_one_trace_spans_the_api_accept_the_worker_stages_the_llm_call_and_the_graph_write(
    spans: InMemorySpanExporter, tmp_path: Path
) -> None:
    components = build_offline_components()
    llm = LiteLLMClient("m", cache_dir=tmp_path, completion_fn=Backend())
    components.orchestrator._coref_stage = LLMCorefStage(llm)
    queue = Queue()
    http = app_over(components, queue, Settings())

    assert (
        http.post("/v1/remember", json={"text": TEXT, "source_id": "ledger.txt"}).status_code == 202
    )
    (trigger,) = queue.triggers
    assert trigger.traceparent  # carried on the message, as it would be through Kafka
    # the worker is another process: it gets only the message
    arrived = IngestTrigger.from_json(trigger.to_json())
    components.orchestrator.process_document(arrived)

    finished = {s.name: s for s in spans.get_finished_spans()}
    names = set(finished)
    assert {"POST /v1/remember", "ingest.document", "llm.call", "ingest.knowledge graph"} <= names
    assert {f"ingest.{s}" for s in ("read", "entity recognition", "coreference")} <= names
    assert len({s.context.trace_id for s in finished.values()}) == 1
    api = finished["POST /v1/remember"]
    assert finished["ingest.document"].parent.span_id == api.context.span_id
    assert (
        finished["ingest.coreference"].parent.span_id == finished["ingest.document"].context.span_id
    )
    assert finished["llm.call"].parent.span_id == finished["ingest.coreference"].context.span_id
    assert finished["llm.call"].attributes["llm.model"] == "m"
    assert api.attributes["http.route"] == "/v1/remember"


def test_without_a_tracer_provider_nothing_is_carried_and_nothing_breaks() -> None:
    components = build_offline_components()
    queue = Queue()
    app_over(components, queue, Settings()).post("/v1/remember", json={"text": TEXT})
    assert queue.triggers[0].traceparent is None
    assert components.orchestrator.process_document(queue.triggers[0]) is not None


def test_stage_durations_and_a_failed_stage_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    components = build_offline_components()
    queue = Queue()
    http = app_over(components, queue, Settings())
    http.post("/v1/remember", json={"text": TEXT, "source_id": "a"})
    done = sample("ingest_documents_total", outcome="done")
    ner = sample("ingest_stage_duration_seconds_count", stage="entity recognition")
    components.orchestrator.process_document(queue.triggers[0])
    assert sample("ingest_documents_total", outcome="done") == done + 1
    assert sample("ingest_stage_duration_seconds_count", stage="entity recognition") == ner + 1

    def boom(self: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("store broke")

    monkeypatch.setattr(InMemoryDocumentStore, "upsert", boom)
    failed = sample("ingest_documents_total", outcome="failed")
    broke = sample("ingest_stage_failures_total", stage="read")
    components.orchestrator.process_document(queue.triggers[0])
    assert sample("ingest_documents_total", outcome="failed") == failed + 1
    assert sample("ingest_stage_failures_total", stage="read") == broke + 1


def test_llm_calls_and_tokens_are_counted_by_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("litellm")
    usage = types.SimpleNamespace(total_tokens=25)
    reply = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"))], usage=usage
    )
    fake.completion = lambda **_: reply  # type: ignore[attr-defined]
    fake.completion_cost = lambda **_: 0.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake)
    calls = sample("llm_calls_total", model="obs-model", outcome="ok")
    tokens = sample("llm_tokens_total", model="obs-model")
    llm = LiteLLMClient("obs-model", cache_dir=tmp_path)
    llm.complete("one")
    llm.complete("one")  # a cache hit: no call
    assert sample("llm_calls_total", model="obs-model", outcome="ok") == calls + 1
    assert sample("llm_tokens_total", model="obs-model") == tokens + 25

    failing = LiteLLMClient("obs-model", cache_dir=tmp_path, completion_fn=_raise)
    with pytest.raises(RuntimeError):
        failing.complete("two")
    assert sample("llm_calls_total", model="obs-model", outcome="error") >= 1


def _raise(**_: object) -> str:
    raise RuntimeError("provider down")


# --- logs and the policy ----------------------------------------------------


def test_json_logs_carry_the_trace_id_and_only_the_message(spans: InMemorySpanExporter) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(_TraceFilter())
    handler.setFormatter(_JsonFormatter())
    log = logging.getLogger("obs-test")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        log.info("outside")
        with observability._tracer().start_as_current_span("s") as span:
            log.info("inside %d", 3)
            expected = format(span.get_span_context().trace_id, "032x")
    finally:
        log.removeHandler(handler)
    outside, inside = (json.loads(line) for line in stream.getvalue().splitlines())
    assert outside["trace_id"] == "" and inside["trace_id"] == expected
    assert inside["message"] == "inside 3" and set(inside) == {
        "time",
        "level",
        "logger",
        "message",
        "trace_id",
    }


def test_no_metric_label_or_span_attribute_holds_content(spans: InMemorySpanExporter) -> None:
    components = build_offline_components()
    queue = Queue()
    http = app_over(components, queue, Settings())
    http.post("/v1/remember", json={"text": TEXT, "source_id": "ledger.txt"})
    components.orchestrator.process_document(queue.triggers[0])
    http.post("/v1/recall", json={"question": "Where does Zebulon Quillfeather work?"})
    seen = metrics_response(Settings(metrics_enabled=True), None)[1].decode()
    for span in spans.get_finished_spans():
        seen += span.name + json.dumps(dict(span.attributes or {}), default=str)
    for secret in SECRETS:
        assert secret not in seen


# --- the dashboard ----------------------------------------------------------


def exported_names() -> set[str]:
    return {s.name for family in METRICS.registry.collect() for s in family.samples}


def test_the_dashboard_queries_only_metrics_the_service_exports() -> None:
    board = json.loads((ROOT / "observability/grafana/dashboards/graph-rag.json").read_text())
    exprs = [t["expr"] for panel in board["panels"] for t in panel.get("targets", [])]
    assert exprs
    # touch each labelled series once so the registry lists it
    METRICS.http_requests.labels("GET", "/x", "200")
    names = exported_names() | {
        f"{n}_{suffix}"
        for n in (
            "http_request_duration_seconds",
            "ingest_stage_duration_seconds",
            "llm_call_duration_seconds",
        )
        for suffix in ("bucket", "count", "sum")
    }
    used = {m for e in exprs for m in re.findall(r"\b([a-z_]+_(?:total|bucket|count|sum))\b", e)}
    assert used and used <= names, used - names
    titles = " ".join(p["title"].lower() for p in board["panels"])
    for need in ("request rate", "error", "latency"):
        assert need in titles


def test_prometheus_scrapes_the_api_and_the_worker_with_the_token() -> None:
    config = yaml.safe_load((ROOT / "observability/prometheus.yml").read_text())
    jobs = {j["job_name"]: j for j in config["scrape_configs"]}
    assert set(jobs) == {"graph-rag-api", "graph-rag-worker"}
    assert jobs["graph-rag-api"]["static_configs"][0]["targets"] == ["app:8000"]
    assert jobs["graph-rag-worker"]["static_configs"][0]["targets"] == ["worker:8001"]
    compose = yaml.safe_load((ROOT / "docker-compose.observability.yml").read_text())
    assert {"prometheus", "grafana", "jaeger"} <= set(compose["services"])
