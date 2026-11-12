"""Metrics, traces and trace ids in logs (GRA-56). Off the hot path and off by default.

**Metrics** are Prometheus's text format over ``prometheus-client`` (small, pure Python), in a
registry of this module's own. Every label is bounded: a route is its *template*
(``/jobs/{doc_id}``, never the path asked for), a status is a code, a model is configuration, a
stage is one of the pipeline's five. Tenant is deliberately **not** a label: tenants are unbounded
and a metric series per tenant is a cost and a leak; per-tenant counts are usage, which
``graph_rag/usage.py`` keeps. No label ever holds document text, a source name or a key
(``docs/LOG-POLICY.md``).

``/metrics`` exists only with ``METRICS_ENABLED``. With ``REQUIRE_AUTH`` on it needs its own bearer
token (``METRICS_TOKEN``), which no tenant's API key equals, so tenants cannot read it; with auth
off it is as open as the rest of the service. The worker serves the same text on its health port.

**Traces** use the OpenTelemetry API, a no-op until an SDK is configured. With
``OTEL_EXPORTER_OTLP_ENDPOINT`` set (and the ``otel`` extra installed) spans are exported over
OTLP/HTTP; unset, nothing is exported and the API costs nothing. One trace spans API accept, the
worker's stages, each LLM call and the graph write: the API puts a W3C ``traceparent`` on the
ingest trigger and the worker continues it (``ingest_run``). Spans carry ids, stage names and
models only.

Durations and counts for the LLM are taken where ``LiteLLMClient`` already reads a call's tokens
for the usage meter, not by a second wrapper.
"""

from __future__ import annotations

import hmac
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import Response
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from graph_rag.logging import get_logger

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = [
    "METRICS",
    "add_api_observability",
    "check_metrics_settings",
    "configure_tracing",
    "current_traceparent",
    "enter_stage",
    "finish_ingest",
    "ingest_run",
    "llm_call",
    "metrics_response",
    "observe_llm_tokens",
    "use_tracer_provider",
]

_logger = get_logger(__name__)
_propagator = TraceContextTextMapPropagator()
_provider: Any = None  # set by configure_tracing; else the global (no-op) provider

_LATENCY = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_SLOW = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class _Metrics:
    """Every series the service exports."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "http_requests_total", "HTTP requests.", ["method", "route", "status"], registry=r
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "Time to the response (headers) by route template.",
            ["method", "route"],
            buckets=_LATENCY,
            registry=r,
        )
        self.stage_duration = Histogram(
            "ingest_stage_duration_seconds",
            "Time one pipeline stage took for one document.",
            ["stage"],
            buckets=_SLOW,
            registry=r,
        )
        self.stage_failures = Counter(
            "ingest_stage_failures_total",
            "Documents that failed in a stage.",
            ["stage"],
            registry=r,
        )
        self.documents = Counter(
            "ingest_documents_total", "Documents the pipeline finished.", ["outcome"], registry=r
        )
        self.llm_calls = Counter(
            "llm_calls_total",
            "Provider calls (retries included).",
            ["model", "outcome"],
            registry=r,
        )
        self.llm_duration = Histogram(
            "llm_call_duration_seconds",
            "Time one provider call took.",
            ["model"],
            buckets=_SLOW,
            registry=r,
        )
        self.llm_tokens = Counter(
            "llm_tokens_total", "Tokens the provider reported.", ["model"], registry=r
        )


METRICS = _Metrics()


# --- Tracing ---------------------------------------------------------------


def use_tracer_provider(provider: Any) -> None:
    """Use ``provider`` for this module's spans (tests pass one with an in-memory exporter)."""
    global _provider
    _provider = provider


def _tracer() -> trace.Tracer:
    return (_provider or trace.get_tracer_provider()).get_tracer("graph_rag")


def _keep_other_exporters_off() -> None:
    """Stop libraries that read ``OTEL_*`` from starting exporters of their own (GRA-153).

    FastAPI's built-in telemetry and the OpenTelemetry SDK's environment setup create OTLP trace,
    metric and log exporters, with periodic readers that retry ``localhost:4318``, when told to. We
    export spans in one place (:func:`configure_tracing`), so turn the rest off unless the
    operator chose otherwise. Call before the app is built: FastAPI reads this at construction.
    """
    os.environ.setdefault("FASTAPI_OTEL_AUTO_CONFIGURE", "false")
    for signal in ("TRACES", "METRICS", "LOGS"):
        os.environ.setdefault(f"OTEL_{signal}_EXPORTER", "none")


def configure_tracing(settings: Settings, service: str) -> bool:
    """Export spans over OTLP/HTTP when an endpoint is set. True when export is on.

    Needs the ``otel`` extra; without it, or with no endpoint, spans stay no-ops: no exporter, no
    reader, no thread, no connection. Only spans are exported, never OTLP metrics (those are
    Prometheus's ``/metrics``). Call it in every process entry point before building the app.
    """
    _keep_other_exporters_off()
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found,unused-ignore]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but the 'otel' extra is not installed")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    use_tracer_provider(provider)
    _logger.info("exporting traces as %s", service)
    return True


def current_traceparent() -> str | None:
    """The W3C ``traceparent`` of the span in progress, to carry on a message; else ``None``."""
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier.get("traceparent")


def _parent(traceparent: str | None) -> otel_context.Context | None:
    if not traceparent:
        return None
    return _propagator.extract({"traceparent": traceparent})


# --- The ingest pipeline: one span per document, one per stage -------------


class _Run:
    """One document's passage through the stages, timed and traced."""

    def __init__(self, span: trace.Span) -> None:
        self.span = span
        self.stage: str | None = None
        self.started = 0.0
        self._stage_span: trace.Span | None = None
        self._token: object | None = None
        self.finished = False

    def enter(self, stage: str) -> None:
        self._close_stage(failed=False)
        self.stage, self.started = stage, time.perf_counter()
        self._stage_span = _tracer().start_span(f"ingest.{stage}")
        self._token = otel_context.attach(trace.set_span_in_context(self._stage_span))

    def _close_stage(self, *, failed: bool) -> None:
        if self.stage is None or self._stage_span is None:
            return
        METRICS.stage_duration.labels(self.stage).observe(time.perf_counter() - self.started)
        if failed:
            METRICS.stage_failures.labels(self.stage).inc()
            self._stage_span.set_status(Status(StatusCode.ERROR))
        self._stage_span.end()
        if self._token is not None:
            otel_context.detach(self._token)  # type: ignore[arg-type]
        self.stage, self._stage_span, self._token = None, None, None

    def finish(self, outcome: str) -> None:
        if self.finished:
            return
        self.finished = True
        self._close_stage(failed=outcome == "failed")
        METRICS.documents.labels(outcome).inc()
        if outcome == "failed":
            self.span.set_status(Status(StatusCode.ERROR))


_run: ContextVar[_Run | None] = ContextVar("ingest_run", default=None)


@contextmanager
def ingest_run(traceparent: str | None) -> Iterator[None]:
    """Trace one document, continuing the trace that ``traceparent`` names (from the API)."""
    span = _tracer().start_span(
        "ingest.document", context=_parent(traceparent), kind=SpanKind.CONSUMER
    )
    token = otel_context.attach(trace.set_span_in_context(span))
    run = _Run(span)
    marker = _run.set(run)
    try:
        yield
    finally:
        run._close_stage(failed=not run.finished)  # an exit with a stage open is a failure
        _run.reset(marker)
        span.end()
        otel_context.detach(token)


def enter_stage(stage: str) -> None:
    """The document in this thread has entered ``stage``; the one before it is over."""
    if (run := _run.get()) is not None:
        run.enter(stage)


def finish_ingest(outcome: str) -> None:
    """The document in this thread is ``done`` or ``failed``."""
    if (run := _run.get()) is not None:
        run.finish(outcome)


# --- LLM calls --------------------------------------------------------------


@contextmanager
def llm_call(model: str) -> Iterator[None]:
    """Span and count one provider call, under whatever stage span is current."""
    started = time.perf_counter()
    outcome = "ok"
    with _tracer().start_as_current_span("llm.call", attributes={"llm.model": model}) as span:
        try:
            yield
        except BaseException:
            outcome = "error"
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            METRICS.llm_calls.labels(model, outcome).inc()
            METRICS.llm_duration.labels(model).observe(time.perf_counter() - started)


def observe_llm_tokens(model: str, tokens: int) -> None:
    if tokens > 0:
        METRICS.llm_tokens.labels(model).inc(tokens)


# --- HTTP: request metrics, spans, /metrics ---------------------------------


def check_metrics_settings(settings: Settings) -> None:
    """Refuse a configuration that would serve ``/metrics`` to tenants.

    Raises:
        ValueError: With auth on and metrics on but no ``METRICS_TOKEN``.
    """
    if settings.metrics_enabled and settings.require_auth and not settings.metrics_token:
        raise ValueError("METRICS_ENABLED with REQUIRE_AUTH needs METRICS_TOKEN (not an API key)")


def metrics_response(settings: Settings, authorization: str | None) -> tuple[int, bytes, str]:
    """``(status, body, content type)`` for a scrape carrying this ``Authorization`` header."""
    if not settings.metrics_enabled:
        return 404, b"", "text/plain"
    if settings.require_auth or settings.metrics_token:
        scheme, _, token = (authorization or "").partition(" ")
        good = scheme.lower() == "bearer" and hmac.compare_digest(
            token.strip().encode(), settings.metrics_token.encode()
        )
        if not good or not settings.metrics_token:  # no token configured: no one is let in
            return 401, b"", "text/plain"
    return 200, generate_latest(METRICS.registry), CONTENT_TYPE_LATEST


def _route_of(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def add_api_observability(app: FastAPI, settings: Settings) -> None:
    """Count and trace every request, and serve ``/metrics`` when enabled.

    Added last so it is the outermost middleware and sees the 401s, 413s and 429s the inner ones
    produce. A streamed response (``/jobs/{id}/events``) is timed to its first byte.
    """
    check_metrics_settings(settings)

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        if request.url.path == "/metrics":
            return await call_next(request)  # type: ignore[no-any-return]
        started = time.perf_counter()
        method = request.method
        status = 500
        with _tracer().start_as_current_span(
            method,
            context=_parent(request.headers.get("traceparent")),
            kind=SpanKind.SERVER,
        ) as span:
            try:
                response: Response = await call_next(request)
                status = response.status_code
                return response
            finally:
                route = _route_of(request)
                span.update_name(f"{method} {route}")
                span.set_attribute("http.route", route)
                span.set_attribute("http.status_code", status)
                if status >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                METRICS.http_requests.labels(method, route, str(status)).inc()
                METRICS.http_duration.labels(method, route).observe(time.perf_counter() - started)

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        def metrics(request: Request) -> Response:
            code, body, kind = metrics_response(settings, request.headers.get("authorization"))
            return Response(body, status_code=code, media_type=kind)
