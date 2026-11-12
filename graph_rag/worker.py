"""The ingest worker: consumes triggers in its own process (GRA-31).

With ``INGEST_MODE=publish`` the API only publishes; ``graph-rag-worker`` runs the pipeline. Run
as many workers as you like (``docker compose up --scale worker=N``): they share one consumer
group. With ``TRIGGER_BACKEND=redis`` (the default) a per-document lock keeps two workers off one
document and a trigger a dead worker held is reclaimed; with ``kafka`` the partitions bound the
workers and a document's triggers share a partition (ADR-0016).

Health: a small HTTP server answers ``GET /healthz`` with 200 while the consume loop is running
and 503 once it has stopped or died. A long document does not fail it; the loop is alive, not
fast. ``SIGTERM`` and ``SIGINT`` stop the loop after the document in hand, then the consumer is
closed. The worker takes one message at a time so nothing fetched but unprocessed is committed
(kafka) or taken from the stream (redis).
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Protocol

from graph_rag.logging import get_logger

__all__ = ["run_worker", "main"]

_logger = get_logger(__name__)


class Consumer(Protocol):
    """What the worker needs from a trigger consumer."""

    def run_forever(self, poll_timeout_ms: int = ..., max_records: int | None = ...) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


# A scrape of ``GET /metrics``: takes the Authorization header, returns (status, body, type).
MetricsHandler = Callable[[str | None], tuple[int, bytes, str]]


def _health_server(
    port: int, alive: threading.Event, metrics: MetricsHandler | None = None
) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server's name
            if self.path == "/metrics" and metrics is not None:
                code, body, kind = metrics(self.headers.get("Authorization"))
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.end_headers()
                self.wfile.write(body)
                return
            ok = self.path == "/healthz" and alive.is_set()
            self.send_response(200 if ok else (404 if self.path != "/healthz" else 503))
            self.end_headers()
            self.wfile.write(b"ok" if ok else b"")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # a probe every few seconds is not worth a log line

    return HTTPServer(("0.0.0.0", port), Handler)  # noqa: S104 — the container's own probe


def run_worker(
    consumer: Consumer,
    health_port: int | None = None,
    *,
    install_signals: bool = True,
    metrics: MetricsHandler | None = None,
) -> None:
    """Run ``consumer`` in this thread until it stops or a termination signal arrives."""
    alive = threading.Event()
    server = _health_server(health_port, alive, metrics) if health_port is not None else None
    if server is not None:
        threading.Thread(target=server.serve_forever, name="worker-health", daemon=True).start()
    if install_signals:

        def stop(signum: int, _frame: object) -> None:
            _logger.info("signal %d: finishing the document in hand, then stopping", signum)
            consumer.stop()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    alive.set()
    try:
        consumer.run_forever(max_records=1)
    finally:
        alive.clear()
        consumer.close()
        if server is not None:
            server.shutdown()
            server.server_close()
        _logger.info("worker stopped")


def main() -> None:
    """Build the real pipeline and consume triggers until told to stop."""
    from graph_rag.bind import enforce_credentials
    from graph_rag.config import Settings
    from graph_rag.logging import configure_logging
    from graph_rag.main import build_pipeline
    from graph_rag.messaging.factory import build_consumer
    from graph_rag.observability import check_metrics_settings, configure_tracing, metrics_response

    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    configure_tracing(settings, "graph-rag-worker")
    check_metrics_settings(settings)
    enforce_credentials(settings)
    if settings.ingest_mode != "publish":
        raise SystemExit(
            "The worker needs INGEST_MODE=publish, so job status is shared with the API."
        )
    pipeline = build_pipeline(settings)
    consumer = build_consumer(settings, pipeline.orchestrator.process_document)
    _logger.info("worker consuming triggers from %s", settings.trigger_backend)
    run_worker(
        consumer,
        settings.worker_health_port,
        metrics=lambda auth: metrics_response(settings, auth),
    )


if __name__ == "__main__":
    main()
