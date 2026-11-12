"""With no OTLP endpoint no OpenTelemetry data is exported, whoever installs the provider (GRA-153).

``transformers`` installs real SDK providers as the global ones when it is imported. Each case
runs in a subprocess (the API lets a global provider be set once per process) and does what it
does with in-memory readers and exporters, so no network is touched and ``transformers`` is not
needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent(
    """
    import json, sys
    from graph_rag import observability
    from graph_rag.config import Settings

    if sys.argv[1] == "configure":
        observability.configure_tracing(Settings(), "probe")

    # What transformers does at import: real SDK providers, set as the global ones.
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    reader, exporter = InMemoryMetricReader(), InMemorySpanExporter()
    meters = MeterProvider(metric_readers=[reader])
    tracers = TracerProvider()
    tracers.add_span_processor(SimpleSpanProcessor(exporter))
    metrics.set_meter_provider(meters)
    trace.set_tracer_provider(tracers)

    with observability._tracer().start_as_current_span("ours"):
        pass
    with trace.get_tracer("library").start_as_current_span("theirs"):
        pass
    metrics.get_meter("library").create_counter("c").add(1)
    data = reader.get_metrics_data()
    points = sum(
        len(m.data.data_points)
        for rm in (data.resource_metrics if data else [])
        for sm in rm.scope_metrics
        for m in sm.metrics
    )
    print(json.dumps({
        "spans": sorted(s.name for s in exporter.get_finished_spans()),
        "metric_points": points,
    }))
    """
)


def run_child(mode: str, **extra: str) -> dict[str, object]:
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    child_env.update(extra)
    out = subprocess.run(  # noqa: S603 - our own interpreter and script
        [sys.executable, "-c", CHILD, mode],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_with_no_endpoint_nothing_is_recorded_by_anyones_provider() -> None:
    assert run_child("configure") == {"spans": [], "metric_points": 0}


def test_without_the_setup_a_library_provider_records_its_own_data() -> None:
    """The old behaviour: the global providers a library installs do record (and would export)."""
    got = run_child("plain")
    assert got["metric_points"] == 1
    assert got["spans"] == ["theirs"]  # ours never goes through the global provider


def test_an_operator_value_wins() -> None:
    got = run_child("configure", OTEL_SDK_DISABLED="false")
    assert got["metric_points"] == 1
    assert got["spans"] == ["theirs"]


@pytest.mark.parametrize(
    "entry", ["graph_rag.main", "graph_rag.worker", "graph_rag.offline_server"]
)
def test_importing_an_entry_point_loads_no_model_library(entry: str) -> None:
    """``configure_tracing`` can run first because model libraries load lazily, after it."""
    heavy = ("transformers", "spacy", "sentence_transformers")
    code = f"import sys, {entry}; print(any(m in sys.modules for m in {heavy}))"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    ).stdout
    assert out.strip().splitlines()[-1] == "False"
