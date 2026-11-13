"""Ingest is refused up front when no LLM is configured, and LLM failures read well (GRA-154)."""

from __future__ import annotations

import logging
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.llm_config import explain_llm_error, llm_not_configured, log_if_llm_missing
from graph_rag.models import IngestTrigger
from graph_rag.offline_server import build_app
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue

FAKE_KEY = "not-a-real-key-just-a-word"
NOTHING: dict[str, str] = {}


def settings(**kwargs: object) -> Settings:
    """Settings that ignore any ``.env`` file, so a developer's key cannot leak into a test."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


# --- the rule ---------------------------------------------------------------------------------


def test_nothing_configured_for_an_openai_model_is_reported() -> None:
    message = llm_not_configured(settings(), NOTHING)
    assert message is not None
    assert "OPENAI_API_KEY" in message and "OPENAI_API_BASE" in message
    assert "docs/QUICKSTART.md" in message


def test_a_key_in_settings_or_the_environment_counts() -> None:
    assert llm_not_configured(settings(openai_api_key=FAKE_KEY), NOTHING) is None
    assert llm_not_configured(settings(), {"OPENAI_API_KEY": FAKE_KEY}) is None


def test_an_openai_compatible_endpoint_counts_even_without_a_key() -> None:
    assert llm_not_configured(settings(), {"OPENAI_API_BASE": "http://localhost:11434/v1"}) is None


def test_blank_values_do_not_count() -> None:
    assert llm_not_configured(settings(openai_api_key="  "), {"OPENAI_API_BASE": ""}) is not None


@pytest.mark.parametrize("model", ["anthropic/claude-3-haiku", "ollama/llama3", "mistral-small"])
def test_a_model_from_another_provider_is_never_refused(model: str) -> None:
    other = settings(coref_model=model, kg_build_model=model)
    assert llm_not_configured(other, NOTHING) is None


def test_one_openai_stage_model_is_enough_to_refuse() -> None:
    mixed = settings(coref_model="ollama/llama3", kg_build_model="openai/gpt-4o-mini")
    assert llm_not_configured(mixed, NOTHING) is not None


def test_tenant_keys_mode_leaves_the_check_to_the_per_tenant_rule() -> None:
    assert llm_not_configured(settings(require_tenant_llm_key=True), NOTHING) is None


# --- the endpoints ----------------------------------------------------------------------------


def app_client(ready: str | None) -> tuple[TestClient, Queue, Any]:
    components = build_offline_components()
    queue = Queue()
    app = create_app(components.object_store, queue, Settings(), llm_ready=lambda: ready)
    return TestClient(app), queue, components


def test_ingest_and_remember_refuse_before_storing_or_queueing() -> None:
    http, queue, components = app_client("no LLM; set OPENAI_API_KEY or OPENAI_API_BASE")
    upload = http.post("/ingest", files={"file": ("a.txt", b"Ada met Bob.")})
    assert upload.status_code == 503
    assert "OPENAI_API_KEY" in upload.json()["detail"]

    remember = http.post("/v1/remember", json={"text": "Ada met Bob.", "source_id": "n"})
    assert remember.status_code == 503
    error = remember.json()["error"]
    assert error["code"] == "unavailable"
    assert error["details"] == {"reason": "llm_not_configured"}
    assert "OPENAI_API_BASE" in error["message"]

    assert queue.triggers == []
    for key in ("a.txt", "n"):
        with pytest.raises(FileNotFoundError):
            components.object_store.get(DEFAULT_SCOPE, components.bucket, key)


def test_a_url_ingest_is_refused_before_it_is_fetched() -> None:
    from tests.unit.test_ingest_sources import FakeFetcher

    components = build_offline_components()
    fetcher = FakeFetcher(error="should not be reached")
    app = create_app(
        components.object_store, Queue(), Settings(), llm_ready=lambda: "nope", fetcher=fetcher
    )
    refused = TestClient(app).post("/ingest", data={"url": "https://example.test/a"})
    assert refused.status_code == 503


def test_a_refused_ingest_is_not_metered() -> None:
    from graph_rag.usage import InMemoryUsageStore, UsageMeter

    store = InMemoryUsageStore()
    components = build_offline_components()
    app = create_app(
        components.object_store,
        Queue(),
        Settings(),
        meter=UsageMeter(store),
        llm_ready=lambda: "nope",
    )
    TestClient(app).post("/v1/remember", json={"text": "Ada met Bob."})
    assert store.get("default").documents == 0


def test_a_configured_llm_is_accepted() -> None:
    http, queue, _ = app_client(None)
    assert http.post("/ingest", files={"file": ("a.txt", b"Ada met Bob.")}).status_code == 200
    assert http.post("/v1/remember", json={"text": "Bob met Cy."}).status_code == 202
    assert len(queue.triggers) == 2


def test_reading_and_forgetting_never_need_an_llm() -> None:
    components = build_offline_components()
    app = create_app(
        components.object_store,
        Queue(),
        Settings(),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
        llm_ready=lambda: "nope",
    )
    http = TestClient(app)
    assert http.post("/v1/recall", json={"question": "Who is Ada?"}).status_code == 200
    forgotten = http.post("/v1/forget", json={"source_id": "x"})
    assert forgotten.json()["error"]["code"] == "not_found"  # asked, not refused as unconfigured
    assert http.post("/query", json={"question": "Who is Ada?"}).status_code == 200


def test_the_default_wiring_does_not_check_so_fakes_and_the_fast_suite_need_no_key() -> None:
    components = build_offline_components()
    queue = Queue()
    http = TestClient(create_app(components.object_store, queue, Settings()))
    assert http.post("/v1/remember", json={"text": "Ada met Bob."}).status_code == 202
    assert len(queue.triggers) == 1


def test_the_offline_server_ingests_with_no_key_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    http = TestClient(build_app(settings()))
    assert http.post("/v1/remember", json={"text": "Ada met Bob."}).status_code == 202


# --- the worker -------------------------------------------------------------------------------


def test_the_worker_logs_one_error_when_it_would_fail_every_document(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("worker-test")
    with caplog.at_level(logging.ERROR, logger="worker-test"):
        assert log_if_llm_missing(settings(), logger, NOTHING) is True
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert "OPENAI_API_KEY" in caplog.records[0].getMessage()
    caplog.clear()
    assert log_if_llm_missing(settings(openai_api_key=FAKE_KEY), logger, NOTHING) is False
    assert caplog.records == []


# --- the job error ----------------------------------------------------------------------------


def stub(kind: type[Exception], **kwargs: object) -> Exception:
    return kind(message=f"boom with {FAKE_KEY} at http://user:pw@llm.internal/v1", **kwargs)  # type: ignore[call-arg]


def test_a_rejected_key_reads_as_such_and_leaks_nothing() -> None:
    error = stub(litellm.AuthenticationError, llm_provider="openai", model="gpt-4o-mini")
    why = explain_llm_error(error, NOTHING)
    assert why is not None and "rejected the API key" in why
    assert FAKE_KEY not in why and "llm.internal" not in why


def test_permission_denied_reads_as_a_rejected_key() -> None:
    error = litellm.PermissionDeniedError(
        message="no", llm_provider="openai", model="m", response=_response(403)
    )
    assert "rejected the API key" in (explain_llm_error(error, NOTHING) or "")


def _response(status: int) -> object:
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "http://llm.internal/v1"))


def test_an_unreachable_endpoint_names_the_host_only() -> None:
    error = stub(litellm.APIConnectionError, llm_provider="openai", model="gpt-4o-mini")
    env = {"OPENAI_API_BASE": "http://user:pw@llm.internal:8000/v1?token=abc"}
    why = explain_llm_error(error, env)
    assert why == "The LLM endpoint at llm.internal is unreachable (check OPENAI_API_BASE)."
    assert "pw" not in why and "token" not in why
    assert "unreachable" in (explain_llm_error(error, NOTHING) or "")


def test_a_timeout_is_an_unreachable_endpoint() -> None:
    error = litellm.Timeout(message="slow", model="m", llm_provider="openai")
    assert "unreachable" in (explain_llm_error(error, NOTHING) or "")


def test_missing_credentials_surfaced_as_an_internal_error_is_explained() -> None:
    error = litellm.InternalServerError(
        message="Missing credentials. Please pass an api_key", llm_provider="openai", model="m"
    )
    assert "No LLM API key" in (explain_llm_error(error, NOTHING) or "")


@pytest.mark.parametrize(
    "message",
    [
        "OpenAIException - Connection error.",
        "OpenAIException - [Errno 111] Connection refused",
        "[Errno -2] Name or service not known",
        "Temporary failure in name resolution",
        "Request timed out.",
        "HTTPConnectionPool(host='x', port=1): Max retries exceeded with url: /v1",
    ],
)
def test_a_refused_connection_arriving_as_an_internal_error_is_unreachable(message: str) -> None:
    error = litellm.InternalServerError(message=message, llm_provider="openai", model="m")
    env = {"OPENAI_API_BASE": "http://user:pw@llm.internal:8000/v1?token=abc"}
    why = explain_llm_error(error, env)
    assert why == "The LLM endpoint at llm.internal is unreachable (check OPENAI_API_BASE)."
    assert "pw" not in why and "token" not in why and "/v1" not in why


def test_the_chained_cause_is_inspected() -> None:
    error = litellm.InternalServerError(message="failed", llm_provider="openai", model="m")
    error.__cause__ = ConnectionRefusedError("[Errno 111] Connection refused")
    assert "unreachable" in (explain_llm_error(error, NOTHING) or "")


def test_a_genuine_provider_500_is_not_called_unreachable() -> None:
    error = litellm.InternalServerError(
        message="OpenAIException - The server had an error while processing your request",
        llm_provider="openai",
        model="m",
    )
    assert explain_llm_error(error, NOTHING) is None


@pytest.mark.parametrize("error", [RuntimeError("db down"), ValueError("bad"), KeyError("k")])
def test_other_errors_are_left_alone(error: Exception) -> None:
    assert explain_llm_error(error, NOTHING) is None


def test_the_job_error_keeps_the_stage_and_type_and_adds_the_reason() -> None:
    components = build_offline_components()
    queue = Queue()
    http = TestClient(
        create_app(components.object_store, queue, Settings(), jobs=components.job_store)
    )
    stored = http.post("/v1/remember", json={"text": "Ada met Bob.", "source_id": "n"}).json()

    class Rejecting:
        def analyze(self, text: str) -> object:
            raise litellm.AuthenticationError(
                message=f"bad {FAKE_KEY}", llm_provider="openai", model="m"
            )

    components.orchestrator._ner_stage = Rejecting()  # type: ignore[assignment]  # noqa: SLF001
    trigger: IngestTrigger = queue.triggers[0]
    assert components.orchestrator.process_document(trigger) is None
    job = http.get(f"/jobs/{stored['document_id']}").json()
    assert job["state"] == "failed"
    assert job["error"].startswith("entity recognition failed: AuthenticationError. ")
    assert "rejected the API key" in job["error"]
    assert FAKE_KEY not in job["error"]


def test_a_failed_llm_call_logs_one_warning_line_and_the_traceback_only_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    components = build_offline_components()
    queue = Queue()
    http = TestClient(
        create_app(components.object_store, queue, Settings(), jobs=components.job_store)
    )
    http.post("/v1/remember", json={"text": "Ada met Bob.", "source_id": "n"})

    class Rejecting:
        def analyze(self, text: str) -> object:
            raise litellm.AuthenticationError(
                message=f"bad {FAKE_KEY}", llm_provider="openai", model="m"
            )

    components.orchestrator._ner_stage = Rejecting()  # type: ignore[assignment]  # noqa: SLF001
    with caplog.at_level(logging.DEBUG, logger="graph_rag.orchestrator"):
        components.orchestrator.process_document(queue.triggers[0])
    loud = [r for r in caplog.records if r.levelno > logging.INFO]
    assert [r.levelno for r in loud] == [logging.WARNING]
    line = loud[0].getMessage()
    assert "entity recognition failed" in line and "rejected the API key" in line
    assert FAKE_KEY not in line and not loud[0].exc_info
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug and debug[0].exc_info is not None
