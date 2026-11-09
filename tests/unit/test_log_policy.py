"""The log policy (GRA-141, docs/LOG-POLICY.md): no content at info level or above.

Content is what a caller sent or got back: question text, document text, source names and
object keys, answers. Counts, sizes, ids and tenant names may be logged.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from graph_rag.config import Settings
from graph_rag.fakes import InMemoryDocumentStore
from graph_rag.offline_server import build_app

QUESTION = "Where does Zebulon Quillfeather keep the secret ledger?"
SOURCE = "diary-of-zebulon-quillfeather.txt"
TEXT = "Zebulon Quillfeather works for Quillco. Quillco is based in Lisbon."
SECRETS = ["Quillfeather", "Zebulon", "secret ledger", SOURCE, "Quillco", "Lisbon"]


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app(Settings()))


def _logged(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)


def test_ingest_query_and_forget_log_no_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        stored = client.post("/v1/remember", json={"text": TEXT, "source_id": SOURCE}).json()
        client.post("/ingest", files={"file": (SOURCE, TEXT.encode())}, data={"source_url": SOURCE})
        client.post("/query", json={"question": QUESTION})
        client.post("/v1/recall", json={"question": QUESTION, "synthesize": True})
        client.delete(f"/documents/{stored['document_id']}")
    text = _logged(caplog)
    assert text, "expected some info logging to check"
    for secret in SECRETS:
        assert secret not in text


def test_a_failed_document_logs_no_content(
    client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("store broke")

    monkeypatch.setattr(InMemoryDocumentStore, "upsert", boom)
    with caplog.at_level(logging.INFO):
        client.post("/ingest", files={"file": (SOURCE, TEXT.encode())}, data={"source_url": SOURCE})
    assert any("dropping document" in r.getMessage() for r in caplog.records)
    for secret in SECRETS:
        assert secret not in _logged(caplog)
