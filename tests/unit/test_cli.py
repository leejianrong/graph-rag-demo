"""The ``graphrag`` command (GRA-39), against the real API over the offline pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.cli import main
from graph_rag.config import Settings
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import TEXT, Queue, drain

QUESTION = "Where is Acme Corp based?"


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        retriever=components.retriever,
        eraser=components.orchestrator.delete_document,
    )
    return TestClient(app, base_url="http://graph-rag"), components, queue


def run(client, capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv), client=client)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_remember_recall_forget_round_trip(stack, capsys) -> None:
    client, components, queue = stack
    code, out, _ = run(client, capsys, "remember", TEXT, "--source-id", "note-1")
    assert code == 0 and out.startswith("accepted: note-1")
    drain(components, queue)

    code, out, _ = run(client, capsys, "recall", QUESTION)
    assert code == 0 and "answer:" in out and "sources:" in out
    assert "Acme Corp is based in London." in out and "characters" in out

    code, out, _ = run(client, capsys, "forget", "--source-id", "note-1")
    assert code == 0 and out.startswith("forgot note-1")
    _, out, _ = run(client, capsys, "recall", QUESTION)
    assert "sources:" not in out


def test_json_output_is_the_apis_json(stack, capsys) -> None:
    client, components, queue = stack
    _, out, _ = run(client, capsys, "--json", "remember", TEXT, "--source-id", "n")
    assert json.loads(out)["status"] == "accepted"
    drain(components, queue)
    _, out, _ = run(client, capsys, "--json", "recall", QUESTION)
    payload = json.loads(out)
    assert payload["supporting_sentences"] and "ranked_nodes" in payload
    _, out, _ = run(client, capsys, "--json", "forget", "--source-id", "n")
    assert json.loads(out)["source_id"] == "n"


def test_remember_reads_a_file_and_stdin(stack, capsys, tmp_path: Path, monkeypatch) -> None:
    client, _, queue = stack
    note = tmp_path / "note.txt"
    note.write_text("from a file")
    assert run(client, capsys, "remember", "--file", str(note))[0] == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert run(client, capsys, "remember", "-f", "-")[0] == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("also stdin"))
    assert run(client, capsys, "remember")[0] == 0
    assert len(queue.triggers) == 3


def test_namespaces_keep_memory_apart(stack, capsys) -> None:
    client, components, queue = stack
    run(client, capsys, "remember", TEXT, "--source-id", "n", "--namespace", "proj")
    drain(components, queue)
    _, out, _ = run(client, capsys, "recall", QUESTION, "--namespace", "proj")
    assert "sources:" in out
    _, out, _ = run(client, capsys, "recall", QUESTION)
    assert "sources:" not in out


def test_an_api_error_exits_1_with_its_message_on_stderr(stack, capsys) -> None:
    client, *_ = stack
    code, out, err = run(client, capsys, "forget", "--source-id", "missing")
    assert (code, out) == (1, "") and err.startswith("error: No source matches missing")
    code, _, err = run(client, capsys, "--json", "forget", "--source-id", "missing")
    assert code == 1 and json.loads(err)["error"]["code"] == "not_found"


def test_an_unreachable_service_exits_2(capsys) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(refuse))
    code, _, err = run(client, capsys, "recall", QUESTION)
    assert code == 2 and "could not reach" in err


def test_a_key_in_the_environment_is_sent_as_a_bearer_token(monkeypatch, capsys) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"answer": None, "supporting_sentences": []})

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setenv("GRAPH_RAG_API_KEY", "grk_abc_secret")
    assert main(["recall", QUESTION]) == 0
    monkeypatch.delenv("GRAPH_RAG_API_KEY")
    assert main(["recall", QUESTION]) == 0
    assert seen == ["Bearer grk_abc_secret", None]
    assert "(nothing found)" in capsys.readouterr().out


def test_against_a_service_that_requires_auth(capsys) -> None:
    components = build_offline_components()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        Queue(),
        Settings(require_auth=True),
        retriever=components.retriever,
        keys=keys,
    )
    code, _, err = run(TestClient(app, base_url="http://graph-rag"), capsys, "recall", QUESTION)
    assert code == 1 and "API key" in err


def test_usage_errors_exit_2_and_help_covers_every_command(capsys) -> None:
    for argv in (["forget"], ["remember", "x", "--file", "y"], []):
        with pytest.raises(SystemExit) as raised:
            main(argv)
        assert raised.value.code in (2, "remember: give the text or --file, not both")
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["--help"])
    top = capsys.readouterr().out
    for word in ("remember", "recall", "forget", "GRAPH_RAG_API_KEY", "exit status", "--json"):
        assert word in top
    for command in ("remember", "recall", "forget"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert "usage: graphrag " + command in capsys.readouterr().out


def test_lookup_shows_neighbours_and_misses_exit_1(capsys) -> None:
    from graph_rag.lookup import EntityLookup

    components = build_offline_components()
    queue = Queue()
    app = create_app(
        components.object_store,
        queue,
        Settings(),
        lookup=EntityLookup(components.entity_store, components.graph_store),
    )
    client = TestClient(app, base_url="http://graph-rag")
    run(client, capsys, "remember", TEXT, "--source-id", "n")
    drain(components, queue)

    code, out, _ = run(client, capsys, "lookup", "--name", "Acme Corp")
    assert code == 0 and "Acme Corp [ORG]" in out and ("->" in out or "<-" in out)
    assert "Acme Corp is based in London." in out
    _, out, _ = run(client, capsys, "--json", "lookup", "--name", "acme corp", "--limit", "1")
    assert len(json.loads(out)["entities"][0]["neighbours"]) == 1
    code, _, err = run(client, capsys, "lookup", "--name", "Nobody Inc")
    assert code == 1 and "No entity matches" in err


def test_recall_takes_a_token_budget(stack, capsys) -> None:
    client, components, queue = stack
    text = " ".join(
        f"Person{i} works for Acme Corp. Acme Corp is based in London." for i in range(8)
    )
    run(client, capsys, "remember", text, "--source-id", "n")
    drain(components, queue)
    _, full, _ = run(client, capsys, "--json", "recall", QUESTION)
    code, out, _ = run(client, capsys, "--json", "recall", QUESTION, "--token-budget", "60")
    assert code == 0 and json.loads(out)["tokens_used"] <= 60 < json.loads(full)["tokens_used"]
    _, out, _ = run(client, capsys, "recall", QUESTION, "--token-budget", "60")
    assert "evidence cut to fit the budget" in out
    code, _, err = run(client, capsys, "recall", QUESTION, "--token-budget", "0")
    assert code == 1 and "token_budget" in err
