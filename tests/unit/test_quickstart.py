"""The commands in docs/QUICKSTART.md work against the app ``graph-rag-offline`` serves (GRA-41)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from graph_rag.auth import ApiKeys, JsonFileKeyStore
from graph_rag.cli import main
from graph_rag.config import Settings
from graph_rag.offline_server import build_app

ROOT = Path(__file__).resolve().parents[2]
DOC = (ROOT / "docs" / "QUICKSTART.md").read_text()


def cli(client: TestClient, capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv), client=client)
    return code, capsys.readouterr().out


def test_the_documented_remember_then_recall_gives_london_with_evidence(capsys) -> None:
    client = TestClient(build_app(), base_url="http://127.0.0.1:8000")
    text = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."
    code, out = cli(client, capsys, "remember", text, "--source-id", "note-1")
    assert code == 0 and out.startswith("accepted: note-1")
    # no waiting step: the offline service reads a document as it is stored
    code, out = cli(client, capsys, "recall", "Where is Acme Corp based?")
    assert code == 0 and "answer: London" in out
    assert "Acme Corp is based in London." in out and "characters 34-63" in out
    assert cli(client, capsys, "lookup", "--name", "Acme Corp")[0] == 0
    assert cli(client, capsys, "forget", "--source-id", "note-1")[0] == 0
    _, out = cli(client, capsys, "recall", "Where is Acme Corp based?")
    assert "sources:" not in out


def test_the_documented_key_step_turns_401_into_a_working_recall(tmp_path: Path, capsys) -> None:
    keys = ApiKeys(JsonFileKeyStore(tmp_path / "keys.json"))
    secret, _ = keys.create("my-team", "laptop")
    app = build_app(Settings(require_auth=True, api_keys_path=str(tmp_path / "keys.json")))
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    code, _ = cli(client, capsys, "recall", "Where is Acme Corp based?")
    assert code == 1
    client.headers["Authorization"] = f"Bearer {secret}"
    assert cli(client, capsys, "recall", "Where is Acme Corp based?")[0] == 0


def test_every_command_the_doc_runs_is_an_installed_script() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    for name in set(re.findall(r"uv run (graph[\w-]*)", DOC)):
        assert name in scripts, f"{name} is in the quickstart but is not a script"
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for extra in re.findall(r"--extra (\w+)", DOC):
        assert extra in extras


def test_the_documented_flags_exist() -> None:
    from graph_rag.cli import _parser

    for argv in (
        ["recall", "q", "--token-budget", "5", "--namespace", "p"],
        ["--json", "recall", "q"],
        ["forget", "--source-id", "x"],
        ["lookup", "--name", "x"],
    ):
        _parser().parse_args(argv)
    assert "--token-budget 500" in DOC and "--namespace my-project" in DOC
