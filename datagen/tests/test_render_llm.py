"""The LLM renderer keeps exact gold, caches every answer, and never trusts the model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from kggen.llm import CachedChat, CacheMiss, ResponseCache
from kggen.render import RenderSpec
from kggen.render_llm import InvalidRendering, render_documents_llm
from kggen.spec import WorldSpec
from kggen.world import generate
from llm_fake import honest_answer


@pytest.fixture(scope="module")
def world():
    return generate(WorldSpec(n_entities=40, avg_degree=3, seed=3, alias_rate=0.5, churn_rate=0.2))


def test_cold_run_builds_documents_with_exact_gold(world, endpoint, tmp_path: Path) -> None:
    chat = CachedChat(ResponseCache(tmp_path), endpoint())
    docs = render_documents_llm(world, chat)
    assert chat.calls == len(docs) > 0
    assert sorted(f for d in docs for s in d.sentences for f in s.fact_ids) == sorted(
        e.id for e in world.edges
    )
    for d in docs:
        assert d.text == " ".join(s.text for s in d.sentences)
        for s in d.sentences:
            assert d.text[s.start : s.end] == s.text
            assert len(s.mentions) == 2
            for m in s.mentions:
                assert d.text[m.start : m.end] == m.surface


def test_warm_run_makes_no_calls_and_needs_no_endpoint(world, endpoint, tmp_path: Path) -> None:
    client = endpoint()
    cold = render_documents_llm(world, CachedChat(ResponseCache(tmp_path), client))
    warm_chat = CachedChat(ResponseCache(tmp_path), None, model="test-model")
    assert render_documents_llm(world, warm_chat) == cold
    assert warm_chat.calls == 0 and warm_chat.hits == len(cold)
    assert len(endpoint.served) == len(cold)  # type: ignore[attr-defined]


def test_cache_records_model_and_prompt(world, endpoint, tmp_path: Path) -> None:
    render_documents_llm(world, CachedChat(ResponseCache(tmp_path), endpoint()))
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["model"] == "test-model"
    assert record["prompt_version"] == 1
    assert [m["role"] for m in record["messages"]] == ["system", "user"]
    assert "F1:" in record["messages"][1]["content"]


def test_a_different_render_seed_asks_again(world, endpoint, tmp_path: Path) -> None:
    chat = CachedChat(ResponseCache(tmp_path), endpoint())
    render_documents_llm(world, chat, RenderSpec(seed=1))
    first = chat.calls
    render_documents_llm(world, chat, RenderSpec(seed=2, alias_use_rate=1.0))
    assert chat.calls > first


def test_cache_only_run_with_an_empty_cache_fails_loudly(world, tmp_path: Path) -> None:
    chat = CachedChat(ResponseCache(tmp_path), None, model="test-model")
    with pytest.raises(CacheMiss):
        render_documents_llm(world, chat)


@pytest.mark.parametrize(
    "answer",
    [
        lambda p: "I cannot help with that.",
        lambda p: json.dumps({"sentences": []}),
        lambda p: honest_answer(p).replace("F1", "F9"),
        lambda p: re.sub(r"\b\d{4}\b", "", honest_answer(p)),
    ],
    ids=["prose", "empty", "wrong-label", "years-dropped"],
)
def test_untrustworthy_answers_are_rejected_and_not_cached(
    world, endpoint, tmp_path: Path, answer
) -> None:
    chat = CachedChat(ResponseCache(tmp_path), endpoint(answer))
    with pytest.raises(InvalidRendering):
        render_documents_llm(world, chat)
    # earlier documents with nothing to reject may be cached; the rejected response never is
    assert len(list(tmp_path.glob("*.json"))) == chat.calls - 1


def test_names_must_appear_whole_and_not_overlap(world, endpoint, tmp_path: Path) -> None:
    def misnamed(prompt: str) -> str:
        data = json.loads(honest_answer(prompt))
        for item in data["sentences"]:
            item["text"] = "Somebody did something."
        return json.dumps(data)

    chat = CachedChat(ResponseCache(tmp_path), endpoint(misnamed))
    with pytest.raises(InvalidRendering, match="names"):
        render_documents_llm(world, chat)


def test_code_fenced_json_is_accepted(world, endpoint, tmp_path: Path) -> None:
    chat = CachedChat(
        ResponseCache(tmp_path), endpoint(lambda p: f"```json\n{honest_answer(p)}\n```")
    )
    assert render_documents_llm(world, chat)


def test_cli_cold_then_warm_with_no_endpoint(endpoint, tmp_path: Path, capsys) -> None:
    from kggen.cli import main

    client = endpoint()
    common = ["docs", "generate", "--preset", "tiny", "--renderer", "llm", "--model", "test-model"]
    common += ["--cache-dir", str(tmp_path / "cache")]
    cold, warm = tmp_path / "cold.json", tmp_path / "warm.json"
    assert main([*common, "--base-url", client.base_url, "--out", str(cold)]) == 0
    assert "0 cache hits" in capsys.readouterr().err
    assert main([*common, "--out", str(warm)]) == 0
    err = capsys.readouterr().err
    assert "0 endpoint calls" in err
    assert warm.read_text() == cold.read_text()


def test_cli_llm_without_cache_dir_or_model_is_a_clear_error() -> None:
    from kggen.cli import main

    with pytest.raises(SystemExit, match="--cache-dir and --model"):
        main(["docs", "generate", "--preset", "tiny", "--renderer", "llm"])


def test_committed_tiny_cache_replays_with_no_calls() -> None:
    from kggen.spec import load_preset

    cache = Path(__file__).parents[1] / "examples" / "llm-cache-tiny"
    chat = CachedChat(ResponseCache(cache), None, model="claude-haiku")
    docs = render_documents_llm(generate(load_preset("tiny")), chat)
    assert chat.calls == 0 and chat.hits == len(docs) == 22
