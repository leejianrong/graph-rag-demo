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
    # a valid answer is cached as it arrives; a rejected response never is
    assert len(list(tmp_path.glob("*.json"))) < chat.calls


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


def rich_docs(world, endpoint, tmp_path, answer=None):
    from llm_fake import rich_answer

    chat = CachedChat(ResponseCache(tmp_path), endpoint(answer or rich_answer))
    return render_documents_llm(world, chat, style="rich")


def test_rich_style_keeps_exact_gold_with_pronouns_and_background(world, endpoint, tmp_path):
    docs = rich_docs(world, endpoint, tmp_path)
    assert sorted(f for d in docs for s in d.sentences for f in s.fact_ids) == sorted(
        e.id for e in world.edges
    )
    sentences = [s for d in docs for s in d.sentences]
    assert any(s.text.startswith("It ") and len(s.mentions) == 1 for s in sentences)  # a pronoun
    assert any(not s.fact_ids and not s.mentions for s in sentences)  # background prose
    for d in docs:
        assert d.text == " ".join(s.text for s in d.sentences)
        for s in d.sentences:
            for m in s.mentions:
                assert d.text[m.start : m.end] == m.surface


@pytest.mark.parametrize(
    ("break_it", "match"),
    [
        (lambda items: items[1:], "exactly once"),  # a fact never stated
        (lambda items: [items[0], *items], "exactly once"),  # a fact stated twice
        (
            lambda items: [
                {"facts": items[0]["facts"], "text": "It is related to Zzz."},
                *items[1:],
            ],
            "object",
        ),
    ],
)
def test_rich_style_rejects_unfaithful_articles(world, endpoint, tmp_path, break_it, match):
    from llm_fake import rich_answer

    def broken(prompt: str) -> str:
        data = json.loads(rich_answer(prompt))
        data["sentences"] = break_it(data["sentences"])
        return json.dumps(data)

    with pytest.raises(InvalidRendering, match=match):
        rich_docs(world, endpoint, tmp_path, broken)
    assert list(tmp_path.glob("*.json")) == []  # the first, rejected answer is not cached


def test_rich_style_rejects_a_pronoun_with_no_antecedent(world, endpoint, tmp_path):
    from llm_fake import rich_answer

    def unnamed(prompt: str) -> str:
        data = json.loads(rich_answer(prompt))
        first = data["sentences"][0]
        subject = re.search(r"^F1: (.*?) \|", prompt, re.M)[1]
        first["text"] = first["text"].replace(subject, "It", 1)
        return json.dumps(data)

    with pytest.raises(InvalidRendering, match="not named yet"):
        rich_docs(world, endpoint, tmp_path, unnamed)


def test_rich_style_rejects_naming_another_entity_of_the_world(world, endpoint, tmp_path):
    from llm_fake import rich_answer

    others = {e.name for e in world.entities}

    def leaks(prompt: str) -> str:
        data = json.loads(rich_answer(prompt))
        listed = set(re.findall(r"(?:^F\d+: | \| )([^|(]+?)(?= \||$| \()", prompt, re.M))
        stranger = next(n for n in sorted(others) if n not in listed)
        data["sentences"][-1]["text"] = f"{stranger} is unrelated."
        return json.dumps(data)

    with pytest.raises(InvalidRendering, match="in the world"):
        rich_docs(world, endpoint, tmp_path, leaks)


def test_committed_hard_dataset_replays_to_the_same_documents() -> None:
    from kggen.render import RenderSpec
    from kggen.spec import load_spec

    folder = Path(__file__).parents[1] / "examples" / "hard-small"
    chat = CachedChat(ResponseCache(folder / "cache"), None, model="claude-haiku")
    docs = render_documents_llm(
        generate(load_spec(folder / "spec.yaml")), chat, RenderSpec(), style="rich"
    )
    assert chat.calls == 0 and chat.hits == len(docs) == 145
    written = json.loads((folder / "documents.json").read_text())
    assert [d.to_dict() for d in docs] == written


def test_documents_round_trip_through_json() -> None:
    from kggen.render import Document

    folder = Path(__file__).parents[1] / "examples" / "hard-small"
    raw = json.loads((folder / "documents.json").read_text())
    docs = [Document.from_dict(d) for d in raw]
    assert [d.to_dict() for d in docs] == raw


def test_the_committed_hard_dataset_is_complete_and_current() -> None:
    from kggen.export import export_files
    from kggen.questions import QuestionSpec
    from kggen.render import Document, RenderSpec
    from kggen.spec import load_spec

    folder = Path(__file__).parents[1] / "examples" / "hard-small"
    docs = [Document.from_dict(d) for d in json.loads((folder / "documents.json").read_text())]
    questions = QuestionSpec.model_validate_json((folder / "questions_spec.json").read_text())
    made = export_files(generate(load_spec(folder / "spec.yaml")), questions, RenderSpec(), docs)
    for name in ("world.json", "questions.json", "documents.json", "world.stats.json"):
        assert made[name] == (folder / name).read_bytes(), f"{name} is stale: re-run kggen export"
    assert b"kept as given" in made["README.md"]
