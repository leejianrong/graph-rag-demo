"""The arithmetic behind the ingest-cost report (GRA-55): tokens, dollars, slopes, timings."""

from __future__ import annotations

from pathlib import Path

import pytest
from kgbench import hard
from kgbench.ingestcost import (
    Checkpoint,
    Price,
    StageTokens,
    dollars_per_thousand,
    measure_tokens,
    per_thousand,
    slope_per_thousand,
    summarise_seconds,
    tap_tokens,
    to_markdown,
)

from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.config import Settings

ROOT = Path(__file__).resolve().parents[2]
HARD_TINY = ROOT / "datagen" / "examples" / "hard-tiny"
TINY_CACHE = ROOT / "bench" / "kgbench" / "data" / "llm-cache-hard-tiny"


def words(text: str) -> int:
    return len(text.split())


def test_the_tap_totals_each_stage_from_the_prompt_and_the_answer() -> None:
    tap, totals = tap_tokens(words)
    tap("coref", "one two three", "a b", True)
    tap("coref", "four five", "c", False)
    tap("kg_build", "x y z w", "q", True)

    assert totals["coref"] == StageTokens(requests=2, prompt=5, completion=3)
    assert totals["kg_build"] == StageTokens(requests=1, prompt=4, completion=1)


def test_dollars_are_tokens_times_the_price_per_million() -> None:
    price = Price(input_per_million=0.15, output_per_million=0.60)
    tokens = StageTokens(requests=1, prompt=2_000_000, completion=1_000_000)
    assert price.dollars(tokens) == pytest.approx(0.30 + 0.60)
    assert dollars_per_thousand({"a": tokens}, price, documents=500) == pytest.approx(1.80)
    assert per_thousand(7, documents=14) == 500


def test_the_slope_ignores_what_a_store_costs_before_it_holds_anything() -> None:
    points = [Checkpoint(n, 0.0, {"s": 1_000_000 + 2_000 * n}) for n in (10, 20, 40, 80)]
    assert slope_per_thousand(points, "s") == pytest.approx(2_000_000)
    with pytest.raises(ValueError):
        slope_per_thousand(points[:1], "s")


def test_a_store_that_stops_growing_has_no_positive_slope() -> None:
    points = [Checkpoint(n, 0.0, {"s": size}) for n, size in ((10, 900), (20, 1000), (40, 1000))]
    assert slope_per_thousand(points, "s") < 20_000


def test_seconds_are_summarised() -> None:
    s = summarise_seconds([1.0, 1.0, 2.0, 4.0])
    assert s["mean"] == 2.0 and s["median"] == 1.5 and s["docs_per_second"] == 0.5
    assert s["p95"] == 4.0


def test_the_client_taps_a_cache_hit_and_a_provider_answer(tmp_path) -> None:
    seen: list[tuple[str, str, bool]] = []
    client = LiteLLMClient(
        "m",
        cache_dir=tmp_path,
        completion_fn=lambda **kw: "answer",
        tap=lambda prompt, text, cached: seen.append((prompt, text, cached)),
    )
    assert client.complete("question") == "answer"  # provider
    assert client.complete("question") == "answer"  # cache
    assert seen == [("question", "answer", False), ("question", "answer", True)]
    assert client.calls == 1 and client.cache_hits == 1


def test_the_committed_hard_tiny_replay_taps_every_request_and_calls_no_provider() -> None:
    _spec, documents = hard.load(HARD_TINY)
    settings = Settings(
        coref_model="openai/claude-haiku",
        kg_build_model="openai/claude-haiku",
        llm_cache_dir=str(TINY_CACHE),
        openai_api_key="unused",
    )
    totals = measure_tokens(documents, settings, words, spacy=False)

    assert totals["coref"].requests == len(documents)
    assert 0 < totals["kg_build"].requests <= len(documents)
    assert all(t.prompt > t.completion > 0 for t in totals.values())


def test_the_report_gives_tokens_dollars_and_says_what_it_did_not_measure() -> None:
    totals = {"coref": StageTokens(10, 1_000_000, 100_000)}
    text = to_markdown(
        command="kgbench ingest-cost",
        documents=10,
        words=40,
        totals=totals,
        prices={"m": Price(0.15, 0.6)},
        tokenizer="words",
        document_tokens=100_000,
    )
    assert "$21.00" in text  # (1M*0.15 + 0.1M*0.6)/1e6 = 0.21 over 10 docs, x100
    assert "10% of the prompt tokens" in text and "Cached: $0" in text
