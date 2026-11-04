"""Cost and latency are recorded with every result (GRA-87)."""

from __future__ import annotations

import json
import time

import pytest
from kgbench.cli import main
from kgbench.compare import SYSTEMS, compare
from kgbench.cost import Cost, Meter
from kggen.spec import load_preset
from pydantic import BaseModel

from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.wiring import LlmUsage, build_offline_components, usage_of


class _Shape(BaseModel):
    value: int


class _Components:
    """Just enough of ``BenchmarkComponents`` for a meter: a usage probe over real clients."""

    def __init__(self, *clients: LiteLLMClient) -> None:
        self.llm_usage = usage_of(*clients)


def _client(tmp_path) -> LiteLLMClient:
    def backend(*, model: str, prompt: str, **params: object) -> str:
        time.sleep(0.01)
        return '{"value": 1}'

    return LiteLLMClient("m", cache_dir=tmp_path, completion_fn=backend)


def test_a_cold_run_is_charged_and_a_warm_rerun_reads_zero_calls(tmp_path) -> None:
    client = _client(tmp_path)
    components = _Components(client)
    prompts = ["a", "b", "c"]

    with Meter(components) as cold:
        for p in prompts:
            client.structured(p, _Shape)
    with Meter(components) as warm:
        for p in prompts:
            client.structured(p, _Shape)

    assert cold.cost.llm.calls == 3 and cold.cost.llm.cache_hits == 0
    assert cold.cost.llm.seconds >= 0.03
    assert warm.cost.llm.calls == 0 and warm.cost.llm.cache_hits == 3  # served from the cache
    assert warm.cost.llm.seconds == 0.0


def test_usage_sums_every_client_and_counts_cache_hits_on_both_entry_points(tmp_path) -> None:
    coref, kg = _client(tmp_path / "a"), _client(tmp_path / "b")
    coref.complete("x")
    coref.complete("x")  # free text, second time: a hit
    kg.structured("y", _Shape)
    kg.structured("y", _Shape)
    assert usage_of(coref, kg)() == LlmUsage(2, 2, coref.seconds + kg.seconds, 0.0)


def test_a_meter_reports_only_what_happened_inside_it(tmp_path) -> None:
    client = _client(tmp_path)
    client.complete("before")  # not part of the measured run
    with Meter(_Components(client)) as meter:
        client.complete("during")
    assert meter.cost.llm.calls == 1


def test_a_system_with_no_pipeline_costs_no_provider_calls() -> None:
    with Meter(None) as meter:
        time.sleep(0.01)
    assert meter.cost.wall_seconds >= 0.01 and meter.cost.llm == LlmUsage()


def test_the_offline_pipeline_never_reaches_a_provider() -> None:
    assert build_offline_components().llm_usage() == LlmUsage()


def test_cost_serialises_every_field() -> None:
    cost = Cost(1.23456, LlmUsage(3, 2, 0.5, 0.0123456789))
    assert cost.to_dict() == {
        "wall_seconds": 1.235,
        "llm_calls": 3,
        "llm_cache_hits": 2,
        "llm_seconds": 0.5,
        "llm_cost_usd": 0.012346,
    }
    assert "3 provider calls" in cost.format_line() and "$0.0123" in cost.format_line()


@pytest.fixture(scope="module")
def tiny_compare():
    return compare(load_preset("tiny"), [0], preset="tiny", n_questions=8, ks=(5,))


def test_every_compared_system_carries_cost_and_latency(tiny_compare) -> None:
    cost = json.loads(json.dumps(tiny_compare.to_dict()))["seeds"][0]["cost"]
    assert set(cost) == set(SYSTEMS)
    for system in SYSTEMS:
        assert cost[system]["wall_seconds"] > 0
        assert cost[system]["llm_calls"] == 0 and cost[system]["llm_cost_usd"] == 0


def test_run_and_extract_and_linking_json_carry_cost(tmp_path, capsys) -> None:
    out = tmp_path / "run.json"
    assert main(["run", "--json", str(out)]) == 0
    run = json.loads(out.read_text())
    assert {"wall_seconds", "llm_calls", "llm_cost_usd"} <= set(run["cost"])
    assert run["aggregate"]["supporting_fact_recall"] >= 0

    capsys.readouterr()
    assert main(["extract", "--preset", "tiny", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cost"]["wall_seconds"] > 0
    assert main(["linking", "--preset", "tiny", "--difficulty", "clean", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["clean"]["cost"]["wall_seconds"] > 0
