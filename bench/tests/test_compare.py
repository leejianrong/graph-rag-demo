"""The comparison runs every system on the same questions; means and spread are over seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kgbench.cli import main
from kgbench.compare import SYSTEMS, CachingEmbedder, Comparison, SeedRun, compare
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import HopScore, RetrievalReport
from kggen.spec import load_preset

from graph_rag.fakes import FakeEmbedder


@pytest.fixture(scope="module")
def result() -> Comparison:
    return compare(load_preset("small"), [0, 1], preset="small", n_questions=20, ks=(1, 5))


def test_every_system_is_scored_on_every_seed_with_the_same_questions(result) -> None:
    assert [r.seed for r in result.runs] == [0, 1]
    for run in result.runs:
        assert tuple(run.reports) == SYSTEMS
        counts = {
            tuple((h, s.questions) for h, s in sorted(rep.by_hops.items()))
            for rep in run.reports.values()
        }
        assert len(counts) == 1  # identical question sets: the same hop mix for every system
        assert run.questions == run.reports[SYSTEMS[0]].overall.questions == 20


def test_the_table_has_a_row_per_hop_count_and_a_column_per_system(result) -> None:
    table = result.format_table(5)
    assert all(system in table for system in SYSTEMS)
    rows = {line.split()[0] for line in table.splitlines()[2:]}
    assert rows == {"all", *map(str, result.hop_counts())}
    assert "mean over seeds [0, 1]" in table


def test_means_and_spread_are_computed_over_seeds() -> None:
    def run(seed: int, path: float) -> SeedRun:
        score = HopScore(10, {5: path}, {5: path})
        report = RetrievalReport((5,), score, {1: score})
        return SeedRun(
            seed, 1, 1, 1, 1, 10, {s: report for s in SYSTEMS}, {s: 0.0 for s in SYSTEMS}
        )

    made = Comparison("p", "e", (5,), (run(0, 0.2), run(1, 0.6)))
    assert made.mean("BM25", 1, 5, "path") == pytest.approx(0.4)
    assert made.spread("BM25", 1, 5, "path") == (0.2, 0.6)
    assert made.questions_at(1) == 20


def test_the_markdown_states_seeds_config_sample_size_and_how_to_reproduce(result) -> None:
    text = result.to_markdown(k=5, command="uv run kgbench compare --preset small")
    for needle in ("seeds [0, 1]", "world seeds", "questions per seed", "Reproduce:", "Caveats"):
        assert needle in text
    assert "uv run kgbench compare --preset small" in text
    assert f"{sum(r.questions for r in result.runs)} in total" in text
    assert "templates" in text and "floor" in text  # the synthetic-text and offline caveats
    assert text.count("| hops |") == 2  # the means table and the spread table
    assert "## Cost and latency" in text
    for system in SYSTEMS:
        assert text.count(f"| {system} |") == 3  # means, spread and cost rows
    assert text.count("| 0 | $0.00 |") == len(SYSTEMS)  # no system reaches a provider


def test_results_are_json_serialisable_and_carry_timing(result) -> None:
    data = json.loads(json.dumps(result.to_dict()))
    assert data["preset"] == "small" and len(data["seeds"]) == 2
    assert set(data["seeds"][0]["systems"]) == set(SYSTEMS)
    assert set(data["seeds"][0]["seconds"]) == set(SYSTEMS)


def test_the_comparison_is_deterministic() -> None:
    a = compare(load_preset("tiny"), [0], preset="tiny", n_questions=8, ks=(5,))
    b = compare(load_preset("tiny"), [0], preset="tiny", n_questions=8, ks=(5,))
    assert a.to_dict()["seeds"][0]["systems"] == b.to_dict()["seeds"][0]["systems"]


def test_the_shared_embedder_embeds_each_sentence_once() -> None:
    inner_calls: list[int] = []

    class Counting(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            inner_calls.append(len(texts))
            return super().embed(texts)

    cache = CachingEmbedder(Counting())
    cache.embed(["a", "b", "a"])
    cache.embed(["b", "c"])
    assert cache.embedded == 3 and sum(inner_calls) == 3  # a, b, c: each reached the model once
    assert cache.embed(["c", "a"]) == FakeEmbedder().embed(["c", "a"])
    assert cache.dim == FakeEmbedder().dim


def test_graph_rag_evidence_is_mapped_to_gold_sentences_with_fact_ids() -> None:
    from kggen.questions import QuestionSpec, generate_questions
    from kggen.render import render_documents
    from kggen.world import generate

    from graph_rag.wiring import build_offline_components

    world = generate(load_preset("tiny"))
    documents = render_documents(world)
    question = generate_questions(world, QuestionSpec(n_questions=4))[0]
    for mode in ("passages", "graph"):
        retriever = GraphRagRetriever(build_offline_components(), documents, mode=mode)
        ranking = retriever.retrieve(question.question, 5)
        assert 0 < len(ranking) <= 5
        assert len({(s.doc, s.text) for s in ranking}) == len(ranking)  # no repeated sentences
        assert any(s.fact_ids for s in ranking)  # evidence maps back to real facts


def test_compare_command_writes_markdown_and_json(tmp_path: Path, capsys) -> None:
    out, data = tmp_path / "r.md", tmp_path / "r.json"
    argv = ["compare", "--preset", "tiny", "--seeds", "0", "--questions", "8"]
    assert main([*argv, "--out", str(out), "--json", str(data)]) == 0
    assert "BM25" in capsys.readouterr().out
    assert "kgbench compare --preset tiny --seeds 0 --questions 8" in out.read_text()
    assert json.loads(data.read_text())["preset"] == "tiny"
