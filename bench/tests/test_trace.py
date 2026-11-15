"""The stage trace accounts for every question and its stages only narrow (GRA-170)."""

from __future__ import annotations

import json
from pathlib import Path

from kgbench import hard, trace

from graph_rag.fakes import FakeEmbedder
from graph_rag.wiring import build_offline_components

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "datagen" / "examples" / "hard-tiny"


def run() -> dict:
    _, documents = hard.load(FOLDER)
    questions = json.loads((FOLDER / "questions.json").read_text())
    embedder = FakeEmbedder()
    result = trace.trace(
        FOLDER, documents, questions, lambda: build_offline_components(embedder=embedder), embedder
    )
    assert sum(c["questions"] for c in result["hops"].values()) == len(questions)
    return result


def test_every_question_is_counted_once_at_the_first_stage_it_is_lost() -> None:
    result = run()

    for hop, counts in result["hops"].items():
        lost = result["first_loss"][hop]
        assert sum(lost.values()) == counts["questions"]
        assert lost.get("solved", 0) == counts["graph_solved"]


def test_a_path_cannot_survive_a_later_stage_it_failed_an_earlier_one() -> None:
    for counts in run()["hops"].values():
        assert (
            counts["questions"]
            >= counts["extracted"]
            >= counts["edges_right"]
            >= counts["path_in_subgraph"]
        )


def test_the_report_names_each_stage_and_the_hop_counts() -> None:
    text = trace.format_markdown(run(), command="kgbench trace", k=5)

    for stage_text in ("extraction or graph build", "linking", "seeding", "traversal", "ranking"):
        assert stage_text in text
    assert "| all |" in text
