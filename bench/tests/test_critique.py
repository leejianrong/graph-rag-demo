"""The Hugging Face-style question critique (GRA-121)."""

from __future__ import annotations

import json

import pytest
from kgbench import critique
from kggen.llm import CachedChat, CacheMiss, ResponseCache

DOCS = [
    {"title": "A", "text": "Aa owns Bb.", "sentences": [{"fact_ids": ["F1"]}]},
    {"title": "B", "text": "Bb is in Cc.", "sentences": [{"fact_ids": ["F2"]}]},
    {"title": "C", "text": "Unrelated.", "sentences": [{"fact_ids": ["F9"]}]},
]
QUESTION = {
    "id": "Q1",
    "hops": 2,
    "type": "entity",
    "question": "Where is Aa's holding?",
    "supporting_facts": ["F1", "F2"],
    "accepted_answers": ["Cc"],
}


class Scripted:
    model = "m"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, messages):
        self.calls += 1
        return self.reply(messages[-1]["content"])


def chat(tmp_path, reply):
    client = Scripted(reply)
    return CachedChat(ResponseCache(tmp_path), client), client


def test_context_is_the_documents_holding_supporting_facts():
    context = critique.context_for(QUESTION, DOCS)
    assert "Aa owns Bb." in context and "Bb is in Cc." in context
    assert "Unrelated" not in context


def test_ratings_are_parsed_and_only_valid_ones_cached(tmp_path):
    replies = iter(["Evaluation: ok\nTotal rating: 5", "no number here", "Total rating: **4**"])
    c, client = chat(tmp_path, lambda _: next(replies))
    row = critique.critique(c, [QUESTION], DOCS, workers=1)[0]
    assert (row["groundedness"], row["relevance"], row["standalone"]) == (5, None, 4)
    assert critique.kept(row) is False  # a missing rating fails
    assert len(list(tmp_path.rglob("*.json"))) == 2  # the unparsable reply was not stored


def test_replay_needs_no_endpoint(tmp_path):
    c, _ = chat(tmp_path, lambda _: "Total rating: 4")
    first = critique.critique(c, [QUESTION], DOCS, workers=1)
    replay = CachedChat(ResponseCache(tmp_path), None, "m")
    assert critique.critique(replay, [QUESTION], DOCS, workers=1) == first
    with pytest.raises(CacheMiss):
        critique.critique(replay, [{**QUESTION, "question": "Other?"}], DOCS, workers=1)


def test_only_the_groundedness_prompt_carries_the_context(tmp_path):
    prompts: list[str] = []

    def reply(prompt):
        prompts.append(prompt)
        return "Total rating: 4"

    c, _ = chat(tmp_path, reply)
    critique.critique(c, [QUESTION], DOCS, workers=1)
    with_context = [p for p in prompts if "Aa owns Bb." in p]
    assert len(prompts) == 3 and len(with_context) == 1
    assert "answer the given question" in with_context[0]


def test_table_and_chart_report_pass_rates():
    base = {"type": "entity", "relevance": 5, "standalone": 5}
    rows = [
        {"id": "a", "hops": 1, "groundedness": 5, **base},
        {"id": "b", "hops": 2, "groundedness": 2, **base},
    ]
    table = critique.format_table(rows)
    assert "hops 2" in table and "0%" in table and "100%" in table
    assert critique.chart_svg(rows, "t").startswith("<svg")


def test_rows_are_tagged_from_the_datasets_flag():
    questions = [{"id": "a", "shared_name": True}, {"id": "b", "shared_name": False}]
    tagged = critique.tag_shared_names([{"id": "a"}, {"id": "b"}], questions)
    assert [r["shared_name"] for r in tagged] == ["yes", "no"]


def test_the_committed_hard_tiny_ratings_replay_offline():
    from pathlib import Path

    from kgbench.cli import _DATA

    folder = Path(__file__).resolve().parents[2] / "datagen" / "examples" / "hard-tiny"
    questions = json.loads((folder / "questions.json").read_text())
    documents = json.loads((folder / "documents.json").read_text())
    replay = CachedChat(ResponseCache(_DATA / "critique-cache-hard-tiny"), None, "claude-haiku")
    rows = critique.critique(replay, questions, documents)
    assert replay.calls == 0 and replay.hits >= 180
    assert sum((r["groundedness"] or 0) >= 4 for r in rows) == 45


def test_the_sample_is_even_across_hops_and_seeded():
    questions = [{"id": f"Q{i:02d}", "hops": 1 + i % 4} for i in range(40)]
    picked = critique.sample(questions, 10, seed=1)
    assert len(picked) == 10 and [q["hops"] for q in picked].count(1) == 3
    assert picked == critique.sample(questions, 10, seed=1)
    assert picked != critique.sample(questions, 10, seed=2)


def test_the_sheet_hides_the_ratings_and_agreement_counts_the_cells():
    sheet = critique.sheet([QUESTION], DOCS)
    assert "Aa owns Bb." in sheet and "Total rating" not in sheet
    rows = [
        {"id": "a", "groundedness": 5},
        {"id": "b", "groundedness": 2},
        {"id": "c", "groundedness": 5},
    ]
    labels = {"a": "answerable", "b": "ambiguous", "c": "not_answerable"}
    text = critique.agreement(rows, labels)
    assert "3 labelled" in text and "2/3" in text
