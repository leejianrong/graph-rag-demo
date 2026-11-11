"""LongMemEval adapter: loading, ingest order, per-type scoring, and the CLI (GRA-16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kgbench.cli import main
from kgbench.longmemeval import (
    DEFAULT_DATA,
    load_longmemeval,
    run_longmemeval,
    score_question,
    session_document,
)
from kgbench.retrieval import RetrievedSentence

TYPES = {"single-session-user", "multi-session", "temporal-reasoning", "knowledge-update"}


def test_fixture_loads_every_question_type_with_sessions_in_date_order() -> None:
    questions = load_longmemeval(DEFAULT_DATA)
    assert {q.type for q in questions} == TYPES
    for q in questions:
        assert [s.date for s in q.sessions] == sorted(s.date for s in q.sessions)
        assert set(q.evidence) <= {s.id for s in q.sessions}


def test_sessions_are_ingested_oldest_first_even_if_the_file_lists_them_otherwise(
    tmp_path: Path,
) -> None:
    raw = json.loads(DEFAULT_DATA.read_text())[:1]
    for key in ("haystack_session_ids", "haystack_dates", "haystack_sessions"):
        raw[0][key].reverse()
    path = tmp_path / "rev.json"
    path.write_text(json.dumps(raw))
    seen: list[list[str]] = []

    class Spy:
        def __init__(self, documents) -> None:
            seen.append([d.id for d in documents])

        def retrieve(self, question: str, k: int) -> list[RetrievedSentence]:
            return []

    run_longmemeval(load_longmemeval(path), systems=["spy"], factories={"spy": Spy})
    assert seen == [["q1_s1", "q1_s2", "q1_s3"]]


def test_session_document_has_a_date_line_and_turn_offsets_that_match_the_text() -> None:
    doc = session_document(load_longmemeval(DEFAULT_DATA)[0].sessions[0])
    assert doc.text.startswith("Session date: 2024/01/05")
    assert all(doc.text[s.start : s.end] == s.text for s in doc.sentences)


def test_score_question_session_recall_and_answer_hit_at_k() -> None:
    q = next(q for q in load_longmemeval(DEFAULT_DATA) if q.id == "q3")  # two evidence sessions
    ranking = [
        RetrievedSentence("q3_s1", "I practise the piano", ()),
        RetrievedSentence("q3_s2", "noise", ()),
        RetrievedSentence("q3_s3", "cello strings", ()),
    ]
    score = score_question(q, ranking, (1, 3))
    assert score.session_recall == {1: 0.5, 3: 1.0}
    assert score.answer_hit == {1: 0.0, 3: 0.0}  # "piano and cello" is in no sentence
    hit = score_question(q, [RetrievedSentence("q3_s1", "She plays piano and cello.", ())], (1,))
    assert hit.answer_hit == {1: 1.0}


def test_abstention_questions_are_counted_and_not_scored(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_DATA.read_text())[:2]
    raw[1]["question_id"] += "_abs"
    path = tmp_path / "abs.json"
    path.write_text(json.dumps(raw))
    result = run_longmemeval(load_longmemeval(path), systems=["bm25"])
    assert (result.questions, result.abstention) == (1, 1)


def test_bm25_and_graph_rag_find_every_evidence_session_by_rank_five() -> None:
    result = run_longmemeval(load_longmemeval(DEFAULT_DATA), ks=(5,))
    assert result.questions == 6
    for system in ("bm25", "graph-rag"):
        report = result.systems[system]
        assert set(report.by_type) == TYPES
        assert report.overall.session_recall[5] == 1.0


def test_cli_run_dataset_longmemeval_prints_per_type_rows_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "lme.json"
    assert main(["run", "--dataset", "longmemeval", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    for kind in TYPES:
        assert kind in text
    data = json.loads(out.read_text())
    assert data["questions"] == 6
    assert set(data["systems"]) == {"bm25", "graph-rag"}
    assert set(data["systems"]["bm25"]["by_type"]) == TYPES


def test_cli_longmemeval_subcommand_limit(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["longmemeval", "--limit", "2", "--systems", "bm25"]) == 0
    assert "2 questions scored" in capsys.readouterr().out
