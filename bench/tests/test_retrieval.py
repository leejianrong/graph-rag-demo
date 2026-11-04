"""Retrieval scoring on hand-built cases: recall@k, whole-path@k, per hop count."""

from __future__ import annotations

import json

import pytest
from kgbench.retrieval import RetrievedSentence, evaluate, facts_in_top, score_retrieval
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import render_documents
from kggen.spec import WorldSpec
from kggen.world import generate


def question(qid: str, facts: tuple[str, ...]):
    from kggen.questions import Question

    return Question(qid, f"q {qid}?", len(facts), "E0", "E1", "answer", ("answer",), facts)


def sentence(*facts: str, doc: str = "D0") -> RetrievedSentence:
    return RetrievedSentence(doc, "text " + ",".join(facts), tuple(facts))


ONE, THREE = question("Q1", ("F1",)), question("Q3", ("F1", "F2", "F3"))


def test_a_retriever_that_returns_nothing_scores_zero_everywhere() -> None:
    for rankings in ({}, {"Q1": [], "Q3": []}):
        report = score_retrieval([ONE, THREE], rankings)
        for score in (report.overall, *report.by_hops.values()):
            assert set(score.recall_at.values()) == {0.0}
            assert set(score.path_at.values()) == {0.0}


def test_a_hit_at_rank_one_scores_one_for_a_single_hop_question() -> None:
    report = score_retrieval([ONE], {"Q1": [sentence("F1"), sentence("X")]}, ks=(1, 3))
    assert report.overall.recall_at == {1: 1.0, 3: 1.0}
    assert report.overall.path_at == {1: 1.0, 3: 1.0}


def test_recall_grows_with_k_and_the_path_needs_every_fact() -> None:
    ranking = [sentence("F1"), sentence("noise"), sentence("F2"), sentence("F3")]
    report = score_retrieval([THREE], {"Q3": ranking}, ks=(1, 3, 5))
    assert report.overall.recall_at[1] == pytest.approx(1 / 3)
    assert report.overall.recall_at[3] == pytest.approx(2 / 3)
    assert report.overall.recall_at[5] == 1.0
    assert report.overall.path_at == {1: 0.0, 3: 0.0, 5: 1.0}


def test_one_sentence_can_cover_several_facts() -> None:
    assert facts_in_top([sentence("F1", "F2"), sentence("F3")], 1) == {"F1", "F2"}
    report = score_retrieval([THREE], {"Q3": [sentence("F1", "F2", "F3")]}, ks=(1,))
    assert report.overall.path_at[1] == 1.0


def test_repeated_sentences_do_not_inflate_the_score() -> None:
    report = score_retrieval([THREE], {"Q3": [sentence("F1")] * 5}, ks=(5,))
    assert report.overall.recall_at[5] == pytest.approx(1 / 3)


def test_sentences_without_fact_ids_count_for_nothing() -> None:
    report = score_retrieval([ONE], {"Q1": [RetrievedSentence("D9", "text", ())]}, ks=(1,))
    assert report.overall.recall_at[1] == 0.0


def test_scores_are_reported_per_hop_count_and_overall_is_per_question() -> None:
    rankings = {"Q1": [sentence("F1")], "Q3": [sentence("noise")]}
    report = score_retrieval([ONE, THREE], rankings, ks=(1,))
    assert sorted(report.by_hops) == [1, 3]
    assert report.by_hops[1].questions == 1 and report.by_hops[1].path_at[1] == 1.0
    assert report.by_hops[3].path_at[1] == 0.0
    assert report.overall.questions == 2
    assert report.overall.path_at[1] == pytest.approx(0.5)


def test_ks_are_sorted_and_deduplicated_and_deeper_never_scores_lower() -> None:
    ranking = [sentence("F1"), sentence("F2"), sentence("F3")]
    report = score_retrieval([THREE], {"Q3": ranking}, ks=(5, 1, 3, 3))
    assert report.ks == (1, 3, 5)
    values = [report.overall.recall_at[k] for k in report.ks]
    assert values == sorted(values)


def test_no_questions_is_an_empty_report_not_an_error() -> None:
    report = score_retrieval([], {}, ks=(1, 3))
    assert report.overall.questions == 0 and report.by_hops == {}


def test_evaluate_asks_the_retriever_for_the_deepest_k() -> None:
    asked: list[tuple[str, int]] = []

    class Stub:
        def retrieve(self, question: str, k: int):
            asked.append((question, k))
            return [sentence("F1")]

    report = evaluate(Stub(), [ONE], ks=(1, 5, 10))
    assert asked == [("q Q1?", 10)]
    assert report.overall.path_at[10] == 1.0


def test_report_is_serialisable_and_the_table_has_a_row_per_hop_count() -> None:
    report = score_retrieval([ONE, THREE], {"Q1": [sentence("F1")]}, ks=(1, 3))
    json.dumps(report.to_dict())
    table = report.format_table()
    assert "recall@k" in table and "all" in table
    assert any(line.split()[:1] == ["3"] for line in table.splitlines())


def test_an_oracle_retriever_on_generated_questions_scores_one() -> None:
    world = generate(WorldSpec(n_entities=150, avg_degree=3, seed=3, churn_rate=0.1))
    documents = render_documents(world)
    by_fact: dict[str, RetrievedSentence] = {}
    for d in documents:
        for s in d.sentences:
            for f in s.fact_ids:
                by_fact[f] = RetrievedSentence(d.id, s.text, s.fact_ids)
    questions = generate_questions(world, QuestionSpec(n_questions=30))

    class Oracle:
        def __init__(self) -> None:
            self.answers = {q.question: q for q in questions}

        def retrieve(self, question: str, k: int):
            return [by_fact[f] for f in self.answers[question].supporting_facts][:k]

    report = evaluate(Oracle(), questions, ks=(5,))
    assert report.overall.recall_at[5] == 1.0 and report.overall.path_at[5] == 1.0
    assert len(report.by_hops) > 1  # the questions span several hop counts

    class Nothing:
        def retrieve(self, question: str, k: int):
            return []

    assert evaluate(Nothing(), questions, ks=(5,)).overall.recall_at[5] == 0.0
