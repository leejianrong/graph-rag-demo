"""BM25, dense and graph-rag on a slice of a real dataset (GRA-127)."""

from __future__ import annotations

from kgbench import real
from kgbench.cli import _DEFAULT_DATASET
from kgbench.dataset import load_examples
from kgbench.retrieval import facts_in_top


def test_the_corpus_holds_one_document_per_paragraph_and_each_sentence_is_a_fact():
    examples = load_examples(_DEFAULT_DATASET)
    docs = real.corpus(examples)
    assert len({d.title for d in docs}) == len(docs)
    gold = {f for q in real.questions(examples) for f in q.supporting_facts}
    held = {f for d in docs for s in d.sentences for f in s.fact_ids}
    assert gold <= held  # every gold sentence exists in the corpus


def test_questions_count_hops_as_distinct_gold_paragraphs():
    examples = load_examples(_DEFAULT_DATASET)
    for example, question in zip(examples, real.questions(examples), strict=True):
        assert question.hops == len({t for t, _ in example.supporting_facts})
        assert question.supporting_facts == tuple(f"{t}#{i}" for t, i in example.supporting_facts)


def test_every_system_scores_the_mini_fixture():
    from graph_rag.wiring import build_embedder

    reports = real.measure(load_examples(_DEFAULT_DATASET), build_embedder(real=False), (1, 5))
    assert list(reports) == list(real.SYSTEMS)
    assert all(r.overall.path_at[5] >= r.overall.path_at[1] for r in reports.values())
    assert facts_in_top([], 5) == set()
    assert "Ranking by whole-path@5" in real.format_comparison(reports, 5)
