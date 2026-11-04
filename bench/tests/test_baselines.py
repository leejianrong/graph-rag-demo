"""BM25 baseline: ranked sentences with fact ids, deterministic, no network."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from kgbench.baselines import BM25Retriever, DenseRetriever, sentences_of, tokenize
from kgbench.retrieval import RetrievedSentence, evaluate
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import Document, Sentence, render_documents
from kggen.spec import load_preset
from kggen.world import generate

from graph_rag.wiring import build_embedder

FIXTURE = Path(__file__).resolve().parents[2] / "datagen" / "fixtures" / "small"


@pytest.fixture(scope="module")
def small():
    world = generate(load_preset("small"))
    documents = render_documents(world)
    return world, documents, generate_questions(world, QuestionSpec(n_questions=50))


def doc(doc_id: str, *sentences: tuple[str, str]) -> Document:
    """A hand-built document: each sentence is (text, fact id)."""
    built, offset = [], 0
    for text, fact in sentences:
        built.append(Sentence(text, offset, offset + len(text), (fact,), ()))
        offset += len(text) + 1
    return Document(doc_id, doc_id, " ".join(s.text for s in built), tuple(built), False)


def test_the_corpus_is_the_committed_small_fixture(small) -> None:
    committed = json.loads((FIXTURE / "documents.json").read_text())
    assert [(d["id"], d["text"]) for d in committed] == [(d.id, d.text) for d in small[1]]


def test_returns_ranked_sentences_that_carry_fact_ids(small) -> None:
    _, documents, questions = small
    retriever = BM25Retriever(documents)
    ranking = retriever.retrieve(questions[0].question, 5)
    assert 0 < len(ranking) <= 5
    assert all(isinstance(s, RetrievedSentence) and s.fact_ids and s.doc for s in ranking)


def test_one_hop_questions_are_found_and_deeper_ones_are_not(small) -> None:
    _, documents, questions = small
    report = evaluate(BM25Retriever(documents), questions, ks=(5,))
    assert report.by_hops[1].path_at[5] == 1.0  # the question names its entity: easy for keywords
    assert report.by_hops[4].path_at[5] == 0.0  # the whole chain is not in any one sentence's words
    assert report.by_hops[4].recall_at[5] < report.by_hops[1].recall_at[5]


def test_ranking_is_deterministic(small) -> None:
    _, documents, questions = small
    first, second = BM25Retriever(documents), BM25Retriever(documents)
    for q in questions[:10]:
        assert first.retrieve(q.question, 10) == second.retrieve(q.question, 10)


def test_it_never_touches_the_network(small, monkeypatch: pytest.MonkeyPatch) -> None:
    _, documents, questions = small

    def refuse(*args: object, **kwargs: object):
        raise AssertionError("BM25 must not open a socket")

    retriever = BM25Retriever(documents)
    monkeypatch.setattr(socket, "socket", refuse)
    assert retriever.retrieve(questions[0].question, 3)


def test_a_rare_term_outranks_a_common_one_even_when_the_common_one_comes_first() -> None:
    docs = [
        doc("A", ("Acme sells widgets.", "F1"), ("Beta sells widgets.", "F2")),
        doc("B", ("Gamma sells widgets.", "F3"), ("Delta sells widgets.", "F4")),
        doc("C", ("Zork builds cranes.", "F5")),
    ]
    # "widgets" is in four sentences, "zork" in one: IDF must put F5 above the earlier F1
    ranked = BM25Retriever(docs).retrieve("zork widgets", 5)
    assert ranked[0].fact_ids == ("F5",)


def test_equal_scores_break_ties_by_corpus_position() -> None:
    docs = [
        doc("A", ("Acme owns widgets.", "F1"), ("Zork owns widgets.", "F2")),
        doc("B", ("Beta owns widgets.", "F3"), ("Zork builds cranes.", "F4")),
    ]
    retriever = BM25Retriever(docs)
    assert [s.fact_ids[0] for s in retriever.retrieve("widgets", 3)] == ["F1", "F2", "F3"]
    # F2 and F4 both contain Zork once at the same length, so they tie and keep document order
    assert [s.fact_ids[0] for s in retriever.retrieve("zork", 2)] == ["F2", "F4"]


def test_k_is_respected_and_no_overlap_returns_nothing() -> None:
    retriever = BM25Retriever(
        [doc("A", ("Acme owns widgets.", "F1"), ("Beta owns gadgets.", "F2"))]
    )
    assert len(retriever.retrieve("owns", 1)) == 1
    assert retriever.retrieve("completely unrelated words", 5) == []
    assert BM25Retriever([]).retrieve("anything", 5) == []


def test_sentences_keep_document_order_and_facts() -> None:
    docs = [doc("A", ("One.", "F1"), ("Two.", "F2")), doc("B", ("Three.", "F3"))]
    assert [(s.doc, s.fact_ids) for s in sentences_of(docs)] == [
        ("A", ("F1",)),
        ("A", ("F2",)),
        ("B", ("F3",)),
    ]
    assert tokenize("Dune Partners, Inc.") == ["dune", "partners", "inc"]


# --- dense baseline --------------------------------------------------------------------------


class OneHot:
    """An exact embedder for hand-built cases: one dimension per word of a fixed vocabulary."""

    def __init__(self, vocabulary: list[str], scale: float = 1.0) -> None:
        self.vocabulary, self.scale = vocabulary, scale
        self.calls: list[int] = []  # how many texts each embed() call received

    @property
    def dim(self) -> int:
        return len(self.vocabulary)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [
            [self.scale * float(w in tokenize(text)) for w in self.vocabulary] for text in texts
        ]


def test_dense_ranks_by_cosine_and_ties_break_by_position() -> None:
    docs = [
        doc("A", ("zork cranes", "F1"), ("zork", "F2")),
        doc("B", ("cranes", "F3"), ("zork cranes", "F4")),
    ]
    dense = DenseRetriever(docs, OneHot(["zork", "cranes"]))
    # the query "zork" is closest to the pure-zork sentence, then the two equal mixed ones in order
    assert [s.fact_ids[0] for s in dense.retrieve("zork", 4)] == ["F2", "F1", "F4", "F3"]


class WordCounts:
    """Unnormalised vectors whose length grows with repetition: "zork zork zork" is 3x "zork"."""

    dim = 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(tokenize(t).count("zork"))] for t in texts]


def test_dense_scores_cosine_so_a_longer_vector_does_not_win() -> None:
    docs = [doc("A", ("zork", "F1"), ("zork zork zork", "F2"))]
    # both point the same way as the query, so cosine ties and corpus order decides: F1 first.
    # A raw dot product would prefer F2, whose vector is three times longer.
    assert [s.fact_ids[0] for s in DenseRetriever(docs, WordCounts()).retrieve("zork", 2)] == [
        "F1",
        "F2",
    ]


def test_dense_embeds_the_corpus_once_in_batches_and_each_question_once() -> None:
    docs = [doc("A", *[(f"word{i}", f"F{i}") for i in range(10)])]
    embedder = OneHot(["word1"])
    dense = DenseRetriever(docs, embedder, batch_size=4)
    assert embedder.calls == [4, 4, 2]
    dense.retrieve("word1", 3)
    dense.retrieve("word2", 3)
    assert embedder.calls[3:] == [1, 1]  # one text per question, the corpus is not re-embedded


def test_dense_with_the_project_fake_embedder_on_the_small_fixture(small) -> None:
    _, documents, questions = small
    dense = DenseRetriever(documents, build_embedder())
    ranking = dense.retrieve(questions[0].question, 5)
    assert len(ranking) == 5 and all(s.fact_ids for s in ranking)
    assert ranking == DenseRetriever(documents, build_embedder()).retrieve(questions[0].question, 5)
    report = evaluate(dense, questions, ks=(5,))
    assert report.by_hops[1].recall_at[5] > 0.5  # the entity's own name is shared with its sentence
    assert report.by_hops[4].path_at[5] == 0.0


def test_dense_handles_an_empty_corpus_and_a_large_k() -> None:
    assert DenseRetriever([], build_embedder()).retrieve("anything", 5) == []
    docs = [doc("A", ("one", "F1"), ("two", "F2"))]
    assert len(DenseRetriever(docs, OneHot(["one"])).retrieve("one", 50)) == 2


def test_build_embedder_gives_the_fake_by_default_and_the_real_one_without_loading_it() -> None:
    from graph_rag.fakes import FakeEmbedder

    assert isinstance(build_embedder(), FakeEmbedder)
    real = build_embedder(real=True)  # constructing it needs no torch and no model download
    assert type(real).__name__ == "SentenceTransformerEmbedder" and real.dim == 384
