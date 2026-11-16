"""The temporal bench: a changing world in dated documents, scored by question type (GRA-190)."""

from __future__ import annotations

import pytest
from kgbench import temporal
from kgbench.compare import CachingEmbedder
from kgbench.extraction import ingest
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import evaluate
from kggen.spec import load_preset

from graph_rag.models import QueryRequest
from graph_rag.wiring import build_embedder, build_offline_components


@pytest.fixture(scope="module")
def built():
    return temporal.build(load_preset("small"), 60)


def test_the_world_changes_and_every_temporal_type_is_asked(built) -> None:
    world, documents, questions = built
    assert {q.type for q in questions} == set(temporal.TYPES)
    assert all(d.date for d in documents)
    assert any(f.valid_to is not None for f in world.edges)


def test_ingest_gives_the_pipeline_the_document_date(built) -> None:
    """Edges carry the dates of the documents, so an as-of question can leave out later facts."""
    _, documents, questions = built
    components = build_offline_components(embedder=CachingEmbedder(build_embedder()))
    list(ingest(components, documents))
    question = questions[0].question
    now = components.retriever.retrieve(QueryRequest(question=question))
    before_it_all = components.retriever.retrieve(
        QueryRequest(question=question, as_of="1990-12-31")
    )
    assert now.subgraph.edges and not before_it_all.subgraph.edges


class _Recorder:
    def __init__(self, dated: bool) -> None:
        self.calls: list[object] = []
        if dated:
            self.supports_as_of = True

    def retrieve(self, question: str, k: int, as_of: int | None = None):
        self.calls.append(as_of)
        return []


def test_only_a_system_that_can_use_a_date_is_given_one(built) -> None:
    _, _, questions = built
    asked = [q for q in questions if q.as_of is not None]
    assert asked
    dated = _Recorder(True)
    evaluate(dated, questions, (5,))
    assert sorted(c for c in dated.calls if c is not None) == sorted(q.as_of for q in asked)
    plain = _Recorder(False)
    evaluate(plain, questions, (5,))  # would raise on the extra argument if it were passed
    assert len(plain.calls) == len(questions)


def test_graph_rag_retriever_turns_a_year_into_the_end_of_it(built) -> None:
    _, documents, _ = built
    components = build_offline_components(embedder=CachingEmbedder(build_embedder()))
    seen: list[str | None] = []
    inner = components.retriever
    original = inner.retrieve
    inner.retrieve = lambda request: (seen.append(request.as_of), original(request))[1]  # type: ignore[method-assign]
    retriever = GraphRagRetriever(components, documents[:20], mode="graph")
    retriever.retrieve("anything", 5, as_of=2011)
    retriever.retrieve("anything", 5)
    assert seen == ["2011-12-31", None]


def test_a_run_scores_every_system_by_type_and_the_table_reads(built) -> None:
    result = temporal.run(
        load_preset("small"), n_questions=60, embedder=CachingEmbedder(build_embedder())
    )
    assert list(result.solved) == list(temporal.SYSTEMS)
    for rows in result.solved.values():
        assert set(rows) == set(temporal.TYPES)
        assert all(0 <= hit <= n for hit, n in rows.values())
    table = temporal.format_table(result)
    assert "| as_of |" in table and "| all |" in table
    gated = temporal.metrics(result)
    assert gated and all(m.startswith("temporal/graph-rag") for m in gated)
    assert all(0 <= v <= 1 for v in gated.values())


def test_a_stale_sentence_ranked_first_is_the_wrong_version() -> None:
    from kgbench.retrieval import RetrievedSentence
    from kggen.questions import Question

    q = Question("Q0", "?", 1, "E0", "E1", "x", ("x",), ("F2",))
    versions = {"F2": {"F1"}, "F1": {"F2"}}
    stale = RetrievedSentence("D0", "old", ("F1",))
    right = RetrievedSentence("D1", "new", ("F2",))
    other = RetrievedSentence("D2", "unrelated", ("F9",))
    assert temporal.first_is_right(q, [other, right, stale], versions)
    assert not temporal.first_is_right(q, [stale, right], versions)  # gold is there, but second
    assert not temporal.first_is_right(q, [other], versions)  # nothing about the fact at all
