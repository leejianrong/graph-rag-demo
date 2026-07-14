"""Unit tests for ``InMemoryDocumentStore.search_sentences`` (V6, fast, offline).

Exercises the passage/sentence-anchored kNN seam on the fake (ADR-0010, B5): the
same external behaviour the real ``EsDocumentStore`` is proved against by its
contract test. Deterministic, ``$0``, no Docker.
"""

from __future__ import annotations

from graph_rag.fakes import InMemoryDocumentStore
from graph_rag.models import DocumentRecord, Sentence


def _record(
    document_id: str,
    sentences: list[Sentence],
    vectors: list[list[float]] | None,
) -> DocumentRecord:
    """Build a document record with aligned sentences + sentence vectors."""
    return DocumentRecord(
        document_id=document_id,
        bucket="documents",
        object_key=f"{document_id}.md",
        text="".join(s.text for s in sentences),
        sentences=sentences,
        sentence_vectors=vectors,
    )


def _sentence(index: int, text: str, start: int) -> Sentence:
    """Build a sentence with char offsets into a notional raw text."""
    return Sentence(text=text, char_start=start, char_end=start + len(text), index=index)


def test_search_returns_nearest_with_offsets() -> None:
    """The nearest sentence comes back first with its correct offsets + score."""
    store = InMemoryDocumentStore()
    store.upsert(
        _record(
            "d1",
            sentences=[
                _sentence(0, "cats and dogs", 0),
                _sentence(1, "quantum physics", 13),
            ],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
    )

    results = store.search_sentences(vector=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    top = results[0]
    assert top.document_id == "d1"
    assert top.sentence_index == 0
    assert top.text == "cats and dogs"
    assert top.char_start == 0
    assert top.char_end == 13
    assert abs(top.score - 1.0) < 1e-9
    assert results[1].sentence_index == 1


def test_search_top_k_respected() -> None:
    """``top_k`` truncates the result to the k best matches."""
    store = InMemoryDocumentStore()
    store.upsert(
        _record(
            "d1",
            sentences=[_sentence(i, f"s{i}", i * 3) for i in range(4)],
            vectors=[[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
        )
    )
    results = store.search_sentences(vector=[1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert [r.sentence_index for r in results] == [0, 1]


def test_search_deterministic_tie_break() -> None:
    """Equal scores order by document_id then sentence_index (deterministic)."""
    store = InMemoryDocumentStore()
    same = [1.0, 0.0]
    store.upsert(_record("d2", [_sentence(0, "x", 0)], [same]))
    store.upsert(_record("d1", [_sentence(3, "y", 0), _sentence(1, "z", 1)], [same, same]))

    results = store.search_sentences(vector=[1.0, 0.0], top_k=5)
    assert [(r.document_id, r.sentence_index) for r in results] == [
        ("d1", 1),
        ("d1", 3),
        ("d2", 0),
    ]


def test_search_empty_when_no_sentence_vectors() -> None:
    """A record with no sentence vectors contributes nothing."""
    store = InMemoryDocumentStore()
    store.upsert(_record("d1", [_sentence(0, "hello", 0)], vectors=None))
    store.upsert(_record("d2", [_sentence(0, "world", 0)], vectors=[[1.0, 0.0]]))

    results = store.search_sentences(vector=[1.0, 0.0], top_k=5)
    assert [r.document_id for r in results] == ["d2"]


def test_search_empty_store() -> None:
    """An empty store returns no sentences."""
    assert InMemoryDocumentStore().search_sentences(vector=[1.0], top_k=3) == []
