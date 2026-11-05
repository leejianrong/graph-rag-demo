"""The single-hop shortcut measures (GRA-123)."""

from __future__ import annotations

import pytest
from kgbench import shortcut
from kgbench.retrieval import RetrievedSentence
from kggen.render import Document, Sentence

WORLD = {
    "edges": [
        {"id": "F1", "subject": "E1", "predicate": "OWNS", "object": "E2"},
        {"id": "F2", "subject": "E3", "predicate": "OWNS", "object": "E4"},
        {"id": "F3", "subject": "E2", "predicate": "LOCATED_IN", "object": "E5"},
    ]
}
QUESTION = {
    "id": "Q1",
    "hops": 2,
    "question": "Where is the thing that E1 owns?",
    "answer": "E5",
    "accepted_answers": ["Cc City"],
    "gold_paths": [["F1", "F3"]],
    "supporting_facts": ["F1", "F3"],
}


def doc(*sentences: tuple[str, tuple[str, ...]]) -> Document:
    items = tuple(Sentence(t, 0, len(t), f, ()) for t, f in sentences)
    return Document("D", "T", " ".join(t for t, _ in sentences), items, False)


class Fixed:
    def __init__(self, texts):
        self.texts = texts

    def retrieve(self, question, k):
        return [RetrievedSentence("D", t, f) for t, f in self.texts[:k]]


def test_prior_is_one_over_the_entities_that_fill_the_answer_slot():
    q = {**QUESTION, "answer": "E2", "gold_paths": [["F1"]]}  # E2 is the object of an OWNS fact
    assert shortcut.prior(q, WORLD) == pytest.approx(1 / 2)  # E2 or E4
    assert shortcut.prior(QUESTION, WORLD) == 1.0  # only E5 is ever LOCATED_IN's object


def test_chance_is_exact():
    assert shortcut._chance(1, 4, 2) == pytest.approx(0.5)
    assert shortcut._chance(0, 4, 2) == 0.0


def test_the_last_fact_alone_is_a_shortcut_and_the_full_path_is_not():
    documents = [doc(("Aa owns Bb.", ("F1",)), ("Bb is in Cc City.", ("F3",)), ("Noise.", ()))]
    last_only = Fixed([("Bb is in Cc City.", ("F3",)), ("Noise.", ())])
    both = Fixed([("Aa owns Bb.", ("F1",)), ("Bb is in Cc City.", ("F3",))])
    out = shortcut.measure([QUESTION], WORLD, documents, {"a": last_only, "b": both}, k=2)
    a, b = out["a"][2], out["b"][2]
    assert (a["path"], a["last"], a["last_only"], a["answer_only"]) == (0, 1, 1, 1)
    assert (b["path"], b["last_only"], b["answer_only"]) == (1, 0, 0)


def test_table_and_chart_render():
    documents = [doc(("Bb is in Cc City.", ("F3",)))]
    out = shortcut.measure([QUESTION], WORLD, documents, {"BM25": Fixed([])}, k=1)
    assert "BM25" in shortcut.format_table(out, 1)
    assert shortcut.chart_svg(out, "BM25", "t").startswith("<svg")
