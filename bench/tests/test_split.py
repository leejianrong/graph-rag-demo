"""Scores split by shared names (GRA-130)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from kgbench import split
from kgbench.extraction import Extraction, Gold, PredictedMention
from kgbench.retrieval import RetrievedSentence


def entity(i: str, name: str, *aliases: str) -> NS:
    return NS(id=i, name=name, aliases=aliases)


def test_shared_ids_counts_a_name_or_alias_held_by_another_entity() -> None:
    world = NS(
        entities=[
            entity("E1", "Mira Holt"),
            entity("E2", "Acme", "Mira Holt"),  # alias equals E1's name
            entity("E3", "ZED", "zed"),  # one entity carrying a label twice is not shared
            entity("E4", "Orin"),
        ]
    )
    assert split.shared_ids(world) == {"E1", "E2"}


class Table:
    """Returns the sentence list it was given for every question."""

    def __init__(self, facts: list[tuple[str, ...]]) -> None:
        self.facts = facts

    def retrieve(self, question: str, k: int):
        return [RetrievedSentence("D", "s", f) for f in self.facts][:k]


def q(i: str, hops: int, facts: list[str], shared: bool) -> dict:
    return {
        "id": i,
        "hops": hops,
        "question": i,
        "supporting_facts": facts,
        "shared_name": shared,
    }


def test_retrieval_counts_path_and_recall_per_hops_and_flag() -> None:
    questions = [
        q("a", 2, ["F1", "F2"], False),
        q("b", 2, ["F1", "F3"], True),
        q("c", 2, ["F1", "F2"], True),
    ]
    result = split.retrieval(questions, {"T": Table([("F1",), ("F2",)])}, k=2)["T"]
    assert result[(2, False)] == {"n": 1, "path": 1, "recall": 1}
    assert result[(2, True)] == {"n": 2, "path": 1, "recall": 1.5}


def test_extraction_splits_mentions_and_triples_by_shared_entities() -> None:
    gold = Gold(
        mentions={("D", 0, 1): "E1", ("D", 2, 3): "E2", ("D", 4, 5): "E3"},
        entities=frozenset({"E1", "E2", "E3"}),
        triples=frozenset({("E1", "OWNS", "E2"), ("E2", "OWNS", "E3")}),
    )
    found = Extraction(
        mentions=(
            PredictedMention("D", 0, 1, "c1"),  # E1: shared, found
            PredictedMention("D", 4, 5, "c3"),  # E3: not shared, found; E2 is missed
        ),
        triples=(("c1", "OWNS", "c2"),),  # c2 has no aligned mention, so E1 OWNS E2 is not found
    )
    out = split.extraction(gold, found, shared=frozenset({"E1"}))
    assert out["mentions"] == {False: (1, 2), True: (1, 1)}
    assert out["triples"] == {False: (0, 1), True: (0, 1)}


def test_a_triple_is_found_when_its_entities_align() -> None:
    gold = Gold(
        mentions={("D", 0, 1): "E1", ("D", 2, 3): "E2"},
        entities=frozenset({"E1", "E2"}),
        triples=frozenset({("E1", "OWNS", "E2")}),
    )
    found = Extraction(
        mentions=(PredictedMention("D", 0, 1, "c1"), PredictedMention("D", 2, 3, "c2")),
        triples=(("c1", "OWNS", "c2"),),
    )
    out = split.extraction(gold, found, shared=frozenset({"E2"}))
    assert out["triples"] == {True: (1, 1)}


def test_tables_hold_the_gap_and_handle_an_empty_group() -> None:
    result = {
        "T": {
            (2, False): {"n": 4, "path": 3, "recall": 3},
            (2, True): {"n": 4, "path": 1, "recall": 1},
        }
    }
    text = split.format_retrieval(result, 5)
    assert "| T | 2 | 75% (3/4," in text and "-50" in text
    only = split.format_retrieval({"T": {(1, False): {"n": 1, "path": 1, "recall": 1}}}, 5)
    assert "n/a" in only
