"""Alias, look-alike and distractor statistics (GRA-125)."""

from __future__ import annotations

from kgbench import difficulty
from kggen.render import Document, Mention, Sentence

WORLD = {
    "entities": [
        {"id": "A", "name": "Aa Works", "type": "ORG", "aliases": ["Aa"]},
        {"id": "B", "name": "Bb Group", "type": "ORG", "aliases": []},
        {"id": "C", "name": "Bb Groups", "type": "ORG", "aliases": ["Aa"]},
    ],
    "edges": [{"id": "F1", "subject": "A", "object": "B"}],
}


def doc(i: str, text: str, facts: tuple[str, ...] = (), distractor: bool = False, mention=None):
    mentions = (mention,) if mention else ()
    sentence = Sentence(text, 0, len(text), facts, mentions)
    return Document(i, "T", text, (sentence,), distractor)


def test_world_stats_count_aliases_shared_names_and_near_duplicates():
    stats = difficulty.world_stats(WORLD)
    assert stats["aliased"] == 2 / 3  # A and C
    assert stats["shared_name"] == 2 / 3  # both carry the alias "Aa"
    assert stats["near_duplicate_pairs"] == 1  # "bb group" and "bb groups"


def test_text_stats_count_alias_mentions_and_distractors():
    mention = Mention(start=0, end=2, entity_id="A", surface="Aa")
    docs = [doc("D1", "Aa owns it.", ("F1",), mention=mention), doc("D2", "x", distractor=True)]
    stats = difficulty.text_stats(docs, WORLD)
    assert stats["alias_mentions"] == 1.0 and stats["distractor_documents"] == 0.5


def test_documents_without_the_facts_that_name_a_path_entity_are_counted():
    docs = [
        doc("D1", "Aa Works owns Bb Group.", ("F1",)),
        doc("D2", "Bb Group is mentioned elsewhere.", distractor=True),
        doc("D3", "Nothing relevant."),
    ]
    q = {"hops": 1, "supporting_facts": ["F1"]}
    row = difficulty.question_stats([q], WORLD, docs)[1]
    assert (row["confusing"], row["distractors"], row["any_distractor"]) == (1, 1, 1)
