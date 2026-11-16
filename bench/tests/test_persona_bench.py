"""The persona bench: newcomer, learner and expert questions, scored by who asks (GRA-187)."""

from __future__ import annotations

import pytest
from kgbench import personas
from kgbench.compare import CachingEmbedder
from kggen.spec import load_preset
from kggen.world import generate

from graph_rag.wiring import build_embedder


@pytest.fixture(scope="module")
def world():
    return generate(load_preset("small", "hard").with_overrides(name_style="bank"))


def test_the_rows_isolate_what_resolving_a_reference_is_worth(world) -> None:
    rows = personas.groups(world, 40)
    alone = rows["learner: last turn alone, its own fact"]
    with_history = rows["learner: last turn with the earlier turns, its own fact"]
    chain = rows["learner: whole chain, asked with the earlier turns"]
    assert alone and len(alone) == len(with_history) == len(chain)
    for a, h, c in zip(alone, with_history, chain, strict=True):
        assert a.supporting_facts == h.supporting_facts == (a.turns[-1]["supporting_facts"][0],)
        assert c.supporting_facts == tuple(t["supporting_facts"][0] for t in c.turns)
        assert h.question.startswith("Q:") and h.question.endswith(a.question)
        assert a.question == a.turns[-1]["question"]
    assert rows["newcomer"] and rows["expert: what two entities share"]
    assert "learner" not in rows


def test_a_run_scores_every_system_on_every_row(world) -> None:
    result = personas.run(
        load_preset("small", "hard").with_overrides(name_style="bank"),
        n_questions=30,
        embedder=CachingEmbedder(build_embedder()),
    )
    assert list(result.solved) == list(personas.SYSTEMS)
    rows = [r for r, qs in result.groups.items() if qs]
    for scores in result.solved.values():
        assert set(scores) == set(rows)
        assert all(0 <= hit <= n for hit, n in scores.values())
    table = personas.format_table(result)
    assert "| newcomer |" in table and "last turn alone" in table
    # The earlier turns carry the entity the pronoun stands for, so they help a word search.
    bm25 = result.solved["BM25"]
    assert (
        bm25["learner: last turn with the earlier turns, its own fact"][0]
        > bm25["learner: last turn alone, its own fact"][0]
    )
