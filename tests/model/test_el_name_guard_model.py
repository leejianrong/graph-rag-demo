"""GRA-157 with the real ``bge-small-en-v1.5`` embedder (marked ``model``, not in the fast gate).

The real model scores Acme and Northwind ~0.99 against each other from the shared sentence;
the name guard keeps them apart. Skips when the model is unavailable.
"""

from __future__ import annotations

import pytest

from graph_rag.fakes import InMemoryEntityStore
from graph_rag.stages.entity_linking import EntityLinkingStage
from tests.unit.test_el_name_guard import DOC_C, doc_c_inputs

pytestmark = pytest.mark.model


def test_two_orgs_in_one_sentence_stay_separate_with_real_embedder() -> None:
    pytest.importorskip("sentence_transformers")
    from graph_rag.adapters.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model="BAAI/bge-small-en-v1.5", dim=384)
    try:
        embedder.embed(["warmup"])
    except Exception as exc:  # noqa: BLE001 - offline model download
        pytest.skip(f"bge model unavailable: {exc}")
    mentions, sentences, clusters = doc_c_inputs()
    result = EntityLinkingStage(InMemoryEntityStore(), embedder).link(
        DOC_C, mentions, sentences, clusters
    )
    ids = {link.mention_text: link.canonical_id for link in result.links}
    assert ids["Acme Logistics"] != ids["Northwind Shipping"]


def test_two_logistics_companies_in_one_sentence_stay_separate_with_real_embedder() -> None:
    """GRA-172: the shared token "Logistics" is not enough when each name adds its own.

    Named in one sentence, the two score about 0.99 against each other with the real model, so
    only the name guard keeps them apart.
    """
    pytest.importorskip("sentence_transformers")
    from graph_rag.adapters.embedder import SentenceTransformerEmbedder
    from graph_rag.models import CorefCluster, Mention, Sentence

    embedder = SentenceTransformerEmbedder(model="BAAI/bge-small-en-v1.5", dim=384)
    try:
        embedder.embed(["warmup"])
    except Exception as exc:  # noqa: BLE001 - offline model download
        pytest.skip(f"bge model unavailable: {exc}")
    stage = EntityLinkingStage(InMemoryEntityStore(), embedder)

    def link(text: str, names: list[str]) -> dict[str, tuple[str, bool]]:
        mentions = [
            Mention(text=n, type="ORG", char_start=text.index(n), char_end=text.index(n) + len(n))
            for n in names
        ]
        sentences = [Sentence(text=text, char_start=0, char_end=len(text), index=0)]
        clusters = [CorefCluster(canonical=n, members=[n]) for n in names]
        result = stage.link(text, mentions, sentences, clusters)
        return {x.mention_text: (x.canonical_id, x.is_new) for x in result.links}

    both = link(
        "Acme Logistics acquired Northwind Logistics in 2021.",
        ["Acme Logistics", "Northwind Logistics"],
    )
    assert both["Acme Logistics"][0] != both["Northwind Logistics"][0]
    assert both["Northwind Logistics"][1]  # a new entity, not a merge
