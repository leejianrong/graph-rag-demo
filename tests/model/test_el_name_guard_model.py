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
