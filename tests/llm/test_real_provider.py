"""Opt-in real-provider LLM test (marked ``llm``; TESTING §5 V3, SLICES V3).

Excluded from the fast pre-push gate and from CI's required jobs — it makes a real
provider call (which costs money and needs a key). It proves the *integration*:
:class:`~graph_rag.adapters.llm_client.LiteLLMClient` against a real provider
returns a schema-valid :class:`~graph_rag.models.ClusterMap` for a tiny coref
prompt.

Skips cleanly (never fails) when no API key is present, so ``uv run pytest -m
llm`` is safe to run anywhere — including a CI job without secrets. Run it
deliberately with a key set:

    OPENAI_API_KEY=... uv run pytest -m llm -q
"""

from __future__ import annotations

import pytest

from graph_rag.config import Settings
from graph_rag.models import ClusterMap
from graph_rag.stages.coref import LLMCorefStage, build_coref_prompt

pytestmark = pytest.mark.llm


def _settings_or_skip() -> Settings:
    """Return Settings, or skip the module cleanly if no API key is configured."""
    settings = Settings()
    if not settings.openai_api_key:
        pytest.skip("no LLM API key in env (set OPENAI_API_KEY) — opt-in real-provider test")
    return settings


def test_real_provider_returns_schema_valid_clusters() -> None:
    """A real provider call yields a schema-valid, non-destructive cluster map."""
    from graph_rag.adapters.llm_client import LiteLLMClient

    settings = _settings_or_skip()
    client = LiteLLMClient.from_settings(settings)

    text = "Ada Lovelace lived in London. She loved math."
    prompt = build_coref_prompt(text, [])
    result = client.structured(prompt, ClusterMap)

    # Schema-valid by construction (structured() validated it); assert it is a
    # ClusterMap and non-destructive — canonicals are surface strings.
    assert isinstance(result, ClusterMap)
    for cluster in result.clusters:
        assert isinstance(cluster.canonical, str)
        assert isinstance(cluster.members, list)


def test_real_provider_via_stage() -> None:
    """The LLMCorefStage end-to-end against a real provider returns clusters."""
    settings = _settings_or_skip()
    stage = LLMCorefStage.from_settings(settings)

    text = "Marie Curie won a Nobel Prize. She won a second one later."
    clusters = stage.resolve(text, [])

    assert isinstance(clusters, list)


def test_real_provider_synthesizes_prose() -> None:
    """A real-provider V7 synthesis call returns usable prose from a small prompt.

    Proves the integration: :class:`~graph_rag.query.synthesis.AnswerSynthesizer`
    over a real :class:`~graph_rag.adapters.llm_client.LiteLLMClient` (pinned to
    ``settings.synthesis_model``) turns a tiny assembled evidence prompt into
    grounded prose. Skips cleanly without an API key.
    """
    from graph_rag.models import (
        CanonicalEntity,
        EdgeProvenance,
        QueryResponse,
        RankedNode,
        Subgraph,
        SupportingSentence,
        Triple,
    )
    from graph_rag.query.synthesis import AnswerSynthesizer

    settings = _settings_or_skip()
    synthesizer = AnswerSynthesizer.from_settings(settings)

    node_a = CanonicalEntity(canonical_id="e-ada", name="Ada Lovelace", type="PERSON")
    node_b = CanonicalEntity(canonical_id="e-engine", name="Analytical Engine", type="PRODUCT")
    response = QueryResponse(
        answer="Ada Lovelace",
        answer_entity=RankedNode(
            canonical_id="e-ada", name="Ada Lovelace", type="PERSON", score=1.0
        ),
        subgraph=Subgraph(
            nodes=[node_a, node_b],
            edges=[
                Triple(
                    subject_id="e-ada",
                    predicate="WORKED_ON",
                    object_id="e-engine",
                    provenance=EdgeProvenance(
                        source_doc_id="doc1",
                        sentence_index=0,
                        source_sentence="Ada Lovelace worked on the Analytical Engine.",
                    ),
                )
            ],
        ),
        supporting_sentences=[
            SupportingSentence(
                document_id="doc1",
                text="Ada Lovelace worked on the Analytical Engine.",
                char_start=0,
                char_end=45,
                sentence_index=0,
                score=0.99,
            )
        ],
    )

    prose = synthesizer.synthesize(question="What did Ada Lovelace work on?", response=response)

    assert isinstance(prose, str)
    assert prose.strip()
