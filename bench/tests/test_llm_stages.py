"""The offline pipeline can swap in the real LLM stages, and says what they cost."""

from __future__ import annotations

from graph_rag.adapters.llm_client import LiteLLMClient
from graph_rag.config import Settings
from graph_rag.models import IngestTrigger
from graph_rag.wiring import build_offline_components

TEXT = b"Aa Works is part of Bb Group. Aa Works is located in Cc Land."


def process(components):
    components.object_store.put(components.bucket, "d.txt", TEXT)
    return components.orchestrator.process_document(
        IngestTrigger(bucket=components.bucket, object_key="d.txt")
    )


def test_llm_stages_ask_the_model_once_per_stage_and_cache_the_answers(tmp_path, monkeypatch):
    asked = []

    def model(self, *, model, prompt, **params):
        asked.append(prompt)
        return '{"clusters": [], "triples": []}'

    monkeypatch.setattr(LiteLLMClient, "_default_completion", model)
    settings = Settings(llm_cache_dir=str(tmp_path))
    first = build_offline_components(settings=settings, llm_stages=True)
    process(first)
    assert len(asked) == 2  # one coref call and one graph-build call
    assert first.llm_usage().calls == 2

    again = build_offline_components(settings=settings, llm_stages=True)
    process(again)  # a new pipeline over the same cache: no new calls
    assert len(asked) == 2 and again.llm_usage().cache_hits == 2


def test_the_default_pipeline_makes_no_provider_call(monkeypatch):
    def forbidden(self, **_):
        raise AssertionError("the offline pipeline must not call a model")

    monkeypatch.setattr(LiteLLMClient, "_default_completion", forbidden)
    assert process(build_offline_components()) is not None
