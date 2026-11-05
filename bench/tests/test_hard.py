"""The committed hard-tiny dataset and its recorded LLM-stage answers replay offline (GRA-117)."""

from __future__ import annotations

from pathlib import Path

from kgbench import hard
from kgbench.extraction import gold_from, run_pipeline, score
from kggen.spec import load_spec
from kggen.world import generate

from graph_rag.config import Settings
from graph_rag.fakes import FakeEmbedder
from graph_rag.wiring import build_offline_components

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "datagen" / "examples" / "hard-tiny"
CACHE = ROOT / "bench" / "kgbench" / "data" / "llm-cache-hard-tiny"


def test_recorded_haiku_answers_replay_with_no_provider_call_and_give_typed_triples():
    spec_path, documents = hard.load(FOLDER)
    world = generate(load_spec(spec_path))
    # an unreachable endpoint and no key: any cache miss would fail, not ask a model
    settings = Settings(
        coref_model="openai/claude-haiku",
        kg_build_model="openai/claude-haiku",
        llm_cache_dir=str(CACHE),
        openai_api_key="unused",
    )
    components = build_offline_components(
        settings=settings, embedder=FakeEmbedder(), llm_stages=True
    )
    report = score(gold_from(world, documents), run_pipeline(components, documents))
    assert components.llm_usage().calls == 0 and components.llm_usage().cache_hits > 0
    assert hard.typed(report) and report.triples.f1 > 0.9  # the offline builder scores 0 here


import pytest  # noqa: E402


@pytest.mark.model
def test_hard_small_replays_through_llm_and_spacy_with_no_provider_call():
    folder = ROOT / "datagen" / "examples" / "hard-small"
    spec_path, documents = hard.load(folder)
    world = generate(load_spec(spec_path))
    settings = Settings(
        coref_model="openai/claude-haiku",
        kg_build_model="openai/claude-haiku",
        llm_cache_dir=str(ROOT / "bench" / "kgbench" / "data" / "llm-cache-hard-small"),
        ner_model="en_core_web_sm",
        openai_api_key="unused",
    )
    components = build_offline_components(
        settings=settings, embedder=FakeEmbedder(), llm_stages=True, spacy_ner=True
    )
    report = score(gold_from(world, documents), run_pipeline(components, documents))
    assert components.llm_usage().calls == 0 and components.llm_usage().cache_hits > 200
    assert hard.typed(report) and 0.7 < report.triples.f1 < 0.9
