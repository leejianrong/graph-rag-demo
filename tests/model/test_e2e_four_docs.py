"""Four documents and three questions end to end, replayed from recorded Haiku answers (GRA-163).

The documents and questions are the ones that found five real bugs in two live runs (a name merge
of two organisations in one sentence, a pronoun that split a person, ``/query`` answering with the
entity the question names). This replays them with no key and no endpoint: ``tests/model/four_docs/
llm-cache`` holds the coreference and graph-build answers Claude Haiku gave through
semula, in the LLM client's own cache format, so the client answers from it and never calls out.

Marked ``model`` because the recording used spaCy ``en_core_web_trf`` (CI installs it):
the coreference prompt embeds the mentions NER found, so a different NER model gives
different prompts, misses the cache and (with no endpoint) fails. To re-record see docs/TESTING.md.

The `linked` fixture swaps in the product linker (merge on similarity, name guard on), fed an
embedder that scores every pair 1.0 as a sentence-in-context vector can, so only the GRA-157 name
guard keeps Acme Logistics and Northwind Shipping apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graph_rag.config import Settings
from graph_rag.fakes import FakeEmbedder
from graph_rag.models import IngestTrigger, QueryRequest
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.stages.entity_linking import ELResult, EntityLinkingStage
from graph_rag.wiring import BenchmarkComponents, build_offline_components

pytestmark = pytest.mark.model

DATA = Path(__file__).parent / "four_docs"


class _EverythingAlike:
    """Every text embeds to the same unit vector: the worst case for a similarity merge."""

    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 7 for _ in texts]


class _RecordingLinker:
    """The product linker, remembering what it returned for each document, in order."""

    def __init__(self, inner: EntityLinkingStage) -> None:
        self._inner = inner
        self.results: list[ELResult] = []

    def link(self, *args: Any, **kwargs: Any) -> ELResult:
        result = self._inner.link(*args, **kwargs)
        self.results.append(result)
        return result


def _assert_kept_apart(result: ELResult, stem: str) -> None:
    """A document's link result, checked before kg-build runs (which needs the recorded cache).

    Without the GRA-157 name guard the entities merge and this fails here, on the merge itself,
    and not later as a prompt missing the recorded cache.
    """
    assert not result.merged, f"{stem}.txt: entities merged into one another: {result.merged}"
    ids = {link.mention_text: link.canonical_id for link in result.links}
    clashes = {i: [n for n, j in ids.items() if j == i] for i in ids.values()}
    clashes = {i: names for i, names in clashes.items() if len(names) > 1}
    assert not clashes, f"{stem}.txt: different entities linked to one canonical id: {clashes}"
    if stem == "c":
        assert ids["Acme Logistics"] != ids["Northwind Shipping"], "Acme and Northwind merged"


def _ingest(product_linker: bool) -> BenchmarkComponents:
    try:
        import spacy

        spacy.load("en_core_web_trf")
    except Exception as exc:  # noqa: BLE001 - model not installed
        pytest.skip(f"en_core_web_trf unavailable (make models-trf): {exc}")
    settings = Settings(
        coref_model="openai/semula",
        kg_build_model="openai/semula",
        ner_model="en_core_web_trf",
        llm_cache_dir=str(DATA / "llm-cache"),
        openai_api_key="unused",
    )
    components = build_offline_components(
        settings=settings, embedder=FakeEmbedder(), llm_stages=True, spacy_ner=True
    )
    recorder: _RecordingLinker | None = None
    if product_linker:  # similarity merge + name guard, not the offline merge-by-name linker
        recorder = _RecordingLinker(EntityLinkingStage(components.entity_store, _EverythingAlike()))
        components.orchestrator._entity_linking_stage = recorder  # type: ignore[attr-defined]
    for stem in "abcd":
        text = (DATA / "docs" / f"{stem}.txt").read_bytes()
        components.object_store.put(DEFAULT_SCOPE, components.bucket, f"{stem}.txt", text)
        result = components.orchestrator.process_document(
            IngestTrigger(bucket=components.bucket, object_key=f"{stem}.txt")
        )
        if recorder is not None:
            _assert_kept_apart(recorder.results[-1], stem)
        assert result is not None, (
            f"{stem}.txt was dropped: a prompt missed the recorded cache (a new NER model, or a "
            "change to what the LLM stages are shown); see docs/TESTING.md to re-record"
        )
    return components


@pytest.fixture(scope="module")
def ingested() -> BenchmarkComponents:
    """The offline linker (merge by name) and the real FakeEmbedder, so /query seeds by name."""
    return _ingest(product_linker=False)


@pytest.fixture(scope="module")
def linked() -> BenchmarkComponents:
    """The product's linker, with an embedder under which only the name guard separates entities."""
    return _ingest(product_linker=True)


def _edges(components: BenchmarkComponents) -> set[tuple[str, str, str]]:
    names = {e.canonical_id: e.name for e in components.entity_store.all(DEFAULT_SCOPE)}
    return {
        (names[t.subject_id], t.predicate, names[t.object_id])
        for cid in names
        for t in components.graph_store.get_node_edges(DEFAULT_SCOPE, cid)
        if t.subject_id == cid
    }


def test_replays_with_no_provider_call(ingested: BenchmarkComponents) -> None:
    usage = ingested.llm_usage()
    assert usage.calls == 0 and usage.cache_hits == 8


def test_one_entity_each_for_the_organisations(linked: BenchmarkComponents) -> None:
    names = [e.name for e in linked.entity_store.all(DEFAULT_SCOPE)]
    for org in ("Helix Robotics", "Northwind Shipping", "Acme Logistics"):
        assert names.count(org) == 1, org
    assert len(names) == 11  # 4 people, 4 orgs, 2 places, 2 years: nothing duplicated or merged


def test_key_edges(linked: BenchmarkComponents) -> None:
    edges = _edges(linked)
    assert ("Pieter de Vries", "WORKS_FOR", "Helix Robotics") in edges
    assert ("Maria Santos", "WORKS_FOR", "Helix Robotics") in edges
    assert ("Acme Logistics", "ACQUIRED", "Northwind Shipping") in edges  # the buyer is the subject
    assert ("Northwind Shipping", "LOCATED_IN", "Hamburg") in edges
    assert ("Helix Robotics", "LOCATED_IN", "Rotterdam") in edges
    assert not any(s == "Northwind Shipping" and p == "ACQUIRED" for s, p, _ in edges)


@pytest.mark.parametrize("case", json.loads((DATA / "questions.json").read_text()))
def test_questions_answer_with_the_other_end_not_the_named_entity(
    ingested: BenchmarkComponents, case: dict[str, object]
) -> None:
    response = ingested.retriever.retrieve(QueryRequest(question=str(case["question"])))
    assert response.answer_entity is not None
    assert response.answer_entity.name in case["answers"]  # type: ignore[operator]
