"""Unit tests for the V7 gated prose synthesizer (N17) — fast, offline, ``$0``.

Covers the two things V7 owns at the unit seam:

* **The gate defaults OFF.** A synthesizer wired onto the retriever but a request
  with ``synthesize=False`` (the default) makes NO LLM call — proved on the call
  counter of a :class:`~graph_rag.fakes.FakeLLMClient` (``calls == 0``).
* **Prompt assembly is grounded.** :func:`~graph_rag.query.synthesis.build_synthesis_prompt`
  folds the subgraph nodes AND edges (with predicate + provenance) AND the
  supporting sentences into the prompt, and carries the grounding instruction.

Deterministic, ``$0``, no Docker / model / real LLM (ADR-0010).
"""

from __future__ import annotations

from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    FakeLLMClient,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
)
from graph_rag.models import (
    CanonicalEntity,
    EdgeProvenance,
    QueryRequest,
    QueryResponse,
    RankedNode,
    Subgraph,
    SupportingSentence,
    Triple,
)
from graph_rag.query.retriever import QueryRetriever
from graph_rag.query.synthesis import AnswerSynthesizer, build_synthesis_prompt


def _response() -> QueryResponse:
    """A small, fully-populated V6 response to assemble a prompt from."""
    node_a = CanonicalEntity(canonical_id="e-ada", name="Ada Lovelace", type="PERSON")
    node_b = CanonicalEntity(canonical_id="e-engine", name="Analytical Engine", type="PRODUCT")
    edge = Triple(
        subject_id="e-ada",
        predicate="WORKED_ON",
        object_id="e-engine",
        date="1843",
        provenance=EdgeProvenance(
            source_doc_id="doc1",
            sentence_index=0,
            source_sentence="Ada Lovelace worked on the Analytical Engine.",
        ),
    )
    top = RankedNode(canonical_id="e-ada", name="Ada Lovelace", type="PERSON", score=1.0)
    return QueryResponse(
        answer="Ada Lovelace",
        answer_entity=top,
        subgraph=Subgraph(nodes=[node_a, node_b], edges=[edge]),
        ranked_nodes=[top],
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


# --- Prompt assembly (pure) -------------------------------------------------


def test_build_synthesis_prompt_includes_nodes_edges_and_sentences() -> None:
    """The assembled prompt carries the subgraph nodes, edges AND supporting sentences."""
    question = "Who was Ada Lovelace?"
    prompt = build_synthesis_prompt(question, _response())

    # The question itself.
    assert question in prompt
    # Subgraph NODES (names + types).
    assert "Ada Lovelace" in prompt
    assert "Analytical Engine" in prompt
    assert "PRODUCT" in prompt
    # Subgraph EDGE: predicate + DATE qualifier + provenance (doc + source sentence).
    assert "WORKED_ON" in prompt
    assert "1843" in prompt
    assert "doc1" in prompt
    assert "Ada Lovelace worked on the Analytical Engine." in prompt


def test_build_synthesis_prompt_has_grounding_instruction() -> None:
    """The prompt instructs the model to ground strictly in the evidence (no outside knowledge)."""
    prompt = build_synthesis_prompt("q?", _response()).lower()
    assert "only" in prompt or "strictly" in prompt
    assert "outside knowledge" in prompt


def test_build_synthesis_prompt_handles_empty_evidence() -> None:
    """With no nodes/edges/sentences the prompt still renders (placeholders, no crash)."""
    empty = QueryResponse(answer=None, answer_entity=None, subgraph=Subgraph())
    prompt = build_synthesis_prompt("q?", empty)
    assert "(none)" in prompt


# --- The gate defaults OFF --------------------------------------------------


def _retriever_with_synth(
    llm: FakeLLMClient,
) -> tuple[
    QueryRetriever,
    FakeEmbedder,
    InMemoryEntityStore,
    InMemoryDocumentStore,
    InMemoryGraphStore,
]:
    """A retriever WITH a synthesizer wired over the given fake LLM + fresh fakes."""
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    document_store = InMemoryDocumentStore()
    graph_store = InMemoryGraphStore()
    retriever = QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=document_store,
        graph_store=graph_store,
        synthesizer=AnswerSynthesizer(llm),
    )
    return retriever, embedder, entity_store, document_store, graph_store


def _seed(
    embedder: FakeEmbedder,
    entity_store: InMemoryEntityStore,
    graph_store: InMemoryGraphStore,
) -> None:
    """Seed one vector-seedable entity + node so retrieval returns a real answer."""
    entity_store.upsert(
        CanonicalEntity(
            canonical_id="gh",
            name="Grace Hopper",
            type="PERSON",
            vector=embedder.embed(["grace hopper"])[0],
        )
    )
    graph_store.upsert_entities(
        [CanonicalEntity(canonical_id="gh", name="Grace Hopper", type="PERSON")]
    )


def test_gate_off_by_default_makes_no_llm_call() -> None:
    """A synthesizer is WIRED but ``synthesize=False`` (default) → NO LLM call, prose None."""
    llm = FakeLLMClient(completion="should never be produced")
    retriever, embedder, entity_store, _docs, graph_store = _retriever_with_synth(llm)
    _seed(embedder, entity_store, graph_store)

    response = retriever.retrieve(QueryRequest(question="grace hopper"))

    assert response.prose is None
    assert llm.calls == 0  # the gate defaults OFF — the LLM was never touched


def test_gate_on_synthesizes_prose_and_calls_llm() -> None:
    """``synthesize=True`` with a synthesizer wired → prose set, LLM called once."""
    llm = FakeLLMClient(completion="Grace Hopper was a computer scientist.")
    retriever, embedder, entity_store, _docs, graph_store = _retriever_with_synth(llm)
    _seed(embedder, entity_store, graph_store)

    response = retriever.retrieve(QueryRequest(question="grace hopper", synthesize=True))

    assert response.prose == "Grace Hopper was a computer scientist."
    assert llm.calls == 1


def test_gate_on_without_synthesizer_does_not_error() -> None:
    """``synthesize=True`` but NO synthesizer wired → prose stays None, no error."""
    embedder = FakeEmbedder()
    entity_store = InMemoryEntityStore()
    graph_store = InMemoryGraphStore()
    _seed(embedder, entity_store, graph_store)
    retriever = QueryRetriever.from_settings(
        Settings(),
        embedder=embedder,
        entity_store=entity_store,
        document_store=InMemoryDocumentStore(),
        graph_store=graph_store,
    )

    response = retriever.retrieve(QueryRequest(question="grace hopper", synthesize=True))

    assert response.prose is None
