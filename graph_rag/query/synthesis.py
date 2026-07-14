"""The V7 gated prose synthesizer (N17) — the OPTIONAL LLM answer mode (ADR-0009).

The single place an LLM enters the query path. V6 retrieval
(:class:`~graph_rag.query.retriever.QueryRetriever`) is deterministic and ``$0``;
V7 adds an OPTIONAL, gated step on top of it: when ``POST /query`` is called with
``synthesize=true``, this stage turns the *already-retrieved* evidence — the ranked
subgraph (nodes + provenance-carrying edges) and the supporting sentences — into
concise prose grounded **strictly** in that evidence (ADR-0007/0009,
ARCHITECTURE §6).

The gate defaults OFF: retrieval never constructs or calls this synthesizer unless
the request asks for it, so the core path stays free and deterministic. Synthesis
adds NO new retrieval — it reads only what V6 already returned, so the prose is
traceable to the same subgraph + sentences the response already carries.

Like every LLM use in the pipeline, the provider call rides the injectable
:class:`~graph_rag.ports.LLMClient` seam (ADR-0010): the fast suite injects a
:class:`~graph_rag.fakes.FakeLLMClient` (canned prose, ``$0``, offline); the real
stack injects a :class:`~graph_rag.adapters.llm_client.LiteLLMClient` pinned to
``Settings.synthesis_model`` (B6 — a fuller model is reserved for synthesis), all
served through the same persistent response cache.

Prompt assembly is the pure, unit-testable :func:`build_synthesis_prompt`; the
stage only wires the client and calls :meth:`~graph_rag.ports.LLMClient.complete`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_rag.logging import get_logger

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import QueryResponse
    from graph_rag.ports import LLMClient

__all__ = ["AnswerSynthesizer", "build_synthesis_prompt"]

_logger = get_logger(__name__)


def build_synthesis_prompt(question: str, response: QueryResponse) -> str:
    """Render the synthesis prompt from the RETRIEVED EVIDENCE only (pure function).

    Assembles the question, the ranked subgraph (nodes plus each edge's
    subject/predicate/object with its DATE qualifier and source-document
    provenance) and the supporting sentences (with their source document + offsets)
    into one prompt, and instructs the model to answer **grounded strictly in that
    evidence** — no outside knowledge. Pure and side-effect-free so the prompt is
    deterministic (stable LLM cache key) and directly unit-inspectable.

    Args:
        question: The natural-language question being answered.
        response: The V6 :class:`~graph_rag.models.QueryResponse` — the evidence
            (subgraph + supporting sentences) the prose must be grounded in.

    Returns:
        The rendered prompt string handed to :meth:`~graph_rag.ports.LLMClient.complete`.
    """
    # A canonical_id -> name map so edges can be shown as readable names while the
    # graph identity stays traceable.
    names = {node.canonical_id: node.name for node in response.subgraph.nodes}

    nodes_block = (
        "\n".join(
            f"- {node.name} ({node.type}) [id={node.canonical_id}]"
            for node in response.subgraph.nodes
        )
        or "- (none)"
    )

    edge_lines: list[str] = []
    for edge in response.subgraph.edges:
        subject = names.get(edge.subject_id, edge.subject_id)
        obj = names.get(edge.object_id, edge.object_id)
        date = f", date={edge.date}" if edge.date else ""
        edge_lines.append(
            f"- {subject} --{edge.predicate}--> {obj}{date} "
            f"(source: {edge.provenance.source_doc_id}, "
            f"sentence {edge.provenance.sentence_index}: "
            f'"{edge.provenance.source_sentence}")'
        )
    edges_block = "\n".join(edge_lines) or "- (none)"

    sentence_lines = [
        f'- "{sentence.text}" (source: {sentence.document_id}, sentence {sentence.sentence_index})'
        for sentence in response.supporting_sentences
    ]
    sentences_block = "\n".join(sentence_lines) or "- (none)"

    return (
        "You are answering a question using ONLY the retrieved evidence below — a "
        "knowledge-graph subgraph and supporting sentences drawn from the source "
        "documents. Answer the question in concise, factual prose grounded STRICTLY "
        "in this evidence. Do NOT use any outside knowledge, and do NOT invent facts "
        "that are not supported by the evidence. If the evidence does not answer the "
        "question, say so plainly.\n\n"
        f"Question:\n{question}\n\n"
        f"Knowledge-graph entities:\n{nodes_block}\n\n"
        f"Knowledge-graph relations (with provenance):\n{edges_block}\n\n"
        f"Supporting sentences (with source):\n{sentences_block}\n\n"
        "Answer:"
    )


class AnswerSynthesizer:
    """The gated prose synthesizer (N17) — turn retrieved evidence into prose (V7).

    Constructor-injected with the :class:`~graph_rag.ports.LLMClient` it calls, so
    the provider/model, response cache and retry all live in the client (ADR-0010).
    Use :meth:`from_settings` to build one pinned to ``Settings.synthesis_model``
    (B6). This is the ONLY LLM in the query path and it runs only when the caller
    gates it on (``request.synthesize``) — see
    :class:`~graph_rag.query.retriever.QueryRetriever`.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Wire the synthesizer to its LLM client.

        Args:
            llm_client: The client the prose is generated through (a
                :class:`~graph_rag.adapters.llm_client.LiteLLMClient` in prod, a
                :class:`~graph_rag.fakes.FakeLLMClient` in the fast suite).
        """
        self._llm = llm_client

    @classmethod
    def from_settings(
        cls, settings: Settings, *, llm_client: LLMClient | None = None
    ) -> AnswerSynthesizer:
        """Construct from :class:`~graph_rag.config.Settings`.

        By default builds a :class:`~graph_rag.adapters.llm_client.LiteLLMClient`
        pinned to ``settings.synthesis_model`` (B6 — the fuller model reserved for
        synthesis, ARCHITECTURE §6), sharing the LLM cache dir, retry budget and
        env-sourced API key. Mirrors how
        :meth:`~graph_rag.stages.kg_build.KgBuildStage.from_settings` pins its own
        stage model rather than reusing another stage's. Pass ``llm_client`` to
        inject a pre-built client (e.g. the shared client from the composition
        root, or a fake in tests).

        Args:
            settings: The runtime configuration (provides ``synthesis_model``).
            llm_client: An optional pre-built client to use instead of building one.

        Returns:
            A wired :class:`AnswerSynthesizer`.
        """
        if llm_client is None:
            from graph_rag.adapters.llm_client import LiteLLMClient

            llm_client = LiteLLMClient(
                model=settings.synthesis_model,
                cache_dir=settings.llm_cache_dir,
                max_retries=settings.llm_max_retries,
                api_key=settings.openai_api_key,
            )
        return cls(llm_client)

    def synthesize(self, *, question: str, response: QueryResponse) -> str:
        """Return prose answering ``question``, grounded in ``response``'s evidence.

        Assembles the prompt from the retrieved evidence via
        :func:`build_synthesis_prompt` (the subgraph + supporting sentences only —
        no new retrieval, no outside knowledge) and returns the model completion,
        served through the client's persistent response cache (a repeated call is a
        ``$0`` cache hit).

        Args:
            question: The natural-language question being answered.
            response: The V6 :class:`~graph_rag.models.QueryResponse` whose subgraph
                + supporting sentences the prose is grounded in.

        Returns:
            The grounded prose answer.
        """
        prompt = build_synthesis_prompt(question, response)
        prose = self._llm.complete(prompt)
        _logger.info(
            "synthesized prose answer: question=%r subgraph=%d node(s)/%d edge(s), "
            "%d supporting sentence(s), %d char(s)",
            question,
            len(response.subgraph.nodes),
            len(response.subgraph.edges),
            len(response.supporting_sentences),
            len(prose),
        )
        return prose
