"""The coreference stage (N7) — within-document cluster map (ADR-0003).

The second enrichment stage and the pipeline's **first LLM use**. It collapses
within-document references — repeated names and pronouns (``"she"``, ``"they"``,
``"it"``) — into clusters, each mapping its coreferent surface forms onto a chosen
in-document canonical. The output is a **non-destructive cluster map**
(:class:`~graph_rag.models.CorefCluster`): the raw text is preserved, so character
spans stay valid for provenance and each document's clusters become the doc-level
entities handed to entity linking at V4.

Like the NER stage (ADR-0002), coref runs behind the :class:`CorefStage` interface
and is constructor-injected into the orchestrator (the testability seam,
ADR-0010): the fast suite injects :class:`FakeCorefStage` (or an
:class:`LLMCorefStage` backed by ``FakeLLMClient``) so the gate stays LLM-free and
``$0``; the real stack injects an :class:`LLMCorefStage` over the LiteLLM client.

Cross-document identity is explicitly *not* coref's job — that is entity linking
(V4, ADR-0004). Coref is intra-document only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from graph_rag.logging import get_logger
from graph_rag.models import ClusterMap, CorefCluster

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import Mention
    from graph_rag.ports import LLMClient

__all__ = [
    "CorefStage",
    "LLMCorefStage",
    "FakeCorefStage",
    "build_coref_prompt",
]

_logger = get_logger(__name__)


def build_coref_prompt(text: str, mentions: list[Mention]) -> str:
    """Render the coref prompt from the document text + the NER mentions.

    A pure function so the prompt is deterministic (stable cache key) and
    unit-inspectable. It asks the model to group only *within-document*
    coreferent surface forms — including pronouns — onto a canonical drawn
    verbatim from the text, and to preserve original text (never rewrite it).

    Args:
        text: The raw document text (offsets/spans reference this).
        mentions: The typed NER mentions (N6) offered as anchor surface forms.

    Returns:
        The rendered prompt string (the schema instruction is appended by the
        client's structured-output call).
    """
    listed = "\n".join(f"- {m.text} ({m.type})" for m in mentions) or "- (none)"
    return (
        "You are resolving coreference WITHIN a single document. Group the surface "
        "forms that refer to the same real-world entity — including pronouns "
        '("she", "they", "it") and repeated names — into clusters. For each '
        "cluster choose a canonical surface form copied verbatim from the text "
        "(prefer the most complete proper-name mention). Preserve the original "
        "text; do NOT rewrite it. Only group mentions that appear in this "
        "document.\n\n"
        f"Document text:\n{text}\n\n"
        f"Detected entity mentions:\n{listed}"
    )


@runtime_checkable
class CorefStage(Protocol):
    """The coref stage seam: raw text + mentions → a within-doc cluster map.

    Implementations must be **non-destructive** — return a cluster map keyed on
    surface forms taken from the text, never a rewrite. Constructor-injected into
    the orchestrator; the fast suite injects a fake instead of calling an LLM.
    """

    def resolve(self, text: str, mentions: list[Mention]) -> list[CorefCluster]:
        """Return the within-document coref clusters for ``text``."""
        ...


class LLMCorefStage:
    """Real :class:`CorefStage` — LLM-backed structured output (ADR-0003/0008).

    Calls the injected :class:`~graph_rag.ports.LLMClient` with a structured-output
    prompt and validates the response into a
    :class:`~graph_rag.models.ClusterMap`, so the provider/model, response cache
    and retry all live in the client (V3-active).
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Wire the stage to an LLM client.

        Args:
            llm_client: The client to call. Defaults to a real
                :class:`~graph_rag.adapters.llm_client.LiteLLMClient` built from
                :class:`~graph_rag.config.Settings`; the fast suite injects a fake
                so no provider is called.
        """
        if llm_client is None:
            from graph_rag.adapters.llm_client import LiteLLMClient
            from graph_rag.config import get_settings

            llm_client = LiteLLMClient.from_settings(get_settings())
        self._llm = llm_client

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMCorefStage:
        """Construct with a :class:`LiteLLMClient` built from ``settings`` (B6 model)."""
        from graph_rag.adapters.llm_client import LiteLLMClient

        return cls(LiteLLMClient.from_settings(settings))

    def resolve(self, text: str, mentions: list[Mention]) -> list[CorefCluster]:
        """Run the LLM once → a validated, non-destructive cluster map."""
        prompt = build_coref_prompt(text, mentions)
        cluster_map = self._llm.structured(prompt, ClusterMap)
        return cluster_map.clusters


class FakeCorefStage:
    """Canned :class:`CorefStage` for the fast suite (no LLM call).

    Returns configurable canned clusters so the fast E2E proves the wiring + the
    :class:`~graph_rag.models.PipelineResult` carry (not coref quality) —
    deterministic, ``$0``, offline. The last ``(text, mentions)`` passed to
    :meth:`resolve` are recorded for assertions.
    """

    def __init__(self, clusters: list[CorefCluster] | None = None) -> None:
        """Configure the canned clusters returned from every :meth:`resolve`."""
        self._clusters = list(clusters or [])
        self.last_text: str | None = None
        self.last_mentions: list[Mention] | None = None

    def resolve(self, text: str, mentions: list[Mention]) -> list[CorefCluster]:
        """Return the canned clusters, recording the inputs; ignores content."""
        self.last_text = text
        self.last_mentions = list(mentions)
        return list(self._clusters)
