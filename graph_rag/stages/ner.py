"""The NER stage (N6) — typed mentions + char spans + sentences (ADR-0002).

The stage is the first real enrichment step. It runs behind the :class:`NerStage`
interface so the orchestrator takes it as a constructor-injected collaborator (the
testability seam, ADR-0010): the fast suite injects
:class:`~graph_rag.fakes.FakeNerStage` (canned output, no model download) while the
real stack injects :class:`SpacyNerStage`.

:class:`SpacyNerStage` runs spaCy in ONE pass to produce, over the raw document
text:

* typed :class:`~graph_rag.models.Mention` s with character offsets, mapping
  spaCy's OntoNotes labels down to the curated set (``GPE``+``LOC`` → ``LOCATION``;
  labels outside the set are dropped);
* segmented :class:`~graph_rag.models.Sentence` s.

No LLM is involved and the result is deterministic for a fixed model + input. The
result is carried in-memory by the orchestrator and is NOT persisted until the V4
EL checkpoint (ARCHITECTURE §4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from graph_rag.logging import get_logger
from graph_rag.models import CuratedType, Mention, Sentence

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = [
    "NerResult",
    "NerStage",
    "SpacyNerStage",
    "curated_type",
    "SPACY_LABEL_TO_CURATED",
]

_logger = get_logger(__name__)

# spaCy OntoNotes label -> curated type (ADR-0002). ``GPE`` (geo-political
# entity) and ``LOC`` (non-GPE location) both collapse to ``LOCATION``; any label
# not in this map is dropped by :func:`curated_type`.
SPACY_LABEL_TO_CURATED: dict[str, CuratedType] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "EVENT": "EVENT",
    "NORP": "NORP",
    "PRODUCT": "PRODUCT",
}

# Ordered fallback chain (ADR-0002): the accurate transformer model first, then
# progressively lighter pipelines. The configured model is tried before these.
_FALLBACK_MODELS: tuple[str, ...] = ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm")


def curated_type(spacy_label: str) -> CuratedType | None:
    """Map a spaCy entity label to a curated type, or ``None`` to drop it.

    Merges ``GPE`` and ``LOC`` into ``LOCATION`` (ADR-0002). A pure function so
    the label-mapping policy is unit-testable with no model load.

    Args:
        spacy_label: A spaCy ``ent.label_`` (e.g. ``"GPE"``, ``"PERSON"``).

    Returns:
        The curated type, or ``None`` when the label is outside the curated set
        (the mention should be dropped).
    """
    return SPACY_LABEL_TO_CURATED.get(spacy_label)


class NerResult:
    """The NER stage's output — mentions + sentences from one pass.

    A plain carrier so the :class:`NerStage` contract is explicit and the
    orchestrator can copy both lists onto the
    :class:`~graph_rag.models.PipelineResult`.
    """

    __slots__ = ("mentions", "sentences")

    def __init__(self, mentions: list[Mention], sentences: list[Sentence]) -> None:
        """Bundle the mentions and sentences produced for one document."""
        self.mentions = mentions
        self.sentences = sentences


@runtime_checkable
class NerStage(Protocol):
    """The NER stage seam: turn raw text into typed mentions + sentences.

    Implementations must be deterministic for a fixed input and must return
    character offsets that index into the *same* ``text`` passed in (so
    ``text[m.char_start:m.char_end] == m.text``). Constructor-injected into the
    orchestrator; the fast suite injects a fake instead of loading a model.
    """

    def analyze(self, text: str) -> NerResult:
        """Return the mentions + sentences for ``text`` in a single pass."""
        ...


class SpacyNerStage:
    """Real :class:`NerStage` backed by spaCy (ADR-0002).

    Loads a spaCy pipeline once (lazily, on first :meth:`analyze`) and runs it in
    one pass per document, producing curated-type mentions with char spans and
    sentence segmentation. The model is resolved via a graceful fallback chain
    (configured model → ``en_core_web_trf`` → ``en_core_web_lg`` →
    ``en_core_web_sm``); the first installed model wins, with a clear log line
    when a preferred model is missing and a fallback is used.
    """

    def __init__(self, model: str = "en_core_web_trf") -> None:
        """Configure (but do not yet load) the stage.

        Args:
            model: The preferred spaCy pipeline name (from ``Settings.ner_model``).
        """
        self._model = model
        self._nlp = None  # lazily loaded on first analyze()

    @classmethod
    def from_settings(cls, settings: Settings) -> SpacyNerStage:
        """Construct from :class:`~graph_rag.config.Settings` (``settings.ner_model``)."""
        return cls(model=settings.ner_model)

    def _load(self):  # type: ignore[no-untyped-def]
        """Load the spaCy pipeline, walking the fallback chain (idempotent).

        Tries the configured model first, then each fallback in turn, skipping
        duplicates. Logs a warning when a preferred model is unavailable and a
        lighter one is used.

        Raises:
            RuntimeError: If none of the candidate models can be loaded.
        """
        if self._nlp is not None:
            return self._nlp

        import spacy

        # Configured model first, then the ordered fallbacks (de-duplicated).
        candidates: list[str] = [self._model]
        candidates += [m for m in _FALLBACK_MODELS if m != self._model]

        for i, name in enumerate(candidates):
            try:
                nlp = spacy.load(name)
            except OSError:
                _logger.warning("spaCy model %r is not installed; trying the next fallback", name)
                continue
            if i > 0:
                _logger.warning(
                    "NER falling back to spaCy model %r (preferred %r unavailable)",
                    name,
                    self._model,
                )
            else:
                _logger.info("loaded spaCy NER model %r", name)
            self._nlp = nlp
            return nlp

        raise RuntimeError(
            "no spaCy NER model available; tried "
            + ", ".join(candidates)
            + " (run `make models` or `python -m spacy download en_core_web_sm`)"
        )

    def analyze(self, text: str) -> NerResult:
        """Run spaCy once over ``text`` → curated mentions + sentences.

        Entity labels outside the curated set are dropped (see
        :func:`curated_type`). Offsets are spaCy's char offsets, which index into
        the input ``text``.
        """
        nlp = self._load()
        doc = nlp(text)

        mentions: list[Mention] = []
        for ent in doc.ents:
            ctype = curated_type(ent.label_)
            if ctype is None:
                continue
            mentions.append(
                Mention(
                    text=ent.text,
                    type=ctype,
                    char_start=ent.start_char,
                    char_end=ent.end_char,
                )
            )

        sentences: list[Sentence] = [
            Sentence(
                text=sent.text,
                char_start=sent.start_char,
                char_end=sent.end_char,
                index=idx,
            )
            for idx, sent in enumerate(doc.sents)
        ]

        return NerResult(mentions=mentions, sentences=sentences)
