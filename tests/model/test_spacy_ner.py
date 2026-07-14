"""Model-backed NER tests — real spaCy on ``en_core_web_sm`` (TESTING §5).

Marked ``model`` and excluded from the fast pre-push gate: these load a real spaCy
pipeline and prove :class:`~graph_rag.stages.ner.SpacyNerStage` behaves — curated
mentions whose char spans align to the source text, sentence segmentation, and
determinism. Also includes the **model-availability smoke test** (TESTING §5 V2
audit): the configured model loads (via the graceful fallback chain) and runs, so
a missing model fails fast here rather than mid-pipeline.

Skips cleanly when ``en_core_web_sm`` is not installed (``make models``). Tests pin
``en_core_web_sm`` explicitly so they are deterministic regardless of which heavier
models happen to be present.
"""

from __future__ import annotations

import pytest

from graph_rag.config import Settings
from graph_rag.models import CuratedType
from graph_rag.stages.ner import NerResult, SpacyNerStage

pytestmark = pytest.mark.model

# The curated set (must match graph_rag.models.CuratedType).
_CURATED_TYPES: set[str] = set(CuratedType.__args__)  # type: ignore[attr-defined]

# A fixed fixture doc with entities the small model detects reliably, spanning
# several curated types (person, org, location, date, nationality).
FIXTURE_TEXT = (
    "Barack Obama was born in Hawaii. He served as the 44th President of the "
    "United States. Apple announced a new product in 2020. German researchers "
    "praised the decision."
)


def _require_model(name: str = "en_core_web_sm") -> None:
    """Skip the whole module cleanly if the spaCy model ``name`` is not installed."""
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package(name):
        pytest.skip(f"spaCy model {name!r} not installed (run `make models`)")


@pytest.fixture(scope="module")
def stage() -> SpacyNerStage:
    """A SpacyNerStage pinned to en_core_web_sm (loaded once for the module)."""
    _require_model("en_core_web_sm")
    return SpacyNerStage(model="en_core_web_sm")


@pytest.fixture(scope="module")
def result(stage: SpacyNerStage) -> NerResult:
    """The NER result over the fixed fixture doc."""
    return stage.analyze(FIXTURE_TEXT)


def test_spans_align_to_source_text(result: NerResult) -> None:
    """Every mention's char span slices exactly its own text out of the raw doc."""
    assert result.mentions, "expected the small model to find at least one entity"
    for mention in result.mentions:
        assert FIXTURE_TEXT[mention.char_start : mention.char_end] == mention.text


def test_mentions_are_curated_types(result: NerResult) -> None:
    """Every returned mention carries a curated type (out-of-set labels dropped)."""
    for mention in result.mentions:
        assert mention.type in _CURATED_TYPES


def test_sentence_segmentation(result: NerResult) -> None:
    """Sentences are segmented, indexed in order, and align to the source text."""
    assert len(result.sentences) >= 2, "fixture has multiple sentences"
    for i, sent in enumerate(result.sentences):
        assert sent.index == i
        assert FIXTURE_TEXT[sent.char_start : sent.char_end] == sent.text


def test_deterministic_across_runs(stage: SpacyNerStage) -> None:
    """Two runs over the same text produce identical mentions + sentences."""
    first = stage.analyze(FIXTURE_TEXT)
    second = stage.analyze(FIXTURE_TEXT)
    assert first.mentions == second.mentions
    assert first.sentences == second.sentences


def test_model_availability_smoke() -> None:
    """Smoke test (TESTING §5): the configured model loads (via fallback) and runs.

    Uses ``SpacyNerStage.from_settings`` with the default configuration. If the
    configured ``en_core_web_trf`` is absent, the graceful fallback chain resolves
    to an installed model (e.g. ``en_core_web_sm``); the point is that *a* model
    loads and produces a result, so a totally-missing model fails fast.
    """
    _require_model("en_core_web_sm")
    stage = SpacyNerStage.from_settings(Settings())
    out = stage.analyze("Alice met Bob in Paris.")
    assert isinstance(out, NerResult)
    # It ran end-to-end: sentence segmentation always yields at least one sentence.
    assert len(out.sentences) >= 1
