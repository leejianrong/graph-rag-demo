"""Model-backed embedder test — real ``bge-small-en-v1.5`` (V4, B1, TESTING §5).

Marked ``model`` and excluded from the fast pre-push gate: this loads the real
``sentence-transformers`` model (which pulls ``torch``, the ``embed`` extra) and
proves :class:`~graph_rag.adapters.embedder.SentenceTransformerEmbedder` behaves at
the ``Embedder`` seam — 384-dim, deterministic, L2-normalized vectors whose cosine
similarity separates semantically-similar from dissimilar text.

Skips cleanly when ``sentence-transformers`` is not installed or the model cannot
be downloaded here (offline / air-gapped CI), so a missing model never fails the
suite.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.model


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors (== cosine for unit vectors)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    """A real :class:`SentenceTransformerEmbedder`, skipping if the model is unavailable."""
    pytest.importorskip("sentence_transformers")

    from graph_rag.adapters.embedder import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(model="BAAI/bge-small-en-v1.5", dim=384)
    try:
        # Force the lazy load / download now; skip cleanly if offline.
        emb.embed(["warmup"])
    except Exception as exc:  # noqa: BLE001 - model download/load may fail offline.
        pytest.skip(f"bge model unavailable (offline?): {exc}")
    return emb


def test_dim_is_384(embedder) -> None:  # type: ignore[no-untyped-def]
    """The embedder reports and emits 384-dim vectors (B1)."""
    assert embedder.dim == 384
    [vector] = embedder.embed(["hello world"])
    assert len(vector) == 384


def test_deterministic(embedder) -> None:  # type: ignore[no-untyped-def]
    """The same text embeds to the same vector across calls (deterministic)."""
    first = embedder.embed(["Apple announced a new product."])[0]
    second = embedder.embed(["Apple announced a new product."])[0]
    assert first == pytest.approx(second, abs=1e-6)


def test_l2_normalized(embedder) -> None:  # type: ignore[no-untyped-def]
    """Every vector is unit-length, so cosine similarity equals the dot product."""
    for vector in embedder.embed(["one text", "another entirely different text"]):
        norm = math.sqrt(sum(x * x for x in vector))
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_semantic_similarity_orders_correctly(embedder) -> None:  # type: ignore[no-untyped-def]
    """Similar strings score higher cosine than a clearly dissimilar one."""
    anchor, similar, dissimilar = embedder.embed(
        [
            "The cat sat on the mat.",
            "A kitten rested on the rug.",
            "Quarterly revenue exceeded analyst expectations.",
        ]
    )
    sim_score = _dot(anchor, similar)
    dissim_score = _dot(anchor, dissimilar)
    assert sim_score > dissim_score
