"""Sentence-transformer adapter for the ``Embedder`` port (V4, ADR-0004, B1).

Wraps ``sentence-transformers`` behind the :class:`~graph_rag.ports.Embedder`
port so entity linking (and later query-side seeding) scores over dense vectors
without knowing which model produced them. The default model
``BAAI/bge-small-en-v1.5`` emits **384-dim** vectors (:attr:`dim`), which pins the
ES ``dense_vector`` mapping in :mod:`graph_rag.adapters.es_entity_store`.

The model is HEAVY (it pulls ``torch``), so — mirroring
:class:`~graph_rag.stages.ner.SpacyNerStage` — it is loaded **lazily** on first
:meth:`embed`, never at construction or import. That keeps ``import`` and the
fakes-only fast suite offline and torch-free; the fast suite injects
:class:`~graph_rag.fakes.FakeEmbedder` instead. The real adapter is proved to
behave like the fake at the seam by the entity-store contract test (both expose
384-dim, deterministic, L2-normalized vectors so cosine == dot).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_rag.logging import get_logger

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["SentenceTransformerEmbedder"]

_logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Real :class:`~graph_rag.ports.Embedder` backed by ``sentence-transformers``.

    Loads the model once (lazily, on first :meth:`embed`) and returns one
    L2-normalized vector per input text, so cosine similarity equals the dot
    product — the form the ES ``dense_vector`` ``cosine`` similarity and the
    :class:`~graph_rag.fakes.FakeEmbedder` both assume. Deterministic for a fixed
    model + input.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        """Configure (but do not yet load) the embedder.

        Args:
            model: The ``sentence-transformers`` model name (``Settings.embed_model``).
            dim: The embedding dimension the model emits (``Settings.embed_dim``);
                384 for ``bge-small-en-v1.5`` (B1). Pins the ES ``dense_vector``
                mapping, so it is fixed configuration rather than inferred.
        """
        self._model_name = model
        self._dim = dim
        self._model = None  # lazily loaded on first embed()

    @classmethod
    def from_settings(cls, settings: Settings) -> SentenceTransformerEmbedder:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.embed_model`` and ``settings.embed_dim``.
        """
        return cls(model=settings.embed_model, dim=settings.embed_dim)

    @property
    def dim(self) -> int:
        """The embedding dimension (384 for ``bge-small-en-v1.5``)."""
        return self._dim

    def _load(self):  # type: ignore[no-untyped-def]
        """Load the sentence-transformer model once (idempotent, lazy).

        Imported inside the method so neither ``import graph_rag.adapters.embedder``
        nor the fast suite pulls in ``sentence-transformers`` / ``torch``.
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        _logger.info("loading sentence-transformer model %r", self._model_name)
        model = SentenceTransformer(self._model_name)
        self._model = model
        return model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized dense vector (length :attr:`dim`) per text, in order.

        ``normalize_embeddings=True`` makes each vector unit-length so cosine
        similarity equals the dot product — consistent with the fake and with the
        ES ``cosine`` ``dense_vector`` similarity. Deterministic for a fixed input.
        """
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in vector] for vector in vectors]
