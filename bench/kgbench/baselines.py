"""Baseline retrievers over the rendered documents, behind the same interface as graph-rag.

A lift claim needs plain competitors on the same questions (ADR-0011). Each baseline indexes the
sentences of a set of ``kggen`` documents and returns ranked ``RetrievedSentence``s that carry the
fact ids the sentence expresses, so ``kgbench.retrieval`` scores them exactly like any other system.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from kggen.render import Document

from graph_rag.ports import Embedder
from kgbench.retrieval import RetrievedSentence

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def sentences_of(documents: Sequence[Document]) -> list[RetrievedSentence]:
    """Every sentence of every document, in document order, with the facts it expresses."""
    return [RetrievedSentence(d.id, s.text, s.fact_ids) for d in documents for s in d.sentences]


class BM25Retriever:
    """Okapi BM25 over sentences: deterministic, in-process, no network and no model.

    Ties break by position in the corpus, so the same documents and question always give the
    same ranking. Term statistics are over sentences, which is also what is returned.
    """

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._sentences = sentences_of(documents)
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(s.text)) for s in self._sentences]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._sentences)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score(self, query: Sequence[str], i: int) -> float:
        tf, length = self._tf[i], self._len[i]
        norm = self._k1 * (1 - self._b + self._b * length / self._avg) if self._avg else self._k1
        return sum(
            self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in set(query) if t in tf
        )

    def retrieve(self, question: str, k: int) -> list[RetrievedSentence]:
        query = tokenize(question)
        scored = [(self._score(query, i), i) for i in range(len(self._sentences))]
        ranked = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]))
        return [self._sentences[i] for _, i in ranked[:k]]


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


class DenseRetriever:
    """Dense-passage retrieval over sentences with the project's own ``Embedder`` port.

    The corpus is embedded once, in batches, when the retriever is built; each question costs one
    embedding call. Vectors are normalised here, so the score is cosine similarity whatever the
    embedder returns. It always returns the ``k`` nearest sentences (there is no score cut-off),
    and ties break by corpus position. No service beyond the embedder is needed.
    """

    def __init__(
        self, documents: Sequence[Document], embedder: Embedder, *, batch_size: int = 64
    ) -> None:
        self._sentences = sentences_of(documents)
        self._embedder = embedder
        self._vectors: list[list[float]] = []
        texts = [s.text for s in self._sentences]
        for i in range(0, len(texts), batch_size):
            self._vectors += [_unit(v) for v in embedder.embed(texts[i : i + batch_size])]

    def retrieve(self, question: str, k: int) -> list[RetrievedSentence]:
        if not self._sentences:
            return []
        query = _unit(self._embedder.embed([question])[0])
        scored = [
            (sum(a * b for a, b in zip(query, v, strict=True)), i)
            for i, v in enumerate(self._vectors)
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._sentences[i] for _, i in scored[:k]]
