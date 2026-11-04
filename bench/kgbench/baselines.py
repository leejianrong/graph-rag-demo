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
