"""graph-rag as a ``Retriever``: its evidence mapped back to gold sentences and their fact ids.

graph-rag answers with two kinds of evidence. Its *supporting sentences* are a nearest-neighbour
search over sentence vectors, so on their own they behave like the dense baseline. What a graph adds
is the *edges* of the retrieved subgraph, each with the sentence it was built from. Two modes keep
that visible instead of hiding it:

* ``passages``: only the supporting sentences, in the order graph-rag returns them;
* ``graph``: the sentences behind the subgraph's edges first, ordered by how highly graph-rag ranked
  the edge's endpoints, then the supporting sentences to fill the list.

Evidence is mapped to the generator's sentences by document and sentence text (falling back to the
sentence index), so each returned sentence carries the facts it truly expresses. Evidence that
matches no gold sentence is kept with no fact ids, so it still takes a slot and counts as noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from kggen.render import Document

from graph_rag.ids import document_id
from graph_rag.models import QueryRequest
from kgbench.extraction import ingest
from kgbench.retrieval import RetrievedSentence

Mode = Literal["passages", "graph"]


class GraphRagRetriever:
    """Ingest ``documents`` into wired graph-rag components, then answer questions from them."""

    def __init__(self, components, documents: Sequence[Document], *, mode: Mode = "graph") -> None:
        self._components = components
        self.mode = mode
        self._by_hash = {
            document_id(components.bucket, f"{doc.id}.txt"): doc
            for doc, _ in self._ingest(documents)
        }

    def _ingest(self, documents: Sequence[Document]):
        return ingest(self._components, tuple(documents))

    def _locate(self, doc_hash: str, text: str, index: int | None) -> RetrievedSentence | None:
        doc = self._by_hash.get(doc_hash)
        if doc is None:
            return None
        wanted = text.strip()
        for s in doc.sentences:
            if s.text.strip() == wanted:
                return RetrievedSentence(doc.id, s.text, s.fact_ids)
        if index is not None and 0 <= index < len(doc.sentences):
            s = doc.sentences[index]
            return RetrievedSentence(doc.id, s.text, s.fact_ids)
        return RetrievedSentence(doc.id, text, ())

    def retrieve(self, question: str, k: int) -> list[RetrievedSentence]:
        response = self._components.retriever.retrieve(QueryRequest(question=question))
        ranked: list[RetrievedSentence] = []
        if self.mode == "graph":
            rank = {n.canonical_id: i for i, n in enumerate(response.ranked_nodes)}
            worst = len(rank)
            edges = sorted(
                enumerate(response.subgraph.edges),
                key=lambda ie: (
                    min(rank.get(ie[1].subject_id, worst), rank.get(ie[1].object_id, worst)),
                    ie[0],
                ),
            )
            for _, edge in edges:
                p = edge.provenance
                found = self._locate(p.source_doc_id, p.source_sentence, p.sentence_index)
                if found is not None:
                    ranked.append(found)
        for s in response.supporting_sentences:
            found = self._locate(s.document_id, s.text, s.sentence_index)
            if found is not None:
                ranked.append(found)
        seen: set[tuple[str, str]] = set()
        unique = []
        for s in ranked:
            if (s.doc, s.text) not in seen:
                seen.add((s.doc, s.text))
                unique.append(s)
        return unique[:k]
