"""The benchmark harness (N18) — build the graph once, query the subset, score (ADR-0009).

Drives the capability measurement end-to-end over injected
:class:`~graph_rag.benchmark.pipeline.BenchmarkComponents`:

1. **Ingest** every example's context paragraphs through the V1 ingestion path
   (:meth:`~graph_rag.orchestrator.Orchestrator.process_document`) in a **FIXED
   order** (examples sorted by id, paragraphs in listed order; each unique title
   ingested once). Entity linking is order-sensitive (ADR-0004), so the fixed
   order makes the constructed graph — and therefore the scores — reproducible
   (ADR-0009). The graph is built **ONCE**: a second :meth:`run` reuses it.
2. **Query** each example's ``question`` through the V6 retriever
   (:meth:`~graph_rag.query.retriever.QueryRetriever.retrieve`) — the non-LLM,
   ``$0`` retrieval path.
3. **Score** (non-LLM): map the predicted ``answer_entity`` (its ``name`` +
   ``aliases``) to answer EM / token-F1 against the gold answer + aliases, and map
   the returned ``supporting_sentences`` (their ``(title, sentence_index)``) to
   supporting-fact P/R/F1 against the gold supporting facts.

The result carries per-question + aggregate metrics plus an ``llm_calls`` count
**for that run**: the first run pays the ingestion LLM cost, a warm re-run (graph
already built, retrieval non-LLM) reports ``0`` — so the ~$0 re-run is observable
(ADR-0008/0009).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graph_rag.benchmark.metrics import aggregate, exact_match, supporting_fact_prf, token_f1
from graph_rag.ids import document_id
from graph_rag.logging import get_logger
from graph_rag.models import IngestTrigger, QueryRequest

if TYPE_CHECKING:
    from graph_rag.benchmark.dataset import BenchmarkExample
    from graph_rag.benchmark.pipeline import BenchmarkComponents

__all__ = ["QuestionScore", "BenchmarkResult", "BenchmarkHarness"]

_logger = get_logger(__name__)


@dataclass
class QuestionScore:
    """Per-question scoring row (answer EM/F1 + supporting-fact P/R/F1)."""

    id: str
    question: str
    predicted_answer: str | None
    gold_answers: list[str]
    exact_match: float
    token_f1: float
    supporting_fact_precision: float
    supporting_fact_recall: float
    supporting_fact_f1: float

    def as_metrics(self) -> dict[str, float]:
        """Return just the numeric metrics (for :func:`~graph_rag.benchmark.metrics.aggregate`)."""
        return {
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "supporting_fact_precision": self.supporting_fact_precision,
            "supporting_fact_recall": self.supporting_fact_recall,
            "supporting_fact_f1": self.supporting_fact_f1,
        }


@dataclass
class BenchmarkResult:
    """The aggregate + per-question outcome of one benchmark run.

    ``llm_calls`` is the provider-call count made **during this run** — the
    observable ~$0 signal: a warm re-run (graph already built, non-LLM retrieval)
    reports ``0`` (ADR-0009).
    """

    subset: str
    num_questions: int
    num_documents: int
    llm_calls: int
    aggregate: dict[str, float]
    per_question: list[QuestionScore] = field(default_factory=list)

    def format_table(self, *, per_question: bool = False) -> str:
        """Render a clean metrics table for the CLI (aggregate, optionally per-question)."""
        agg = self.aggregate
        lines = [
            "Benchmark results",
            "=================",
            f"subset            : {self.subset}",
            f"questions         : {self.num_questions}",
            f"documents ingested: {self.num_documents}",
            f"llm calls (run)   : {self.llm_calls}",
            "",
            "Supporting-fact retrieval",
            "-------------------------",
            f"  precision : {agg.get('supporting_fact_precision', 0.0):.4f}",
            f"  recall    : {agg.get('supporting_fact_recall', 0.0):.4f}",
            f"  f1        : {agg.get('supporting_fact_f1', 0.0):.4f}",
            "",
            "Answer",
            "------",
            f"  exact match : {agg.get('exact_match', 0.0):.4f}",
            f"  token f1    : {agg.get('token_f1', 0.0):.4f}",
        ]
        if per_question:
            lines += ["", "Per-question", "------------"]
            for score in self.per_question:
                lines.append(
                    f"  [{score.id}] EM={score.exact_match:.0f} "
                    f"F1={score.token_f1:.2f} "
                    f"SF-F1={score.supporting_fact_f1:.2f} "
                    f"pred={score.predicted_answer!r} gold={score.gold_answers!r}"
                )
        return "\n".join(lines)


def _object_key(title: str) -> str:
    """Deterministic, filesystem-safe object key for a paragraph title."""
    slug = re.sub(r"[^\w.-]+", "_", title).strip("_") or "untitled"
    return f"{slug}.txt"


class BenchmarkHarness:
    """Ingest the corpus once, run the subset through retrieval, score non-LLM (N18)."""

    def __init__(self, components: BenchmarkComponents) -> None:
        """Wire the harness to its (offline or real-stack) pipeline components."""
        self._components = components
        self._doc_title: dict[str, str] = {}  # document_id -> paragraph title
        self._ingested = False

    def ingest_corpus(self, examples: list[BenchmarkExample]) -> int:
        """Ingest every example's context paragraphs ONCE, in a fixed order.

        Each unique title is written to the object store and processed through the
        orchestrator exactly once (idempotent by deterministic id); the graph is
        built the first time and reused thereafter. Returns the number of unique
        documents ingested.
        """
        if self._ingested:
            return len(self._doc_title)

        bucket = self._components.bucket
        seen_keys: set[str] = set()
        for example in examples:
            for paragraph in example.context:
                key = _object_key(paragraph.title)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                text = " ".join(paragraph.sentences)
                self._components.object_store.put(bucket, key, text.encode("utf-8"))
                self._components.orchestrator.process_document(
                    IngestTrigger(bucket=bucket, object_key=key)
                )
                self._doc_title[document_id(bucket, key)] = paragraph.title
        self._ingested = True
        _logger.info("benchmark corpus ingested: %d document(s)", len(self._doc_title))
        return len(self._doc_title)

    def run(self, examples: list[BenchmarkExample], *, subset: str = "small") -> BenchmarkResult:
        """Ingest (once), query every example, score, and aggregate.

        Args:
            examples: The selected benchmark examples (a fixed subset).
            subset: The subset label recorded on the result (for reporting).

        Returns:
            A :class:`BenchmarkResult` with per-question + aggregate metrics and the
            LLM-call count made during THIS run.
        """
        calls_before = self._components.llm_calls()
        num_documents = self.ingest_corpus(examples)

        scores: list[QuestionScore] = []
        for example in examples:
            scores.append(self._score_example(example))

        calls_after = self._components.llm_calls()
        return BenchmarkResult(
            subset=subset,
            num_questions=len(scores),
            num_documents=num_documents,
            llm_calls=calls_after - calls_before,
            aggregate=aggregate([s.as_metrics() for s in scores]),
            per_question=scores,
        )

    def _score_example(self, example: BenchmarkExample) -> QuestionScore:
        """Query one example and score its answer + supporting facts (non-LLM)."""
        response = self._components.retriever.retrieve(QueryRequest(question=example.question))

        # Answer: prefer the ranked entity's surfaces (name + aliases) so a
        # differently-phrased-but-correct entity is credited (ADR-0009).
        if response.answer_entity is not None:
            predicted_forms = [response.answer_entity.name]
        elif response.answer is not None:
            predicted_forms = [response.answer]
        else:
            predicted_forms = []
        gold = example.gold_answers
        em = max((exact_match(p, gold) for p in predicted_forms), default=0.0)
        f1 = max((token_f1(p, gold) for p in predicted_forms), default=0.0)

        # Supporting facts: map each retrieved sentence to (title, sentence_index)
        # via the corpus's document_id -> title mapping, and compare to the gold set.
        predicted_sf: set[tuple[str, int]] = set()
        for sentence in response.supporting_sentences:
            title = self._doc_title.get(sentence.document_id)
            if title is not None:
                predicted_sf.add((title, sentence.sentence_index))
        gold_sf = set(example.supporting_facts)
        sf_precision, sf_recall, sf_f1 = supporting_fact_prf(predicted_sf, gold_sf)

        return QuestionScore(
            id=example.id,
            question=example.question,
            predicted_answer=response.answer,
            gold_answers=gold,
            exact_match=em,
            token_f1=f1,
            supporting_fact_precision=sf_precision,
            supporting_fact_recall=sf_recall,
            supporting_fact_f1=sf_f1,
        )
