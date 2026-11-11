"""LongMemEval adapter: chat-history memory questions scored per question type (GRA-16).

LongMemEval gives each question its own "haystack" of dated chat sessions and asks about a fact in
one or more of them. This loads that format, ingests a question's sessions into a fresh pipeline in
date order (one document per session), asks the question, and scores what comes back without an
LLM, against the gold the dataset ships (``answer_session_ids`` and ``answer``); nothing is gold
because graph-rag extracted it.

Two non-LLM numbers per question, over the top ``k`` retrieved sentences:

* **session recall@k**: the share of the question's evidence sessions that supply a top-``k``
  sentence (LongMemEval's own retrieval metric, at session level);
* **answer@k**: whether the normalised gold answer appears in those sentences. It is a lenient
  proxy for answer accuracy, and only meaningful when the answer is a phrase the history states
  (not for "which came first" or counting questions). The LLM-judged QA score is out of scope.

Questions whose id ends ``_abs`` are abstention questions, which have no evidence session; they
are counted and left out of both scores. Results are grouped by ``question_type``.

Every system gets the same sessions: BM25 over the turns, and offline graph-rag (the ``graph``
view of ``GraphRagRetriever``). Each question builds fresh components, so the haystacks never mix.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kggen.render import Document, Sentence

from graph_rag.wiring import build_offline_components
from kgbench.baselines import BM25Retriever
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.metrics import normalize_answer
from kgbench.retrieval import RetrievedSentence, Retriever

DEFAULT_KS = (1, 3, 5)
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "longmemeval_mini.json"
SYSTEMS = ("bm25", "graph-rag")


@dataclass(frozen=True)
class Session:
    id: str
    date: str
    turns: tuple[tuple[str, str], ...]  # (role, content)


@dataclass(frozen=True)
class MemQuestion:
    id: str
    type: str
    question: str
    answer: str
    evidence: tuple[str, ...]  # answer_session_ids
    sessions: tuple[Session, ...]  # in ingest order (oldest first)

    @property
    def abstention(self) -> bool:
        return self.id.endswith("_abs")


def load_longmemeval(path: Path | str) -> list[MemQuestion]:
    """Read a LongMemEval JSON file (a list of question objects) into ``MemQuestion``s.

    Sessions are ordered by their ``haystack_dates`` (stable, so equal or missing dates keep file
    order); LongMemEval dates sort correctly as text (``2023/05/20 (Sat) 02:21``).
    """
    raw = json.loads(Path(path).read_text())
    questions = []
    for item in raw:
        ids = item["haystack_session_ids"]
        dates = item.get("haystack_dates") or [""] * len(ids)
        sessions = [
            Session(sid, str(date), tuple((t["role"], t["content"]) for t in turns))
            for sid, date, turns in zip(ids, dates, item["haystack_sessions"], strict=True)
        ]
        sessions.sort(key=lambda s: s.date)
        questions.append(
            MemQuestion(
                id=str(item["question_id"]),
                type=item["question_type"],
                question=item["question"],
                answer=str(item["answer"]),
                evidence=tuple(item.get("answer_session_ids") or ()),
                sessions=tuple(sessions),
            )
        )
    return questions


def session_document(session: Session) -> Document:
    """A session as one document: a date line, then one ``role: content`` line per turn."""
    lines = [f"Session date: {session.date}"] if session.date else []
    lines += [f"{role}: {content}" for role, content in session.turns]
    sentences, offset = [], 0
    for line in lines:
        sentences.append(Sentence(line, offset, offset + len(line), (), ()))
        offset += len(line) + 1
    return Document(session.id, session.id, "\n".join(lines), tuple(sentences), False)


@dataclass(frozen=True)
class QuestionScore:
    id: str
    type: str
    session_recall: dict[int, float]
    answer_hit: dict[int, float]


def score_question(
    question: MemQuestion, ranking: Sequence[RetrievedSentence], ks: Sequence[int]
) -> QuestionScore:
    """Score one ranked list of sentences (each carries its session id in ``doc``)."""
    gold = set(question.evidence)
    answer = normalize_answer(question.answer)
    recall, hit = {}, {}
    for k in ks:
        top = ranking[:k]
        found = {s.doc for s in top} & gold
        recall[k] = len(found) / len(gold) if gold else 0.0
        text = normalize_answer(" ".join(s.text for s in top))
        hit[k] = float(bool(answer) and answer in text)
    return QuestionScore(question.id, question.type, recall, hit)


@dataclass(frozen=True)
class TypeScore:
    questions: int
    session_recall: dict[int, float]
    answer_hit: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "questions": self.questions,
            "session_recall": {k: round(v, 4) for k, v in self.session_recall.items()},
            "answer_hit": {k: round(v, 4) for k, v in self.answer_hit.items()},
        }


@dataclass(frozen=True)
class SystemReport:
    ks: tuple[int, ...]
    overall: TypeScore
    by_type: dict[str, TypeScore]

    def to_dict(self) -> dict:
        return {
            "ks": list(self.ks),
            "overall": self.overall.to_dict(),
            "by_type": {t: s.to_dict() for t, s in sorted(self.by_type.items())},
        }


def summarize(scores: Sequence[QuestionScore], ks: Sequence[int]) -> TypeScore:
    n = len(scores)
    return TypeScore(
        n,
        {k: sum(s.session_recall[k] for s in scores) / n if n else 0.0 for k in ks},
        {k: sum(s.answer_hit[k] for s in scores) / n if n else 0.0 for k in ks},
    )


def report_of(scores: Sequence[QuestionScore], ks: Sequence[int]) -> SystemReport:
    groups: dict[str, list[QuestionScore]] = defaultdict(list)
    for s in scores:
        groups[s.type].append(s)
    return SystemReport(
        tuple(ks), summarize(scores, ks), {t: summarize(g, ks) for t, g in groups.items()}
    )


@dataclass
class LongMemEvalResult:
    ks: tuple[int, ...]
    questions: int
    abstention: int  # counted, not scored
    systems: dict[str, SystemReport] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ks": list(self.ks),
            "questions": self.questions,
            "abstention_skipped": self.abstention,
            "systems": {name: r.to_dict() for name, r in self.systems.items()},
        }

    def format_table(self) -> str:
        title = (
            f"LongMemEval: {self.questions} questions scored ({self.abstention} abstention skipped)"
        )
        lines = [title, "-" * len(title)]
        for name, report in self.systems.items():
            head = (
                "  type"
                + " " * 22
                + "n   "
                + "  ".join(f"sessR@{k:<2} ans@{k:<2}" for k in self.ks)
            )
            lines += ["", name, head]
            rows = [("all", report.overall)] + sorted(report.by_type.items())
            for label, s in rows:
                cells = "  ".join(
                    f"{s.session_recall[k]:.2f}    {s.answer_hit[k]:.2f}  " for k in self.ks
                )
                lines.append(f"  {label:<26}{s.questions:<4}{cells}")
        return "\n".join(lines)


RetrieverFactory = Callable[[Sequence[Document]], Retriever]


def _bm25(documents: Sequence[Document]) -> Retriever:
    return BM25Retriever(documents)


def _graph_rag(documents: Sequence[Document]) -> Retriever:
    return GraphRagRetriever(build_offline_components(), documents, mode="graph")


FACTORIES: Mapping[str, RetrieverFactory] = {"bm25": _bm25, "graph-rag": _graph_rag}


def run_longmemeval(
    questions: Sequence[MemQuestion],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    systems: Sequence[str] = SYSTEMS,
    factories: Mapping[str, RetrieverFactory] = FACTORIES,
) -> LongMemEvalResult:
    """Ingest each question's sessions in date order into every system, ask, and score by type."""
    ks = tuple(sorted(set(ks)))
    scored = [q for q in questions if not q.abstention]
    per_system: dict[str, list[QuestionScore]] = {name: [] for name in systems}
    for q in scored:
        documents = [session_document(s) for s in q.sessions]
        for name in systems:
            retriever = factories[name](documents)
            ranking = list(retriever.retrieve(q.question, max(ks)))
            per_system[name].append(score_question(q, ranking, ks))
    return LongMemEvalResult(
        ks,
        len(scored),
        len(questions) - len(scored),
        {name: report_of(per_system[name], ks) for name in systems},
    )
