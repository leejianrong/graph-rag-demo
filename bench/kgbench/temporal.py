"""Do the systems answer questions about when? A world that changes, in dated documents (GRA-190).

The world gains and loses entities, facts are replaced and ended, and organisations are renamed.
Its documents are written in date order, each dated, as a memory store would receive them. The
questions ask what held in a year (``as_of``), what holds now (``latest``), when a fact began or
ended (``when``), what came just before (``before``) and how long a fact held (``duration``). A
question is solved at k when every gold fact is expressed by one of the top k sentences, as
everywhere in the bench, so the scores are comparable with the other tables; here they are split
by question type. A system that can use a date is given the question's year; BM25 and dense search
cannot, so they must find the right version from the words alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kggen.questions import Question, QuestionSpec, generate_questions
from kggen.render import Document, RenderSpec, render_documents
from kggen.spec import WorldSpec, with_evolution
from kggen.world import World, generate

from graph_rag.ports import Embedder
from graph_rag.wiring import BenchmarkComponents, build_offline_components
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import RetrievedSentence, Retriever, evaluate

TYPES = ("as_of", "latest", "when", "before", "duration")
SYSTEMS = ("BM25", "Dense", "graph-rag passages", "graph-rag graph")


@dataclass(frozen=True)
class TemporalRun:
    world: World
    documents: tuple[Document, ...]
    questions: tuple[Question, ...]
    k: int
    solved: dict[str, dict[str, tuple[int, int]]]  # system -> type -> (solved, asked)
    # system -> type -> (right version first, asked): of the retrieved sentences about the same
    # subject and relation, the first states the gold fact and not another version of it.
    first: dict[str, dict[str, tuple[int, int]]]


def first_is_right(
    question: Question, ranking: Sequence[RetrievedSentence], versions: dict[str, set[str]]
) -> bool:
    """Whether the first sentence about the question's fact states the right version of it.

    A reader who takes the top evidence about "who employs X" gets the wrong answer when an older
    employer's sentence ranks above the current one, even if the right sentence is also in the
    list, so retrieving the gold sentence somewhere in the top k is not enough for a temporal
    question. ``versions`` maps a fact id to every fact with the same subject and relation."""
    gold = set(question.supporting_facts)
    rivals = {v for f in gold for v in versions.get(f, ())} | gold
    for sentence in ranking:
        hit = rivals & set(sentence.fact_ids)
        if hit:
            return bool(hit & gold)
    return False


class RecencyRetriever:
    """A plain retriever that also reads dates: the baseline a memory store with timestamps gets for
    free. It fetches a wide pool, drops documents dated after the question's year, and puts the
    newest documents first, so the latest version of a fact outranks the one it replaced. It cannot
    tell which sentence is about the fact asked; the base retriever's ranking breaks ties."""

    supports_as_of = True

    def __init__(self, base: Retriever, documents: Sequence[Document], pool: int = 40) -> None:
        self._base = base
        self._year = {d.id: int(d.date[:4]) for d in documents if d.date}
        self._pool = pool

    def retrieve(
        self, question: str, k: int, as_of: int | None = None
    ) -> Sequence[RetrievedSentence]:
        found = self._base.retrieve(question, max(self._pool, k))
        if as_of is not None:
            found = [s for s in found if self._year.get(s.doc, 0) <= as_of]
        return sorted(found, key=lambda s: -self._year.get(s.doc, 0))[:k]


def build(
    spec: WorldSpec, n_questions: int, evolution: str = "heavy"
) -> tuple[World, tuple[Document, ...], tuple[Question, ...]]:
    """An evolving world, its dated documents and a balanced set of temporal questions."""
    world = generate(with_evolution(spec, evolution))
    documents = render_documents(world, RenderSpec(timeline=True))
    questions = generate_questions(
        world, QuestionSpec(n_questions=n_questions, type_mix={t: 1.0 for t in TYPES})
    )
    # A world too small to have every kind fills the count with ordinary questions; they say
    # nothing about time, so they are left out.
    return world, documents, tuple(q for q in questions if q.type in TYPES)


def run(
    spec: WorldSpec,
    *,
    n_questions: int,
    embedder: Embedder,
    k: int = 5,
    evolution: str = "heavy",
    pipeline: Callable[[], BenchmarkComponents] | None = None,
    recency: bool = False,
) -> TemporalRun:
    """Score every system on one evolving world, split by question type."""
    world, documents, questions = build(spec, n_questions, evolution)
    make = pipeline or (lambda: build_offline_components(embedder=embedder))
    builders: dict[str, Callable[[], Retriever]] = {
        "BM25": lambda: BM25Retriever(documents),
        "Dense": lambda: DenseRetriever(documents, embedder),
        "graph-rag passages": lambda: GraphRagRetriever(make(), documents, mode="passages"),
        "graph-rag graph": lambda: GraphRagRetriever(make(), documents, mode="graph"),
    }
    if recency:
        builders["BM25 + recency"] = lambda: RecencyRetriever(BM25Retriever(documents), documents)
        builders["Dense + recency"] = lambda: RecencyRetriever(
            DenseRetriever(documents, embedder), documents
        )
    same: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in world.edges:
        same[(f.subject, f.predicate)].add(f.id)
    versions = {f.id: same[(f.subject, f.predicate)] - {f.id} for f in world.edges}
    solved: dict[str, dict[str, tuple[int, int]]] = {}
    first: dict[str, dict[str, tuple[int, int]]] = {}
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.type].append(q)
    for name, build_system in builders.items():
        retriever = build_system()
        solved[name], first[name] = {}, {}
        for kind, asked in by_type.items():
            report = evaluate(retriever, asked, (k,))
            solved[name][kind] = (
                round(report.overall.path_at[k] * report.overall.questions),
                report.overall.questions,
            )
            right = 0
            for q in asked:
                ranking = (
                    retriever.retrieve(q.question, k, as_of=q.as_of)  # type: ignore[call-arg]
                    if getattr(retriever, "supports_as_of", False)
                    else retriever.retrieve(q.question, k)
                )
                right += first_is_right(q, ranking, versions)
            first[name][kind] = (right, len(asked))
    return TemporalRun(world, documents, questions, k, solved, first)


def format_table(result: TemporalRun, which: str = "solved") -> str:
    """Per system and question type, with 95% Wilson intervals: whole-path@k (``solved``) or the
    share whose first sentence about the fact states the right version (``first``)."""
    from kgbench.charts import wilson

    table = result.solved if which == "solved" else result.first

    kinds = [t for t in TYPES if any(q.type == t for q in result.questions)]
    asked = {t: sum(q.type == t for q in result.questions) for t in kinds}
    lines = [
        "| type | n | " + " | ".join(table) + " |",
        "|---|---|" + "---|" * len(table),
    ]
    for t in [*kinds, "all"]:
        cells = []
        for system in table:
            if t == "all":
                hit = sum(h for h, _ in table[system].values())
                n = sum(n for _, n in table[system].values())
            else:
                hit, n = table[system].get(t, (0, 0))
            low, high = wilson(hit, n)
            cells.append(f"{hit / n:.0%} ({hit}/{n}, {low:.0%}–{high:.0%})" if n else "-")
        lines.append(
            f"| {t} | {sum(asked.values()) if t == 'all' else asked[t]} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def metrics(result: TemporalRun) -> dict[str, float]:
    """Flat ``{metric: value}`` for the quality gate."""
    out: dict[str, float] = {}
    for system, rows in result.solved.items():
        if not system.startswith("graph-rag"):
            continue
        for kind, (hit, n) in rows.items():
            out[f"temporal/{system}/{kind}/path@{result.k}"] = hit / n if n else 0.0
        total = sum(n for _, n in rows.values())
        out[f"temporal/{system}/all/path@{result.k}"] = (
            sum(h for h, _ in rows.values()) / total if total else 0.0
        )
        right = result.first[system]
        out[f"temporal/{system}/all/first-version@{result.k}"] = (
            sum(h for h, _ in right.values()) / total if total else 0.0
        )
    return out
