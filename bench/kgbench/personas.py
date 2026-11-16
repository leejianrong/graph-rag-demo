"""Does a retriever cope with who is asking? Newcomer, learner and expert questions (GRA-187).

Each persona asks a different kind of question over the same world: a newcomer asks discovery
questions about one thing, a learner asks a chain one step at a time with each turn referring back
("name the organization that *it* is affiliated with"), an expert asks what two entities share and
how they are linked. Scored as everywhere in the bench: a question is solved at k when every gold
fact is expressed by one of the top k sentences. A conversation is scored three ways: its last turn
alone (the reference to the earlier answer is unresolved) and with the earlier turns written out
before it, each against the one fact that answers that turn, which shows what resolving the
reference is worth; and the whole chain asked with the earlier turns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from kggen.questions import Question, QuestionSpec, generate_questions, with_persona
from kggen.render import Document, render_documents
from kggen.spec import WorldSpec
from kggen.world import World, generate

from graph_rag.ports import Embedder
from graph_rag.wiring import BenchmarkComponents, build_offline_components
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import Retriever, evaluate

SYSTEMS = ("BM25", "Dense", "graph-rag passages", "graph-rag graph")
PERSONAS = ("newcomer", "learner", "expert")


@dataclass(frozen=True)
class PersonaRun:
    world: World
    k: int
    groups: dict[str, tuple[Question, ...]]  # row name -> the questions in it
    solved: dict[str, dict[str, tuple[int, int]]]  # system -> row -> (solved, asked)


def groups(
    world: World, n_questions: int, phrasing: str = "varied"
) -> dict[str, tuple[Question, ...]]:
    """The rows of the table: each persona's questions, a conversation both ways, by key type."""
    rows: dict[str, tuple[Question, ...]] = {}
    for persona in PERSONAS:
        spec = with_persona(QuestionSpec(n_questions=n_questions, phrasing=phrasing), persona)  # type: ignore[arg-type]
        found = generate_questions(world, spec)
        rows[persona] = found
    talk = tuple(q for q in rows["learner"] if q.type == "conversation")

    def own_fact(q: Question, text: str) -> Question:
        """The last turn, scored on the one fact that answers it (the earlier turns' evidence was
        found when they were asked), asked as ``text``."""
        last = (q.turns[-1]["supporting_facts"][0],)
        return replace(q, question=text, supporting_facts=last, gold_paths=(last,), hops=1)

    rows["learner: last turn alone, its own fact"] = tuple(own_fact(q, q.question) for q in talk)
    rows["learner: last turn with the earlier turns, its own fact"] = tuple(
        own_fact(q, q.transcript()) for q in talk
    )
    rows["learner: whole chain, asked with the earlier turns"] = tuple(
        replace(q, question=q.transcript()) for q in talk
    )
    del rows["learner"]
    rows["expert: what two entities share"] = tuple(q for q in rows["expert"] if q.type == "common")
    return rows


def run(
    spec: WorldSpec,
    *,
    n_questions: int,
    embedder: Embedder,
    k: int = 5,
    pipeline: Callable[[], BenchmarkComponents] | None = None,
) -> PersonaRun:
    world = generate(spec)
    documents: tuple[Document, ...] = render_documents(world)
    rows = groups(world, n_questions)
    make = pipeline or (lambda: build_offline_components(embedder=embedder))
    builders: dict[str, Callable[[], Retriever]] = {
        "BM25": lambda: BM25Retriever(documents),
        "Dense": lambda: DenseRetriever(documents, embedder),
        "graph-rag passages": lambda: GraphRagRetriever(make(), documents, mode="passages"),
        "graph-rag graph": lambda: GraphRagRetriever(make(), documents, mode="graph"),
    }
    solved: dict[str, dict[str, tuple[int, int]]] = {}
    for name, build in builders.items():
        retriever = build()
        solved[name] = {}
        for row, questions in rows.items():
            if not questions:
                continue
            report = evaluate(retriever, list(questions), (k,))
            n = report.overall.questions
            solved[name][row] = (round(report.overall.path_at[k] * n), n)
    return PersonaRun(world, k, rows, solved)


def format_table(result: PersonaRun) -> str:
    """Whole-path@k per system and row, with 95% Wilson intervals."""
    from kgbench.charts import wilson

    rows = [r for r, qs in result.groups.items() if qs]
    lines = [
        "| asker | n | " + " | ".join(result.solved) + " |",
        "|---|---|" + "---|" * len(result.solved),
    ]
    for row in rows:
        cells = []
        for system in result.solved:
            hit, n = result.solved[system].get(row, (0, 0))
            low, high = wilson(hit, n)
            cells.append(f"{hit / n:.0%} ({hit}/{n}, {low:.0%}–{high:.0%})" if n else "-")
        lines.append(f"| {row} | {len(result.groups[row])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
