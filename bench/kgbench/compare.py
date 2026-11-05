"""Compare retrieval systems on the same generated questions, over several world seeds.

Each seed makes a different world, its documents and questions. BM25, a dense retriever and
graph-rag (its passages alone, and its graph evidence) are scored with ``kgbench.retrieval`` on
every one. The report gives the mean per hop count with the spread across seeds, so one lucky
world cannot stand for the result. One embedder instance, with a cache, serves both the dense
baseline and graph-rag, so they use the same vectors and each sentence is embedded once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kggen.questions import QuestionSpec, generate_questions
from kggen.render import Document, render_documents
from kggen.spec import WorldSpec
from kggen.world import generate

from graph_rag.ports import Embedder
from graph_rag.wiring import BenchmarkComponents, build_embedder, build_offline_components
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.cost import Cost, Meter
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import DEFAULT_KS, RetrievalReport, Retriever, evaluate

SYSTEMS = ("BM25", "Dense", "graph-rag passages", "graph-rag graph")

CAVEATS = (
    "The text is rendered from templates: one fact per sentence, made-up names, no paraphrase. "
    "That is easier than real prose in some ways (names are unambiguous tokens) and unlike it in "
    "others, so public sets stay in the picture.",
    "graph-rag here is the offline pipeline: heuristic entity detection, merge-by-name linking and "
    "a co-occurrence graph whose only relation is RELATED_TO. It is a floor, not the real stack "
    "(spaCy, an LLM-built graph), and it was not tuned.",
    "graph-rag passages are a nearest-neighbour search over sentence vectors, so they behave like "
    "the dense baseline by construction; graph-rag graph adds the sentences behind the retrieved "
    "subgraph's edges.",
    "A question is solved at k only if every fact on its path is expressed by one of the top k "
    "sentences. Questions come from the generator, never from graph-rag's own extraction.",
)


class CachingEmbedder:
    """An ``Embedder`` that remembers vectors by text, so a repeated sentence costs nothing."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.embedded = 0  # texts that actually went to the inner embedder

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self.embedded += len(missing)
            self._cache.update(zip(missing, self._inner.embed(missing), strict=True))
        return [self._cache[t] for t in texts]


@dataclass(frozen=True)
class SeedRun:
    seed: int
    entities: int
    facts: int
    documents: int
    sentences: int
    questions: int
    reports: dict[str, RetrievalReport]
    costs: dict[str, Cost]  # per system: wall time (index build plus questions) and LLM usage


@dataclass(frozen=True)
class Comparison:
    preset: str
    embedder: str
    ks: tuple[int, ...]
    runs: tuple[SeedRun, ...]

    def _values(self, system: str, hops: int | None, k: int, metric: str) -> list[float]:
        out = []
        for run in self.runs:
            report = run.reports[system]
            score = report.overall if hops is None else report.by_hops.get(hops)
            if score is not None:
                out.append((score.path_at if metric == "path" else score.recall_at)[k])
        return out

    def hop_counts(self) -> list[int]:
        return sorted({h for r in self.runs for rep in r.reports.values() for h in rep.by_hops})

    def mean(self, system: str, hops: int | None, k: int, metric: str) -> float:
        values = self._values(system, hops, k, metric)
        return sum(values) / len(values) if values else 0.0

    def spread(self, system: str, hops: int | None, k: int, metric: str) -> tuple[float, float]:
        values = self._values(system, hops, k, metric)
        return (min(values), max(values)) if values else (0.0, 0.0)

    def questions_at(self, hops: int | None) -> int:
        """Questions behind a table row, summed over seeds (``None`` is every hop count)."""
        total = 0
        for run in self.runs:
            report = run.reports[SYSTEMS[0]]
            score = report.overall if hops is None else report.by_hops.get(hops)
            total += score.questions if score else 0
        return total

    def _cell(self, system: str, hops: int | None, k: int) -> str:
        path, recall = self.mean(system, hops, k, "path"), self.mean(system, hops, k, "recall")
        return f"{path:.2f} / {recall:.2f}"

    def format_table(self, k: int) -> str:
        """Per hop count: ``path@k / recall@k`` (means over seeds) for every system."""
        rows = [("all", None)] + [(str(h), h) for h in self.hop_counts()]
        width = max(len(s) for s in SYSTEMS)
        lines = [
            f"path@{k} / recall@{k}, mean over seeds {[r.seed for r in self.runs]}",
            "  hops  n     " + "  ".join(f"{s:<{width}}" for s in SYSTEMS),
        ]
        for name, h in rows:
            cells = "  ".join(self._cell(s, h, k).ljust(width) for s in SYSTEMS)
            lines.append(f"  {name:<5} {self.questions_at(h):<5} {cells}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "embedder": self.embedder,
            "ks": list(self.ks),
            "seeds": [
                {
                    "seed": r.seed,
                    "entities": r.entities,
                    "facts": r.facts,
                    "documents": r.documents,
                    "sentences": r.sentences,
                    "questions": r.questions,
                    "seconds": {s: round(c.wall_seconds, 2) for s, c in r.costs.items()},
                    "cost": {s: c.to_dict() for s, c in r.costs.items()},
                    "systems": {s: rep.to_dict() for s, rep in r.reports.items()},
                }
                for r in self.runs
            ],
        }

    def cost_rows(self) -> list[str]:
        """One markdown row per system: mean wall time per seed, and LLM usage summed over seeds."""
        rows = []
        for system in SYSTEMS:
            costs = [r.costs[system] for r in self.runs]
            seconds = sum(c.wall_seconds for c in costs) / len(costs)
            calls = sum(c.llm.calls for c in costs)
            usd = sum(c.llm.cost_usd for c in costs)
            rows.append(f"| {system} | {seconds:.1f} | {calls} | ${usd:.2f} |")
        return rows

    def to_markdown(self, *, k: int, command: str) -> str:
        first = self.runs[0]
        seeds = [r.seed for r in self.runs]
        rows = [("all", None)] + [(str(h), h) for h in self.hop_counts()]

        def table(metric_cells) -> list[str]:
            out = ["| hops | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
            for name, h in rows:
                out.append(f"| {name} | " + " | ".join(metric_cells(s, h) for s in SYSTEMS) + " |")
            return out

        def mean_cell(s: str, h: int | None) -> str:
            return self._cell(s, h, k)

        def range_cell(s: str, h: int | None) -> str:
            lo, hi = self.spread(s, h, k, "path")
            return f"{lo:.2f} to {hi:.2f}"

        lines = [
            f"# Retrieval comparison: {self.preset} worlds",
            "",
            f"Can a graph beat keyword and dense retrieval on multi-hop questions? Every system "
            f"answers the same generated questions; a question counts as solved at k when every "
            f"fact on its path is expressed by one of the top {k} sentences.",
            "",
            "## Setup",
            "",
            f"- Preset `{self.preset}`, world seeds {seeds}, one world per seed "
            f"({first.entities} entities, {first.facts} facts, {first.documents} documents, "
            f"{first.sentences} sentences in the first).",
            f"- {first.questions} questions per seed, equal weight on each hop count; "
            f"{sum(r.questions for r in self.runs)} in total.",
            f"- Embedder for the dense baseline and graph-rag: {self.embedder}.",
            f"- Reproduce: `{command}`",
            "",
            f"## path@{k} / recall@{k} (mean over seeds)",
            "",
            *table(mean_cell),
            "",
            f"## path@{k}: lowest to highest seed",
            "",
            *table(range_cell),
            "",
            "## Cost and latency",
            "",
            "Wall time covers building the index and answering every question, on CPU with no GPU. "
            "The systems share one embedding cache, so the times are not a like-for-like speed "
            "comparison. LLM calls and dollars are totals over the seeds; none of these systems "
            "reaches a provider.",
            "",
            "| system | seconds per seed (mean) | LLM calls | LLM dollars |",
            "|---|---|---|---|",
            *self.cost_rows(),
            "",
            "## Caveats",
            "",
            *[f"- {c}" for c in CAVEATS],
            "",
        ]
        return "\n".join(lines)


def run_seed(
    spec: WorldSpec,
    *,
    n_questions: int,
    embedder: Embedder,
    ks: Sequence[int],
    documents: tuple[Document, ...] | None = None,
) -> SeedRun:
    """Score every system on one world; ``documents`` replaces the template text for it."""
    world = generate(spec)
    documents = documents or render_documents(world)
    questions = generate_questions(world, QuestionSpec(n_questions=n_questions))
    # Each system is (the pipeline it runs, which has an LLM bill, or None; how to build it).
    graph = {m: build_offline_components(embedder=embedder) for m in ("passages", "graph")}
    systems: dict[str, tuple[BenchmarkComponents | None, Callable[[], Retriever]]] = {
        "BM25": (None, lambda: BM25Retriever(documents)),
        "Dense": (None, lambda: DenseRetriever(documents, embedder)),
        "graph-rag passages": (
            graph["passages"],
            lambda: GraphRagRetriever(graph["passages"], documents, mode="passages"),
        ),
        "graph-rag graph": (
            graph["graph"],
            lambda: GraphRagRetriever(graph["graph"], documents, mode="graph"),
        ),
    }
    reports: dict[str, RetrievalReport] = {}
    costs: dict[str, Cost] = {}
    for name, (components, build) in systems.items():
        with Meter(components) as meter:
            reports[name] = evaluate(build(), questions, ks)
        costs[name] = meter.cost
    return SeedRun(
        spec.seed,
        len(world.entities),
        len(world.edges),
        len(documents),
        sum(len(d.sentences) for d in documents),
        len(questions),
        reports,
        costs,
    )


def compare(
    base: WorldSpec,
    seeds: Sequence[int],
    *,
    preset: str,
    n_questions: int = 200,
    real_embedder: bool = False,
    ks: Sequence[int] = DEFAULT_KS,
) -> Comparison:
    embedder = CachingEmbedder(build_embedder(real=real_embedder))
    runs = tuple(
        run_seed(
            WorldSpec.model_validate({**base.model_dump(), "seed": seed}),
            n_questions=n_questions,
            embedder=embedder,
            ks=ks,
        )
        for seed in seeds
    )
    name = (
        "bge-small-en-v1.5 (local sentence-transformers)"
        if real_embedder
        else "deterministic hashing fake"
    )
    return Comparison(preset, name, tuple(sorted(set(ks))), runs)
