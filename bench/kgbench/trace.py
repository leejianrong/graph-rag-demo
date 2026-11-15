"""Where does graph-rag lose a multi-hop path? One pass through the stages, per question (GRA-170).

For every question the gold path is a few facts, each stated in a sentence. The trace asks, in the
order the pipeline runs, whether the path survived each stage, and names the first stage where it
did not:

1. **extraction / graph build**: every fact's sentence produced an edge;
2. **linking**: those edges join the right entities (a node whose name or alias is one of the gold
   entity's; lenient, so a merge of two homonyms is not caught here);
3. **seeding**: the question's start entity is among the entity seeds;
4. **traversal**: the correct edges are in the retrieved subgraph;
5. **ranking**: the sentences of the path are in the top ``k`` of the graph view.

BM25 and dense are scored on the same questions, so a loss can be set beside what plain search does.
It reads the built graph from the in-memory store, so it runs on offline components only.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from kggen.render import Document
from kggen.spec import load_spec
from kggen.world import generate

from graph_rag.ids import document_id
from graph_rag.models import QueryRequest
from graph_rag.ports import Embedder
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import BenchmarkComponents
from kgbench.baselines import BM25Retriever, DenseRetriever
from kgbench.graph_retriever import GraphRagRetriever
from kgbench.retrieval import facts_in_top

__all__ = ["STAGES", "format_markdown", "trace"]

STAGES = (
    "solved",
    "extraction",
    "linking",
    "seeding",
    "traversal",
    "ranking",
)
_STAGE_TEXT = {
    "solved": "solved (whole path in the top k)",
    "extraction": "1 extraction or graph build: a path fact has no edge from its sentence",
    "linking": "2 linking: a path fact's edge joins the wrong entities",
    "seeding": "3 seeding: the start entity is not among the seeds",
    "traversal": "4 traversal: the correct edges are not in the subgraph",
    "ranking": "5 ranking: the path is in the subgraph but not in the top k",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def trace(
    folder: Path,
    documents: Sequence[Document],
    questions: Sequence[dict],
    build: Callable[[], BenchmarkComponents],
    embedder: Embedder,
    *,
    k: int = 5,
) -> dict:
    """Run the stages over ``questions`` and count, per hop count, where paths are lost.

    Args:
        folder: The dataset folder (its ``spec.yaml`` gives the world and so the gold entities).
        documents: The documents to ingest, in order.
        questions: The question records (``supporting_facts``, ``hops``, ``start``, ``question``).
        build: Builds the offline components to ingest into (a fresh pipeline).
        embedder: The embedder the dense baseline uses.
        k: The cut-off for the top list.

    Returns:
        ``{"graph": {nodes, edges}, "hops": {hop: counts}, "first_loss": {hop: {stage: n}}}``.
    """
    world = generate(load_spec(folder / "spec.yaml"))
    names = {e.id: {_norm(e.name), *(_norm(a) for a in e.aliases)} for e in world.entities}
    gold = {e.id: e for e in world.edges}

    components = build()
    graph = GraphRagRetriever(components, documents, mode="graph")
    # The bench only runs on offline components, whose graph store keeps edges in memory.
    store = components.graph_store
    edges = list(store._edges[DEFAULT_SCOPE])  # type: ignore[attr-defined]
    nodes = store._nodes[DEFAULT_SCOPE]  # type: ignore[attr-defined]

    sentences_of: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc in documents:
        doc_hash = document_id(components.bucket, f"{doc.id}.txt")
        for sentence in doc.sentences:
            for fact in sentence.fact_ids:
                sentences_of[fact].append((doc_hash, sentence.text.strip()))
    by_sentence: dict[tuple[str, str], list] = defaultdict(list)
    for edge in edges:
        by_sentence[
            (edge.provenance.source_doc_id, edge.provenance.source_sentence.strip())
        ].append(edge)

    def is_entity(canonical_id: str, gold_id: str) -> bool:
        node = nodes.get(canonical_id)
        if node is None:
            return False
        return bool({_norm(node.name), *(_norm(a) for a in node.aliases)} & names[gold_id])

    def edges_of(fact: str) -> tuple[list, list]:
        """The edges built from the fact's sentence, and those joining its two gold entities."""
        want = gold[fact]
        built = [e for key in sentences_of[fact] for e in by_sentence.get(key, [])]
        right = [
            e
            for e in built
            if (is_entity(e.subject_id, want.subject) and is_entity(e.object_id, want.object))
            or (is_entity(e.subject_id, want.object) and is_entity(e.object_id, want.subject))
        ]
        return built, right

    def edge_key(edge) -> tuple:
        p = edge.provenance
        return (edge.subject_id, edge.predicate, edge.object_id, p.source_doc_id, p.source_sentence)

    dense = DenseRetriever(documents, embedder)
    bm25 = BM25Retriever(documents)

    hops: dict[int, Counter] = defaultdict(Counter)
    first_loss: dict[int, Counter] = defaultdict(Counter)
    for q in questions:
        facts = q["supporting_facts"]
        c = hops[q["hops"]]
        c["questions"] += 1
        per_fact = {f: edges_of(f) for f in facts}
        built = all(per_fact[f][0] for f in facts)
        right = all(per_fact[f][1] for f in facts)
        response = components.retriever.retrieve(QueryRequest(question=q["question"]))
        seeded = any(is_entity(s, q["start"]) for s in response.matched_entity_ids)
        in_subgraph = {edge_key(e) for e in response.subgraph.edges}
        reached = right and all(
            any(edge_key(e) in in_subgraph for e in per_fact[f][1]) for f in facts
        )
        need = set(facts)
        top = facts_in_top(graph.retrieve(q["question"], k), k)
        solved = need <= top
        for name, hit in (
            ("extracted", built),
            ("edges_right", right),
            ("start_seeded", seeded),
            ("path_in_subgraph", reached),
            ("graph_solved", solved),
            ("dense_solved", need <= facts_in_top(dense.retrieve(q["question"], k), k)),
            ("bm25_solved", need <= facts_in_top(bm25.retrieve(q["question"], k), k)),
        ):
            c[name] += int(hit)
        stage = (
            "solved"
            if solved
            else "extraction"
            if not built
            else "linking"
            if not right
            else "seeding"
            if not seeded
            else "traversal"
            if not reached
            else "ranking"
        )
        first_loss[q["hops"]][stage] += 1
    return {
        "graph": {"nodes": len(nodes), "edges": len(edges)},
        "hops": {h: dict(c) for h, c in sorted(hops.items())},
        "first_loss": {h: dict(c) for h, c in sorted(first_loss.items())},
    }


_COLUMNS = (
    ("questions", "n"),
    ("extracted", "every fact has an edge"),
    ("edges_right", "edges join the right entities"),
    ("start_seeded", "start entity seeded"),
    ("path_in_subgraph", "path in subgraph"),
    ("graph_solved", "graph top k"),
    ("dense_solved", "dense top k"),
    ("bm25_solved", "BM25 top k"),
)


def format_markdown(result: dict, *, command: str, k: int) -> str:
    """The funnel per hop count, then the first stage that lost each unsolved question."""
    hops = result["hops"]
    total: Counter = Counter()
    for counts in hops.values():
        total.update(counts)
    graph = result["graph"]
    lines = [
        "# Where multi-hop paths are lost",
        "",
        f"`{command}`",
        "",
        f"The built graph has {graph['nodes']} nodes and {graph['edges']} edges. "
        f"Counts are questions; a question needs every fact of its path. The graph view is "
        f"scored at top {k}.",
        "",
        "| hops | " + " | ".join(label for _, label in _COLUMNS) + " |",
        "|---|" + "---|" * len(_COLUMNS),
    ]
    for hop, counts in hops.items():
        lines.append(
            f"| {hop} | " + " | ".join(str(counts.get(key, 0)) for key, _ in _COLUMNS) + " |"
        )
    lines.append("| all | " + " | ".join(str(total.get(key, 0)) for key, _ in _COLUMNS) + " |")
    lines += [
        "",
        "## First stage where a path is lost",
        "",
        "| stage | " + " | ".join(str(h) for h in hops) + " | all |",
        "|---|" + "---|" * (len(hops) + 1),
    ]
    for stage in STAGES:
        row = [result["first_loss"].get(h, {}).get(stage, 0) for h in hops]
        lines.append(f"| {_STAGE_TEXT[stage]} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    return "\n".join(lines) + "\n"
