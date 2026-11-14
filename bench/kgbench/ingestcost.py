"""What ingesting 1,000 documents costs: LLM tokens and dollars, wall time and storage (GRA-55).

Three measurements, kept apart because they need different things:

* **Tokens.** The LLM stages answer from a committed cache, so a replay costs nothing, but every
  request still passes through the client's tap with its prompt and its recorded answer. The
  tokens are counted from those, per stage, so no provider is needed. The prompt is exact; the
  tokenizer is the caller's (OpenAI's for the default model, an approximation for others).
* **Dollars.** Tokens times a price per million tokens that the caller supplies. They are
  computed, not billed.
* **Time and storage.** The real stack (spaCy, bge, Elasticsearch, Neo4j) ingests the corpus with
  the LLM answers replayed from the cache, timing each document, and the stores are measured at
  checkpoints. The time is everything except the LLM's own latency, which a replay does not have.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "Checkpoint",
    "Price",
    "StageTokens",
    "TokenCounter",
    "dollars_per_thousand",
    "per_thousand",
    "slope_per_thousand",
    "summarise_seconds",
    "tap_tokens",
]

TokenCounter = Callable[[str], int]


@dataclass
class StageTokens:
    """Tokens one stage sent and received over a run, and the requests behind them."""

    requests: int = 0
    prompt: int = 0
    completion: int = 0


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens, for one model."""

    input_per_million: float
    output_per_million: float

    def dollars(self, tokens: StageTokens) -> float:
        return (
            tokens.prompt * self.input_per_million + tokens.completion * self.output_per_million
        ) / 1_000_000


@dataclass
class Checkpoint:
    """The stores after ``documents`` have been ingested: sizes in bytes, and record counts."""

    documents: int
    seconds: float
    sizes: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def tap_tokens(count: TokenCounter) -> tuple[Callable[[str, str, str, bool], None], dict]:
    """An ``llm_tap`` for ``build_*_components`` and the per-stage totals it fills in."""
    totals: dict[str, StageTokens] = {}

    def tap(stage: str, prompt: str, text: str, from_cache: bool) -> None:
        entry = totals.setdefault(stage, StageTokens())
        entry.requests += 1
        entry.prompt += count(prompt)
        entry.completion += count(text)

    return tap, totals


def per_thousand(value: float, documents: int) -> float:
    """``value`` over ``documents`` scaled to 1,000 documents."""
    return value * 1000 / documents


def dollars_per_thousand(totals: dict[str, StageTokens], price: Price, documents: int) -> float:
    """What the run's requests cost at ``price``, scaled to 1,000 documents."""
    return per_thousand(sum(price.dollars(t) for t in totals.values()), documents)


def slope_per_thousand(points: Sequence[Checkpoint], store: str) -> float:
    """Bytes a store grows per 1,000 documents: a least-squares line through every checkpoint.

    The slope leaves out what a store costs before it holds anything (segments, indexes), which a
    plain size over the document count would spread over a small corpus and overstate. A store
    that stops growing (shared entities run out) gives a slope near zero or below.
    """
    if len(points) < 2:
        raise ValueError("a slope needs two checkpoints")
    xs = [p.documents for p in points]
    ys = [p.sizes[store] for p in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    return 1000 * sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread


def summarise_seconds(seconds: Sequence[float]) -> dict[str, float]:
    """Mean, median and 95th percentile of per-document wall times, and documents per second."""
    ordered = sorted(seconds)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    mean = statistics.fmean(ordered)
    return {
        "mean": mean,
        "median": statistics.median(ordered),
        "p95": p95,
        "docs_per_second": 1 / mean if mean else 0.0,
    }


# --- running it -----------------------------------------------------------------------------


def count_tokens_o200k() -> TokenCounter:
    """OpenAI's ``o200k_base`` tokenizer (gpt-4o and gpt-4o-mini). Needs tiktoken's vocabulary."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def measure_tokens(documents, settings, count: TokenCounter, *, spacy: bool, embedder=None):
    """Replay the corpus through the LLM stages offline; return per-stage tokens.

    The embedder (the hashing fake unless one is passed), NER and linking must match the run the
    cache was recorded with, so every request is a cache hit and the tap sees exactly the prompts
    a provider would be sent.
    """
    from graph_rag.fakes import FakeEmbedder
    from graph_rag.wiring import build_offline_components
    from kgbench.extraction import ingest

    tap, totals = tap_tokens(count)
    components = build_offline_components(
        settings=settings,
        embedder=embedder or FakeEmbedder(dim=settings.embed_dim),
        llm_stages=True,
        spacy_ner=spacy,
        llm_tap=tap,
    )
    dropped = sum(result is None for _doc, result in ingest(components, documents))
    usage = components.llm_usage()
    if usage.calls or dropped:
        raise RuntimeError(
            f"the replay was not complete: {usage.calls} provider calls, {dropped} dropped"
        )
    return totals


def store_sizes(settings, raw_bytes: int) -> tuple[dict[str, int], dict[str, int]]:
    """What the real stores hold now: Elasticsearch index bytes (merged), the raw text's bytes,
    and the Neo4j node and edge counts (its file size is only exact after a clean stop)."""
    import json
    import urllib.request

    from neo4j import GraphDatabase

    base = settings.elasticsearch_url.rstrip("/")
    sizes: dict[str, int] = {"raw text": raw_bytes}
    for label, index in (
        ("elasticsearch documents", settings.documents_index),
        ("elasticsearch entities", settings.entities_index),
    ):
        for path in (
            f"/{index}/_refresh",
            f"/{index}/_forcemerge?max_num_segments=1",
            f"/{index}/_flush?force=true",
        ):
            urllib.request.urlopen(urllib.request.Request(base + path, method="POST")).read()
        # Segment sizes, not _stats: Elasticsearch caches an index's store size for 10 seconds.
        with urllib.request.urlopen(
            f"{base}/_cat/segments/{index}?bytes=b&h=size&format=json"
        ) as r:
            segments = json.load(r)
        sizes[label] = sum(int(row["size"]) for row in segments)
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()
        edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
    driver.close()
    counts = {"neo4j nodes": nodes["c"] if nodes else 0, "neo4j edges": edges["c"] if edges else 0}
    return sizes, counts


@dataclass
class StackRun:
    """Per-document wall and client-CPU seconds, the checkpoints, and the host's load average."""

    seconds: list[float]
    cpu_seconds: list[float]
    points: list[Checkpoint]
    load_before: float
    load_after: float


def measure_stack(documents, settings, checkpoints: Sequence[int], count: TokenCounter):
    """Ingest through the real stack (replaying the LLM answers), timing each document.

    Returns the per-document seconds and the stores' sizes at each checkpoint.
    """
    from graph_rag.models import IngestTrigger
    from graph_rag.scope import DEFAULT_SCOPE
    from graph_rag.wiring import build_real_components

    components = build_real_components(settings)
    seconds: list[float] = []
    cpu: list[float] = []
    load_before = os.getloadavg()[0]
    points: list[Checkpoint] = []
    raw = 0
    elapsed = 0.0
    for n, doc in enumerate(documents, start=1):
        key = f"{doc.id}.txt"
        data = doc.text.encode("utf-8")
        raw += len(data)
        start, cpu_start = time.perf_counter(), time.process_time()
        components.object_store.put(DEFAULT_SCOPE, components.bucket, key, data)
        result = components.orchestrator.process_document(
            IngestTrigger(bucket=components.bucket, object_key=key)
        )
        seconds.append(time.perf_counter() - start)
        cpu.append(time.process_time() - cpu_start)
        elapsed += seconds[-1]
        if result is None:
            raise RuntimeError(f"the pipeline dropped document {doc.id}")
        if n in checkpoints:
            sizes, counts = store_sizes(settings, raw)
            points.append(Checkpoint(n, elapsed, sizes, counts))
    usage = components.llm_usage()
    if usage.calls:
        raise RuntimeError(f"the replay was not complete: {usage.calls} provider calls")
    return StackRun(seconds, cpu, points, load_before, os.getloadavg()[0])


def _size(value: float) -> str:
    return f"{value / 1_000_000:.2f} MB" if abs(value) >= 100_000 else f"{value / 1000:.0f} KB"


def to_markdown(
    *,
    command: str,
    documents: int,
    words: float,
    totals: dict[str, StageTokens],
    prices: dict[str, Price],
    tokenizer: str,
    seconds: Sequence[float] = (),
    cpu_seconds: Sequence[float] = (),
    load: tuple[float, float] | None = None,
    points: Sequence[Checkpoint] = (),
    notes: Sequence[str] = (),
    document_tokens: int = 0,
) -> str:
    """The report: tokens and dollars per 1,000 documents, then time and storage if measured."""
    out = [
        "# Ingest cost per 1,000 documents\n",
        f"`{command}`\n",
        f"{documents} documents, {words:.0f} words each on average. Scaled linearly to 1,000.\n",
        "## LLM tokens and dollars\n",
        "Tokens are counted from the prompts the two LLM stages build and the answers recorded "
        f"for them, with {tokenizer}. Dollars are tokens times the price shown, not billed.\n",
        "| stage | requests | prompt tokens | answer tokens |",
        "|---|---|---|---|",
    ]
    for stage, t in sorted(totals.items()):
        out.append(
            f"| {stage} | {per_thousand(t.requests, documents):,.0f} | "
            f"{per_thousand(t.prompt, documents):,.0f} | "
            f"{per_thousand(t.completion, documents):,.0f} |"
        )
    prompt_total = sum(t.prompt for t in totals.values())
    if document_tokens and prompt_total:
        out.append(
            f"\nThe documents' own text is {per_thousand(document_tokens, documents):,.0f} tokens "
            f"per 1,000, {100 * document_tokens / prompt_total:.0f}% of the prompt tokens (it goes "
            "into both stages' prompts); the rest is the instructions and entity list each request "
            "carries. These documents are short, so a longer one costs more than shown."
        )
    out += [
        "",
        "Per 1,000 documents, uncached (every request goes to the provider):\n",
        "| model | $ per million in / out | $ per 1,000 documents |",
        "|---|---|---|",
    ]
    for model, price in prices.items():
        out.append(
            f"| {model} | {price.input_per_million} / {price.output_per_million} | "
            f"${dollars_per_thousand(totals, price, documents):.2f} |"
        )
    out.append(
        "\nCached: $0, because a hit never reaches the provider. This replay was served entirely "
        "from the cache (zero provider calls), which is what a re-run of identical prompts gets.\n"
    )
    if seconds:
        s = summarise_seconds(seconds)
        c = summarise_seconds(cpu_seconds) if cpu_seconds else None
        out += [
            "## Time per document, everything except the LLM\n",
            "Real spaCy, bge, Elasticsearch and Neo4j; the LLM answers replayed from the cache, so "
            "this leaves out the provider's latency (not measured). One worker, one document at "
            "a time.\n",
            f"Wall time per document: mean {s['mean']:.2f}s, median {s['median']:.2f}s, 95th "
            f"percentile {s['p95']:.2f}s; 1,000 documents take about "
            f"{per_thousand(sum(seconds), len(seconds)) / 60:.1f} minutes.\n",
        ]
        if c:
            out.append(
                "CPU time of the ingesting process, summed over its threads (spaCy, bge, linking; "
                f"the stores' own work is in other processes): mean {c['mean']:.2f}s, median "
                f"{c['median']:.2f}s per document, so it exceeds the wall time.\n"
            )
        if load:
            out.append(
                f"The host's one-minute load average was {load[0]:.0f} before the run and "
                f"{load[1]:.0f} after. It is a shared machine, so read the wall times as one "
                "noisy measurement.\n"
            )
    if points:
        names, counted = list(points[-1].sizes), list(points[-1].counts)
        out += [
            "## Storage\n",
            "Elasticsearch is merged to one segment per index and flushed before each reading. "
            "The slope is a least-squares line through every checkpoint, so it leaves out each "
            "store's fixed size; a store whose size stops growing (entities are shared across "
            "documents) has no meaningful slope.\n",
            "| documents | " + " | ".join(names + counted) + " |",
            "|---|" + "---|" * (len(names) + len(counted)),
        ]
        for p in points:
            out.append(
                f"| {p.documents} | "
                + " | ".join(
                    [_size(v) for v in p.sizes.values()] + [str(v) for v in p.counts.values()]
                )
                + " |"
            )
        slopes = {name: slope_per_thousand(points, name) for name in names}
        cells = [_size(v) if v > 0 else "none" for v in slopes.values()]
        out.append("| per 1,000 (slope) | " + " | ".join(cells + ["" for _ in counted]) + " |")
        out.append("")
    out += [f"{note}\n" for note in notes]
    return "\n".join(out)
