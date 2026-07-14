"""The runnable demo — reproduce the multi-hop example from the README.

Two ways to run it, both over the SAME bundled corpus
(``examples/supply-chain/``, see that directory's README):

* **offline** (default) — builds a real graph and answers a real multi-hop query
  with **no Docker, no model, no API key**, using the deterministic offline
  pipeline (:func:`graph_rag.benchmark.pipeline.build_offline_components`):
  Title-Case-heuristic NER + co-occurrence KG-build + merge-by-name entity
  linking + the real V6 retriever over a :class:`~graph_rag.fakes.FakeEmbedder`.
  Every edge is a generic ``RELATED_TO``, but the cross-document connection is
  real and the run is ``$0`` and reproducible.

* **--http URL** — drives the REAL running stack over HTTP: ``POST /ingest`` for
  each document (in a fixed order — entity linking is order-sensitive), waits for
  the asynchronous pipeline to build the graph, then ``POST /query``. This is the
  authentic product path (spaCy NER + LLM coref/KG-build, typed edges, optional
  ``--synthesize`` prose) and needs ``make up`` plus an ``OPENAI_API_KEY`` in the
  service's environment.

Both paths print the predicted answer, the two-hop connection between the named
endpoints, the connected subgraph, and the supporting sentences with provenance.

Run it via ``make demo-offline`` / ``make demo`` or directly::

    python -m graph_rag.demo                       # offline
    python -m graph_rag.demo --http http://localhost:8000 [--synthesize]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from graph_rag.benchmark.pipeline import build_offline_components
from graph_rag.ids import document_id
from graph_rag.models import IngestTrigger, QueryRequest, QueryResponse, Triple
from graph_rag.normalize import normalize_name

__all__ = ["main", "build_parser"]

# The bundled corpus and the question it is authored to answer (see the corpus README).
_CORPUS_DIR = Path(__file__).resolve().parents[1] / "examples" / "supply-chain"
_QUESTION = "How is Aurelia Components connected to the German Supply Chain Act?"


# --- corpus loading ---------------------------------------------------------


def _load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    """Return ``(filename, text)`` for every ``.md`` file, sorted by name.

    The sort pins a FIXED ingestion order: entity linking is order-sensitive
    (ADR-0004), so a stable order makes the constructed graph reproducible.
    """
    files = sorted(corpus_dir.glob("*.md"))
    # The corpus's own README is documentation, not an input document.
    return [(p.name, p.read_text(encoding="utf-8")) for p in files if p.name != "README.md"]


# --- rendering (shared by both modes) ---------------------------------------


def _endpoint_nodes(question: str, response: QueryResponse) -> list[str]:
    """The subgraph node ids whose name best overlaps the question's tokens.

    Used to render the connection between the two entities the question names
    (here: *Aurelia Components* and *German Supply Chain Act*). Returns up to two
    canonical ids, most-overlapping first; ties broken by name for determinism.
    """
    q_list = normalize_name(question).split()
    q_tokens = set(q_list)
    scored: list[tuple[int, str, str]] = []
    for node in response.subgraph.nodes:
        overlap = len(q_tokens & set(normalize_name(node.name).split()))
        if overlap:
            scored.append((overlap, node.name, node.canonical_id))
    scored.sort(key=lambda row: (-row[0], row[1]))
    top = [(name, cid) for _, name, cid in scored[:2]]

    # Order the two endpoints by where they first appear in the question, so the
    # rendered path reads in the question's own direction (subject → object).
    def _first_position(name: str) -> int:
        node_tokens = set(normalize_name(name).split())
        return next((i for i, token in enumerate(q_list) if token in node_tokens), len(q_list))

    top.sort(key=lambda pair: _first_position(pair[0]))
    return [cid for _, cid in top]


def _shortest_path(edges: list[Triple], source: str, target: str) -> list[str] | None:
    """BFS the undirected subgraph for a shortest ``source``→``target`` node path."""
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.subject_id, set()).add(edge.object_id)
        adjacency.setdefault(edge.object_id, set()).add(edge.subject_id)
    if source not in adjacency or target not in adjacency:
        return None
    queue: deque[list[str]] = deque([[source]])
    seen = {source}
    while queue:
        path = queue.popleft()
        if path[-1] == target:
            return path
        for neighbour in sorted(adjacency[path[-1]]):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append([*path, neighbour])
    return None


def _edge_between(edges: list[Triple], a: str, b: str) -> Triple | None:
    """Return an edge joining ids ``a`` and ``b`` (either direction), or ``None``."""
    for edge in edges:
        if {edge.subject_id, edge.object_id} == {a, b}:
            return edge
    return None


def _render(
    response: QueryResponse,
    *,
    question: str,
    doc_labels: dict[str, str],
    header: str,
) -> str:
    """Render a query response as the demo scorecard (answer + connection + evidence)."""
    names = {node.canonical_id: node.name for node in response.subgraph.nodes}
    lines = [header, "=" * len(header), f"question : {question}", ""]

    # Predicted answer (top-ranked entity).
    lines += ["Answer", "------"]
    if response.answer_entity is not None:
        lines.append(
            f"  {response.answer_entity.name}  "
            f"(top-ranked entity, score {response.answer_entity.score:.2f})"
        )
    else:
        lines.append("  (no entity retrieved)")
    lines.append("")

    # The connection between the two entities the question names.
    endpoints = _endpoint_nodes(question, response)
    path = (
        _shortest_path(response.subgraph.edges, endpoints[0], endpoints[1])
        if len(endpoints) == 2
        else None
    )
    if path and len(path) >= 2:
        hops = len(path) - 1
        connection_header = f"Connection ({hops} hop{'s' if hops != 1 else ''})"
        lines += [connection_header, "-" * len(connection_header)]
        lines.append(f"  {names.get(path[0], path[0])}")
        for prev, current in zip(path, path[1:], strict=False):
            edge = _edge_between(response.subgraph.edges, prev, current)
            predicate = edge.predicate if edge else "?"
            lines.append(f"    --[{predicate}]--> {names.get(current, current)}")
            if edge is not None:
                label = doc_labels.get(edge.provenance.source_doc_id, edge.provenance.source_doc_id)
                lines.append(f'        "{edge.provenance.source_sentence.strip()}"')
                lines.append(f"        ({label}, sentence {edge.provenance.sentence_index})")
        lines.append("")

    # The connected subgraph the answer was read from.
    lines += ["Connected subgraph", "------------------"]
    node_names = sorted(names.values())
    lines.append(f"  nodes ({len(node_names)}): {', '.join(node_names) or '(none)'}")
    lines.append(f"  edges ({len(response.subgraph.edges)}):")
    for edge in response.subgraph.edges:
        label = doc_labels.get(edge.provenance.source_doc_id, edge.provenance.source_doc_id)
        subject = names.get(edge.subject_id, edge.subject_id)
        obj = names.get(edge.object_id, edge.object_id)
        lines.append(
            f"    {subject} --{edge.predicate}--> {obj}  "
            f"[{label} #{edge.provenance.sentence_index}]"
        )
    lines.append("")

    # Supporting-sentence evidence with provenance.
    lines += ["Supporting sentences", "--------------------"]
    if response.supporting_sentences:
        for sentence in response.supporting_sentences:
            label = doc_labels.get(sentence.document_id, sentence.document_id)
            lines.append(
                f'  [{sentence.score:.2f}] "{sentence.text.strip()}"  '
                f"({label} #{sentence.sentence_index})"
            )
    else:
        lines.append("  (none)")

    if response.prose:
        lines += ["", "Prose answer (LLM synthesis)", "-" * 28, response.prose]

    return "\n".join(lines)


# --- offline mode -----------------------------------------------------------


def run_offline(corpus_dir: Path, question: str) -> int:
    """Ingest the corpus through the offline pipeline, query it, print the result."""
    documents = _load_corpus(corpus_dir)
    if not documents:
        print(f"No .md documents found in {corpus_dir}.", file=sys.stderr)
        return 1

    components = build_offline_components()
    bucket = components.bucket
    doc_labels: dict[str, str] = {}
    for filename, text in documents:
        components.object_store.put(bucket, filename, text.encode("utf-8"))
        components.orchestrator.process_document(IngestTrigger(bucket=bucket, object_key=filename))
        doc_labels[document_id(bucket, filename)] = filename

    response = components.retriever.retrieve(QueryRequest(question=question))
    header = "Graph RAG demo — offline (heuristic, $0, no LLM)"
    print(_render(response, question=question, doc_labels=doc_labels, header=header))
    return 0


# --- http mode (real stack) -------------------------------------------------


def _multipart(filename: str, content: bytes) -> tuple[bytes, str]:
    """Encode one ``file=`` multipart/form-data body; return ``(body, content_type)``."""
    boundary = "----graphragdemoboundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: text/markdown\r\n\r\n",
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _post(url: str, data: bytes, content_type: str, *, timeout: float = 30.0) -> bytes:
    """POST ``data`` and return the response body (raises on non-2xx / network error)."""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (local URL)
        return response.read()


def run_http(
    base_url: str,
    corpus_dir: Path,
    question: str,
    *,
    synthesize: bool = False,
    poll_timeout: float = 90.0,
) -> int:
    """Drive the real running stack over HTTP: ingest → wait → query → print."""
    base_url = base_url.rstrip("/")
    documents = _load_corpus(corpus_dir)
    if not documents:
        print(f"No .md documents found in {corpus_dir}.", file=sys.stderr)
        return 1

    # Health check first, with a friendly hint if the stack isn't up.
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:  # noqa: S310
            response.read()
    except urllib.error.URLError as error:
        print(
            f"Cannot reach the service at {base_url} ({error}).\n"
            "Bring the stack up first with `make up` (docker compose up --build).",
            file=sys.stderr,
        )
        return 1

    # Ingest every document in a fixed order (order-sensitive entity linking).
    doc_labels: dict[str, str] = {}
    print(f"Ingesting {len(documents)} document(s) into {base_url} ...")
    for filename, text in documents:
        body, content_type = _multipart(filename, text.encode("utf-8"))
        raw = _post(f"{base_url}/ingest", body, content_type)
        doc_id = json.loads(raw)["document_id"]
        doc_labels[doc_id] = filename
        print(f"  ingested {filename}  (document_id={doc_id})")

    # Ingestion is asynchronous. Poll /query until the graph has edges (or time out).
    query_body = QueryRequest(question=question, synthesize=synthesize).to_json().encode("utf-8")
    print("Waiting for the pipeline to build the graph ...")
    deadline = time.monotonic() + poll_timeout
    response: QueryResponse | None = None
    while time.monotonic() < deadline:
        raw = _post(f"{base_url}/query", query_body, "application/json")
        response = QueryResponse.from_json(raw)
        if response.subgraph.edges:
            break
        time.sleep(3.0)

    if response is None or not response.subgraph.edges:
        print(
            "\nThe graph came back empty after waiting.\n"
            "The coreference and KG-build stages need an LLM: make sure OPENAI_API_KEY\n"
            "is set in .env (or COREF_MODEL/KG_BUILD_MODEL point at another provider),\n"
            "then re-run. Check `make logs` for per-document errors.",
            file=sys.stderr,
        )
        return 1

    header = "Graph RAG demo — real stack over HTTP"
    print()
    print(_render(response, question=question, doc_labels=doc_labels, header=header))
    return 0


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m graph_rag.demo`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="graph_rag.demo",
        description="Reproduce the multi-hop supply-chain example over the bundled corpus.",
    )
    parser.add_argument(
        "--http",
        metavar="URL",
        default=None,
        help="Drive the real running stack at this base URL (e.g. http://localhost:8000) "
        "instead of the offline pipeline.",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Ask for an LLM prose answer too (http mode only; needs a key).",
    )
    parser.add_argument(
        "--corpus",
        default=str(_CORPUS_DIR),
        help="Directory of .md documents to ingest (default: the bundled supply-chain corpus).",
    )
    parser.add_argument(
        "--question", default=_QUESTION, help="The question to ask (default: the demo question)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    corpus_dir = Path(args.corpus)
    if args.http:
        return run_http(args.http, corpus_dir, args.question, synthesize=args.synthesize)
    return run_offline(corpus_dir, args.question)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
