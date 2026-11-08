"""``graphrag``: remember, recall, forget and lookup from a terminal (GRA-39, GRA-37).

Each command is one call to a running service's ``/v1`` API (ADR-0014). Point it at the service
with ``--url`` or ``GRAPH_RAG_URL`` (default ``http://localhost:8000``). The API key comes from
``GRAPH_RAG_API_KEY`` and has no flag, so it stays out of shell history. ``--json`` prints the
API's own JSON, so a script reads exactly what the HTTP contract says.

Exit status: 0 on success, 1 when the service answered with an error, 2 when it could not be
reached or the command line was wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

__all__ = ["main"]

_EPILOG = """\
environment:
  GRAPH_RAG_URL      the service to call (default http://localhost:8000)
  GRAPH_RAG_API_KEY  the API key, when the service requires one

exit status: 0 success, 1 the service returned an error, 2 it could not be reached or bad usage
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Store text in a graph-rag memory, ask it questions, inspect and forget.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", help="the service URL (default: GRAPH_RAG_URL or localhost:8000)")
    parser.add_argument("--json", action="store_true", help="print the API's JSON, not a summary")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    remember = sub.add_parser(
        "remember",
        help="store text",
        description="Store text. It is read into the graph in the background, so a recall "
        "straight afterwards may not see it yet.",
    )
    remember.add_argument("text", nargs="?", help="the text; omit it to read --file or stdin")
    remember.add_argument("--file", "-f", help="read the text from this file ('-' is stdin)")
    remember.add_argument(
        "--source-id",
        help="a stable name (URL, path, note id); the same id with the same text does nothing",
    )
    remember.add_argument("--namespace", help="keep this memory apart from other namespaces")
    remember.add_argument(
        "--document-date",
        metavar="YYYY-MM-DD",
        help="when the text was written; its facts are dated from then (for recall --as-of)",
    )
    remember.add_argument(
        "--force", action="store_true", help="process it again even if it is unchanged"
    )

    recall = sub.add_parser(
        "recall",
        help="answer a question, with its evidence",
        description="Answer a question from memory and list the sentences the answer rests on.",
    )
    recall.add_argument("question")
    recall.add_argument("--namespace", help="the namespace the memory was stored in")
    recall.add_argument(
        "--as-of", metavar="YYYY-MM-DD", help="answer from the facts that held on this day"
    )
    recall.add_argument(
        "--token-budget",
        type=int,
        help="cap the evidence at about this many tokens, keeping the best-ranked first",
    )
    recall.add_argument(
        "--synthesize", action="store_true", help="also write a grounded answer (uses an LLM)"
    )

    lookup = sub.add_parser(
        "lookup",
        help="describe an entity and its neighbours",
        description="Show the entities matching a name or id, with their aliases and the facts "
        "they take part in.",
    )
    which_entity = lookup.add_mutually_exclusive_group(required=True)
    which_entity.add_argument("--name", help="a name or alias; case and punctuation are ignored")
    which_entity.add_argument("--entity-id", help="a canonical entity id")
    lookup.add_argument("--limit", type=int, default=20, help="most neighbours per entity")
    lookup.add_argument("--namespace", help="the namespace to look in")

    forget = sub.add_parser(
        "forget",
        help="forget a source",
        description="Remove a source's text, facts and sentences, and the entities nothing else "
        "mentions. Name it by --source-id or --document-id.",
    )
    which = forget.add_mutually_exclusive_group(required=True)
    which.add_argument("--source-id", help="the id given to remember")
    which.add_argument("--document-id", help="the document_id remember returned")
    forget.add_argument("--namespace", help="the namespace the source was stored in")
    return parser


def _body(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """The path and JSON body a parsed command stands for."""
    extra = {"namespace": args.namespace} if args.namespace else {}
    if args.command == "remember":
        if args.text is not None and args.file:
            raise SystemExit("remember: give the text or --file, not both")
        if args.file == "-" or (args.text is None and not args.file):
            text = sys.stdin.read()
        elif args.file:
            with open(args.file, encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = args.text
        dated = {"document_date": args.document_date} if args.document_date else {}
        body = {"text": text, "force": args.force} | dated | extra
        return "/v1/remember", body | ({"source_id": args.source_id} if args.source_id else {})
    if args.command == "recall":
        budget = {"token_budget": args.token_budget} if args.token_budget is not None else {}
        body = {"question": args.question, "synthesize": args.synthesize}
        as_of = {"as_of": args.as_of} if args.as_of else {}
        return "/v1/recall", body | budget | as_of | extra
    if args.command == "lookup":
        which = {"name": args.name} if args.name else {"entity_id": args.entity_id}
        return "/v1/lookup", which | {"limit": args.limit} | extra
    key = {"source_id": args.source_id} if args.source_id else {"document_id": args.document_id}
    return "/v1/forget", key | extra


def _show(command: str, result: dict[str, Any]) -> str:
    """A person's view of a successful result."""
    if command == "remember":
        return f"{result['status']}: {result['source_id']} (document {result['document_id']})"
    if command == "forget":
        gone, kept = result["entities_removed"], result["entities_kept"]
        return (
            f"forgot {result['source_id']} (document {result['document_id']})\n"
            f"entities removed: {len(gone)}, kept because another source mentions them: {len(kept)}"
        )
    if command == "lookup":
        lines = []
        for entity in result["entities"]:
            aliases = f" (also {', '.join(entity['aliases'])})" if entity["aliases"] else ""
            lines.append(f"{entity['name']} [{entity['type']}] {entity['canonical_id']}{aliases}")
            for n in entity["neighbours"]:
                arrow = "->" if n["direction"] == "outgoing" else "<-"
                lines.append(f"  {arrow} {n['predicate']} {n['entity']['name']}")
                lines.append(f"     {n['provenance']['source_sentence']}")
        return "\n".join(lines)
    lines = [f"answer: {result['answer'] or '(nothing found)'}"]
    if result.get("prose"):
        lines += ["", result["prose"]]
    sources = result["supporting_sentences"]
    if result.get("truncated"):
        lines.append(f"(evidence cut to fit the budget; about {result['tokens_used']} tokens kept)")
    if sources:
        lines += ["", "sources:"]
        for n, s in enumerate(sources, 1):
            where = f"document {s['document_id']}, characters {s['char_start']}-{s['char_end']}"
            lines.append(f"  {n}. {s['text']}\n     {where}  score {s['score']:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, client: httpx.Client | None = None) -> int:
    """Run one command. ``client`` is for tests; it stands in for the real HTTP client."""
    args = _parser().parse_args(argv)
    path, body = _body(args)
    if client is None:
        try:
            import httpx
        except ImportError:
            print("graphrag needs httpx: install the 'cli' extra.", file=sys.stderr)
            return 2
        key = os.environ.get("GRAPH_RAG_API_KEY")
        client = httpx.Client(
            base_url=args.url or os.environ.get("GRAPH_RAG_URL", "http://localhost:8000"),
            timeout=60,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
    try:
        response = client.post(path, json=body)
    except Exception as error:  # noqa: BLE001 - any transport failure is "could not reach"
        print(f"error: could not reach the service: {error}", file=sys.stderr)
        return 2
    try:
        payload = response.json()
    except ValueError:
        print(f"error: the service answered {response.status_code} with no JSON", file=sys.stderr)
        return 1
    if response.is_error:
        if args.json:
            print(json.dumps(payload, indent=2), file=sys.stderr)
        else:
            print(f"error: {payload.get('error', {}).get('message', payload)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2) if args.json else _show(args.command, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
