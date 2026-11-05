"""A local OpenAI-compatible server for tests, and the answer a good model would give."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

from kggen.llm import OpenAICompatible

_FACT = re.compile(r"^F(\d+): (.*?) \| (.*?) \| (.*?)((?: \(.*?\))*)$", re.M)


def honest_answer(prompt: str) -> str:
    """What a good model would say: one sentence per listed fact, with names and years kept."""
    sentences = []
    for n, subject, relation, obj, notes in _FACT.findall(prompt):
        since = re.search(r"since (\d+)", notes)
        until = re.search(r"ended (\d+)", notes)
        tail = (f" since {since[1]}" if since else "") + (f" until {until[1]}" if until else "")
        sentences.append({"fact": f"F{n}", "text": f"{subject} {relation} {obj}{tail}."})
    return json.dumps({"sentences": sentences})


@contextmanager
def serve() -> Iterator[Callable[[Callable[[str], str] | None], OpenAICompatible]]:
    """Yields ``make(answer)``, which sets the answer function and returns a client for it.
    ``make.served`` lists every request body the server received."""
    state: dict[str, Callable[[str], str]] = {"answer": honest_answer}
    served: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            served.append(body)
            content = state["answer"](body["messages"][-1]["content"])
            payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def make(answer: Callable[[str], str] | None = None) -> OpenAICompatible:
        state["answer"] = answer or honest_answer
        return OpenAICompatible(f"http://127.0.0.1:{server.server_port}/v1", "test-model")

    make.served = served  # type: ignore[attr-defined]
    try:
        yield make
    finally:
        server.shutdown()


def rich_answer(prompt: str) -> str:
    """The rich style done well: the first sentence names the subject, later facts say "It",
    and one background sentence closes the article."""
    sentences, named = [], set()
    for n, subject, relation, obj, notes in _FACT.findall(prompt):
        since = re.search(r"since (\d+)", notes)
        until = re.search(r"ended (\d+)", notes)
        tail = (f" since {since[1]}" if since else "") + (f" until {until[1]}" if until else "")
        lead = "It" if subject in named else subject
        named.add(subject)
        sentences.append({"facts": [f"F{n}"], "text": f"{lead} {relation} {obj}{tail}."})
    sentences.append({"facts": [], "text": "Records of this kind are kept for many years."})
    return json.dumps({"sentences": sentences})
