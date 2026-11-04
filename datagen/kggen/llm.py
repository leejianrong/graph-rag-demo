"""A small OpenAI-compatible chat client and an on-disk response cache, standard library only.

The generator never imports ``graph_rag`` or ``litellm``. Anything that speaks the Chat Completions
protocol works as the endpoint: a provider, a local server, or a recording fake such as semula.
Cached responses are plain JSON files that record the model and the full prompt, so a cache can be
committed and a re-run needs no endpoint at all.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

Messages = list[dict[str, str]]


class CacheMiss(RuntimeError):
    """The request is not cached and no endpoint was given to ask."""


@dataclass(frozen=True)
class OpenAICompatible:
    """``POST {base_url}/chat/completions`` with a fixed model and temperature 0."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 600.0  # a recording server may hold a call until someone answers it

    def complete(self, messages: Messages) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(  # noqa: S310 - the URL is the caller's chosen endpoint
            self.base_url.rstrip("/") + "/chat/completions", body, headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]


class ResponseCache:
    """One JSON file per request, named by a hash of the model and messages."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(model: str, messages: Messages) -> str:
        blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, model: str, messages: Messages) -> str | None:
        path = self.directory / f"{self.key(model, messages)}.json"
        return json.loads(path.read_text())["response"] if path.exists() else None

    def put(self, model: str, messages: Messages, response: str, **note: object) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"model": model, "messages": messages, "response": response, **note}
        path = self.directory / f"{self.key(model, messages)}.json"
        path.write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")


class CachedChat:
    """Ask the cache first, then the endpoint. Counts calls so a warm run can prove it made none."""

    def __init__(
        self, cache: ResponseCache, client: OpenAICompatible | None = None, model: str | None = None
    ) -> None:
        if client is None and model is None:
            raise ValueError("a cache-only chat needs the model name the cache was built with")
        self.cache, self.client = cache, client
        self.model = client.model if client else str(model)
        self.hits = 0
        self.calls = 0

    def cached(self, messages: Messages) -> str | None:
        found = self.cache.get(self.model, messages)
        if found is not None:
            self.hits += 1
        return found

    def ask(self, messages: Messages) -> str:
        """The cached response, else a fresh one (not stored: call ``store`` once it is valid)."""
        found = self.cached(messages)
        if found is not None:
            return found
        if self.client is None:
            raise CacheMiss("not in the cache and no endpoint was given")
        self.calls += 1
        return self.client.complete(messages)

    def store(self, messages: Messages, response: str, **note: object) -> None:
        self.cache.put(self.model, messages, response, **note)
