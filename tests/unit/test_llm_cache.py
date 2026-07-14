"""Unit seams for the LiteLLM client — cache, cache-key stability, retry (§4).

All offline: the provider is replaced by a **call-counting fake backend** injected
through the ``completion_fn`` seam, and the cache writes under a per-test
``tmp_path``. No network, no API key, no LiteLLM import. Proves:

* **Cache-key stability** (ADR-0008): identical ``(model, prompt, params)`` → same
  key; changing any → a different key.
* **Cache hit is $0** (R6.2): two identical structured calls invoke the backend
  **once**; the second is served from cache and returns an equal object.
* **Validation + retry** (ADR-0008): malformed-then-valid JSON retries and
  succeeds; always-malformed raises after exhausting the retry budget, and a
  malformed response is never cached.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from graph_rag.adapters.llm_client import LiteLLMClient, cache_key
from graph_rag.models import ClusterMap


class _Shape(BaseModel):
    """A tiny schema for structured-output tests."""

    value: int


class _CountingBackend:
    """A fake ``completion_fn`` that returns scripted texts and counts calls."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, *, model: str, prompt: str, **params: object) -> str:
        self.calls += 1
        # Repeat the last scripted response once the script is exhausted.
        idx = min(self.calls - 1, len(self._responses) - 1)
        return self._responses[idx]


# --- Cache-key stability (ADR-0008) -----------------------------------------


def test_cache_key_is_stable_for_identical_inputs() -> None:
    """Same (model, prompt, params) → same key; params order does not matter."""
    a = cache_key("gpt-4o-mini", "hello", {"temperature": 0, "top_p": 1})
    b = cache_key("gpt-4o-mini", "hello", {"top_p": 1, "temperature": 0})
    assert a == b


def test_cache_key_changes_with_model_prompt_or_params() -> None:
    """Changing the model, prompt, or any param yields a different key."""
    base = cache_key("gpt-4o-mini", "hello", {"temperature": 0})
    assert cache_key("gpt-4o", "hello", {"temperature": 0}) != base  # model
    assert cache_key("gpt-4o-mini", "goodbye", {"temperature": 0}) != base  # prompt
    assert cache_key("gpt-4o-mini", "hello", {"temperature": 1}) != base  # params


# --- Cache hit is $0 (R6.2) -------------------------------------------------


def test_second_identical_structured_call_is_a_cache_hit(tmp_path) -> None:
    """Two identical structured calls hit the provider once; second from cache."""
    backend = _CountingBackend('{"value": 7}')
    client = LiteLLMClient(model="test-model", cache_dir=tmp_path, completion_fn=backend)

    first = client.structured("extract", _Shape)
    second = client.structured("extract", _Shape)

    assert backend.calls == 1  # provider invoked exactly once
    assert first == second == _Shape(value=7)


def test_second_identical_complete_call_is_a_cache_hit(tmp_path) -> None:
    """Plain ``complete`` is cached too: identical prompt → one backend call."""
    backend = _CountingBackend("the answer")
    client = LiteLLMClient(model="test-model", cache_dir=tmp_path, completion_fn=backend)

    assert client.complete("q") == "the answer"
    assert client.complete("q") == "the answer"
    assert backend.calls == 1


def test_structured_cluster_map_round_trips(tmp_path) -> None:
    """A ClusterMap structured response validates and round-trips (coref shape)."""
    backend = _CountingBackend(
        '{"clusters": [{"canonical": "Ada Lovelace", "members": ["Ada", "She"]}]}'
    )
    client = LiteLLMClient(model="test-model", cache_dir=tmp_path, completion_fn=backend)

    result = client.structured("resolve coref", ClusterMap)

    assert len(result.clusters) == 1
    assert result.clusters[0].canonical == "Ada Lovelace"
    assert result.clusters[0].members == ["Ada", "She"]


# --- Validation + retry (ADR-0008) ------------------------------------------


def test_structured_retries_malformed_then_succeeds(tmp_path) -> None:
    """Malformed JSON then valid JSON → retried, succeeds, backend called twice."""
    backend = _CountingBackend("not json at all", '{"value": 42}')
    client = LiteLLMClient(
        model="test-model", cache_dir=tmp_path, max_retries=2, completion_fn=backend
    )

    result = client.structured("extract", _Shape)

    assert result == _Shape(value=42)
    assert backend.calls == 2  # one failed attempt + one success


def test_structured_raises_after_exhausting_retries(tmp_path) -> None:
    """Always-malformed JSON raises after ``max_retries + 1`` attempts."""
    backend = _CountingBackend("still not json")
    client = LiteLLMClient(
        model="test-model", cache_dir=tmp_path, max_retries=2, completion_fn=backend
    )

    with pytest.raises(ValueError, match="did not validate"):
        client.structured("extract", _Shape)

    assert backend.calls == 3  # max_retries (2) + 1


def test_malformed_response_is_not_cached(tmp_path) -> None:
    """A failed call caches nothing: a later valid call still hits the backend."""
    # First: always-malformed backend raises and must not write a cache entry.
    bad = _CountingBackend("nope")
    bad_client = LiteLLMClient(
        model="test-model", cache_dir=tmp_path, max_retries=1, completion_fn=bad
    )
    with pytest.raises(ValueError):
        bad_client.structured("extract", _Shape)

    # A fresh client (same cache dir, same prompt/model) with a valid backend must
    # still call the backend — proving the malformed response was never cached.
    good = _CountingBackend('{"value": 1}')
    good_client = LiteLLMClient(model="test-model", cache_dir=tmp_path, completion_fn=good)
    assert good_client.structured("extract", _Shape) == _Shape(value=1)
    assert good.calls == 1
