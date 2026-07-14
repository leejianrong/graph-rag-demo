"""LiteLLM adapter for the ``LLMClient`` port (ADR-0008, ADR-0010).

The first real LLM use (V3, coref). Wraps LiteLLM so the provider/model is a
per-stage config choice (``gpt-4o-mini`` default; any OpenAI-compatible endpoint
swappable via ``.env``) and adds the two cross-cutting behaviours ADR-0008 asks
for:

* **Structured output** — :meth:`LiteLLMClient.structured` requests JSON, validates
  it against a caller-supplied Pydantic model, and **retries** on parse/validation
  failure before raising. Keeps parsing reliable across providers whose JSON-mode
  support varies.
* **Persistent response cache** — every call is keyed by
  ``sha256(model + prompt + serialized params)`` (see :func:`cache_key`) under a
  gitignored dir. A cache hit returns the stored response and **never** calls the
  provider, so repeated pipeline/benchmark runs cost ``$0`` (R6.2).

The provider call is isolated behind an **injectable backend seam** — the
``completion_fn`` constructor argument (default: :meth:`_default_completion`, which
lazily imports and calls ``litellm.completion``). Tests inject a call-counting
fake backend to prove cache/retry behaviour with no network and no key. LiteLLM is
imported lazily inside the default seam only, so importing this module and
constructing the client never require ``litellm`` to be importable or an API key
to be set — the fast suite stays offline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from graph_rag.logging import get_logger

if TYPE_CHECKING:
    from graph_rag.config import Settings

__all__ = ["LiteLLMClient", "cache_key"]

_logger = get_logger(__name__)

_StructuredT = TypeVar("_StructuredT", bound=BaseModel)

# A backend takes the model + fully-rendered prompt (+ any provider params) and
# returns the raw completion text. This is the seam tests patch to count calls.
CompletionFn = Callable[..., str]


def cache_key(model: str, prompt: str, params: Mapping[str, Any]) -> str:
    """Return the response-cache key ``sha256(model + prompt + serialized params)``.

    A pure, side-effect-free function (ADR-0008): identical ``(model, prompt,
    params)`` always yields the same key, and changing any of the three yields a
    different key — so improving a prompt or switching models naturally bypasses
    stale entries. ``params`` is serialized with sorted keys so key ordering does
    not matter; non-JSON values fall back to ``str``.

    Args:
        model: The LiteLLM model string (e.g. ``"gpt-4o-mini"``).
        prompt: The fully-rendered prompt sent to the provider.
        params: Provider params (temperature, etc.) folded into the key.

    Returns:
        The hex sha256 digest used as the cache filename.
    """
    serialized = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a model completion.

    Strips Markdown code fences and any prose around the object, since providers
    sometimes wrap JSON output. A no-op for the clean JSON the fakes return.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


class LiteLLMClient:
    """LiteLLM-backed :class:`~graph_rag.ports.LLMClient` (V3-active, ADR-0008).

    Provider-agnostic (the ``model`` is any LiteLLM model string), with a
    persistent response cache and Pydantic-validated structured output + retry.
    Construction is cheap and offline: no LiteLLM import, no network, no API key
    required until an actual (uncached) provider call is made.
    """

    def __init__(
        self,
        model: str,
        cache_dir: str | Path = ".cache/llm",
        max_retries: int = 2,
        api_key: str | None = None,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        """Wire the client.

        Args:
            model: LiteLLM model string (e.g. ``"gpt-4o-mini"``).
            cache_dir: Directory for the persistent response cache (gitignored).
            max_retries: Extra attempts after the first on structured parse /
                validation failure (so total attempts = ``max_retries + 1``).
            api_key: Provider API key. Passed to LiteLLM when set; otherwise
                LiteLLM falls back to its own provider env vars. Never hardcoded.
            completion_fn: The injectable backend seam. Defaults to
                :meth:`_default_completion` (lazy ``litellm.completion``); tests
                inject a call-counting fake so no network/key is needed.
        """
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._max_retries = max_retries
        self._api_key = api_key
        self._completion_fn: CompletionFn = completion_fn or self._default_completion

    @classmethod
    def from_settings(cls, settings: Settings) -> LiteLLMClient:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``settings.coref_model`` (the V3 stage model, B6), the shared LLM
        cache dir, retry budget and the env-sourced API key.
        """
        return cls(
            model=settings.coref_model,
            cache_dir=settings.llm_cache_dir,
            max_retries=settings.llm_max_retries,
            api_key=settings.openai_api_key,
        )

    # --- Public API (the LLMClient port) ------------------------------------

    def complete(self, prompt: str, **params: Any) -> str:
        """Return the model completion for ``prompt``, served through the cache.

        A cache hit returns the stored text without calling the provider.
        """
        key = cache_key(self._model, prompt, params)
        cached = self._read_cache(key)
        if cached is not None:
            _logger.debug("LLM cache hit (complete) %s", key[:12])
            return cached
        text = self._completion_fn(model=self._model, prompt=prompt, **params)
        self._write_cache(key, text)
        return text

    def structured(self, prompt: str, schema: type[_StructuredT], **params: Any) -> _StructuredT:
        """Return a validated ``schema`` instance for ``prompt`` (ADR-0008).

        Renders the schema into the prompt as an instruction, requests JSON, and
        validates against ``schema``. On a parse/validation failure the provider
        is re-called up to ``max_retries`` more times before raising; only the
        final *valid* response is cached (a malformed response is never cached).
        A cache hit re-validates the stored JSON and never calls the provider.

        Raises:
            ValueError: If no valid instance is produced within the retry budget.
        """
        full_prompt = self._render_structured_prompt(prompt, schema)
        key = cache_key(self._model, full_prompt, params)

        cached = self._read_cache(key)
        if cached is not None:
            _logger.debug("LLM cache hit (structured %s) %s", schema.__name__, key[:12])
            return schema.model_validate_json(_extract_json(cached))

        last_error: Exception | None = None
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            text = self._completion_fn(model=self._model, prompt=full_prompt, **params)
            try:
                instance = schema.model_validate_json(_extract_json(text))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                _logger.warning(
                    "structured output failed validation against %s (attempt %d/%d)",
                    schema.__name__,
                    attempt,
                    attempts,
                )
                continue
            self._write_cache(key, text)
            return instance

        raise ValueError(
            f"LLM structured output did not validate against {schema.__name__} "
            f"after {attempts} attempt(s)"
        ) from last_error

    # --- Internals ----------------------------------------------------------

    @staticmethod
    def _render_structured_prompt(prompt: str, schema: type[BaseModel]) -> str:
        """Append the JSON-Schema instruction so the model returns matching JSON.

        Folding the schema into the prompt also makes the cache key schema-aware:
        a different schema changes the prompt and therefore the key.
        """
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        return (
            f"{prompt}\n\n"
            "Respond with ONLY a single JSON object (no prose, no code fences) "
            f"that conforms to this JSON Schema:\n{schema_json}"
        )

    def _cache_path(self, key: str) -> Path:
        """Return the on-disk path for a cache ``key``."""
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        """Return the cached completion text for ``key``, or ``None`` on a miss."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, key: str, text: str) -> None:
        """Persist ``text`` under ``key`` (creating the cache dir if needed)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(key).write_text(text, encoding="utf-8")

    def _default_completion(self, *, model: str, prompt: str, **params: Any) -> str:
        """Default backend: call ``litellm.completion`` (imported lazily).

        Isolated here so the module imports and the client constructs without
        LiteLLM present; only a real, uncached call reaches this seam.
        """
        import litellm  # lazy: keeps import/construction offline + key-free

        kwargs: dict[str, Any] = dict(params)
        if self._api_key:
            kwargs["api_key"] = self._api_key
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""
