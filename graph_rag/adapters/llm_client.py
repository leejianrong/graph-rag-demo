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
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from graph_rag.logging import get_logger
from graph_rag.scope import DEFAULT_TENANT, current_tenant, tenant_dir

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.tenant_keys import TenantLlmKeys

__all__ = ["LiteLLMClient", "cache_key", "tenant_cache_dir"]

_logger = get_logger(__name__)

_StructuredT = TypeVar("_StructuredT", bound=BaseModel)

# A backend takes the model + fully-rendered prompt (+ any provider params) and
# returns the raw completion text. This is the seam tests patch to count calls.
CompletionFn = Callable[..., str]


def tenant_cache_dir(cache_dir: str | Path, tenant: str) -> Path:
    """Where ``tenant``'s cached responses live: the root for the default tenant, else its own."""
    root = Path(cache_dir)
    return root if tenant == DEFAULT_TENANT else root / "tenants" / tenant_dir(tenant)


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
        tenant_keys: TenantLlmKeys | None = None,
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
            tenant_keys: Per-tenant provider keys (GRA-25). When set, a named tenant's provider
                calls use its own key and raise ``LlmKeyRequired`` without one, never falling back
                to ``api_key``; the default tenant is unchanged. Cache hits need no key.
        """
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._max_retries = max_retries
        self._api_key = api_key
        self._tenant_keys = tenant_keys
        self._completion_fn: CompletionFn = completion_fn or self._default_completion
        # Usage since construction, read by the bench to price a run (GRA-87).
        self.calls = 0  # provider calls, retries included; a cache hit is not one
        self.cache_hits = 0
        self.seconds = 0.0  # wall time spent waiting on the provider
        self.cost_usd = 0.0  # LiteLLM's price estimate; 0 for a model it cannot price

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None) -> LiteLLMClient:
        """Construct from :class:`~graph_rag.config.Settings`.

        Uses ``model`` (default ``settings.coref_model``, the V3 stage model, B6), the shared LLM
        cache dir, retry budget, the env-sourced API key and, with ``require_tenant_llm_key`` on,
        the tenants' own keys.
        """
        from graph_rag.tenant_keys import TenantLlmKeys

        return cls(
            model=model or settings.coref_model,
            cache_dir=settings.llm_cache_dir,
            max_retries=settings.llm_max_retries,
            api_key=settings.openai_api_key,
            tenant_keys=TenantLlmKeys.from_settings(settings),
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
            self.cache_hits += 1
            return cached
        text = self._call_provider(prompt, params)
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
            self.cache_hits += 1
            return schema.model_validate_json(_extract_json(cached))

        last_error: Exception | None = None
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            text = self._call_provider(full_prompt, params)
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

    def _call_provider(self, prompt: str, params: Mapping[str, Any]) -> str:
        """One provider call through the backend seam, counted and timed."""
        # A tenant's own key goes through the seam, so tests can see which key a call used.
        own = self._tenant_keys.require(current_tenant()) if self._tenant_keys else None
        extra = {"api_key": own} if own else {}
        start = time.perf_counter()
        try:
            return self._completion_fn(model=self._model, prompt=prompt, **params, **extra)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - start

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
        """Return the on-disk path for a cache ``key``, in the acting tenant's own directory.

        The default tenant keeps the cache root, so entries written before tenancy still hit.
        Any other tenant gets a directory of its own: identical text in two tenants is two
        entries, and one tenant never reads what another's documents produced (GRA-139).
        """
        return self._tenant_cache_dir() / f"{key}.json"

    def _tenant_cache_dir(self) -> Path:
        return tenant_cache_dir(self._cache_dir, current_tenant())

    def _read_cache(self, key: str) -> str | None:
        """Return the cached completion text for ``key``, or ``None`` on a miss."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, key: str, text: str) -> None:
        """Persist ``text`` under ``key`` (creating the cache dir if needed)."""
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _default_completion(self, *, model: str, prompt: str, **params: Any) -> str:
        """Default backend: call ``litellm.completion`` (imported lazily).

        Isolated here so the module imports and the client constructs without
        LiteLLM present; only a real, uncached call reaches this seam.
        """
        import litellm  # lazy: keeps import/construction offline + key-free

        kwargs: dict[str, Any] = dict(params)
        if self._api_key and "api_key" not in kwargs:
            kwargs["api_key"] = self._api_key
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        try:
            self.cost_usd += float(litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 - an unpriced model (a local or recording server) is free
            _logger.debug("no price for model %s", model)
        return response.choices[0].message.content or ""
