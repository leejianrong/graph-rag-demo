"""Whether an LLM is configured for ingest, and what a failed LLM call means (GRA-154).

Ingest needs an LLM (coreference and fact extraction); querying does not. Without this check a
document is accepted and its job then fails with ``InternalServerError``. Two pure helpers:

* :func:`llm_not_configured` says, from the settings and the environment, when nothing is set up
  for the models ingest uses. It is **conservative**: it answers only when it can tell.
* :func:`explain_llm_error` turns a provider authentication or connection failure into a readable
  sentence, with no key and no URL (only the host of the configured endpoint).

The rule for "configured" is: a provider key (``OPENAI_API_KEY``, i.e. ``settings.openai_api_key``)
or an OpenAI-compatible endpoint (``OPENAI_API_BASE``, which may accept a dummy key). It applies
to stage models whose provider is OpenAI (a bare ``gpt-*``/``o1``/``o3``/``o4`` name or an
``openai/`` prefix). Any other model name has a key variable we cannot know, so it is never refused.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from urllib.parse import urlsplit

from graph_rag.config import Settings

__all__ = ["LlmNotConfigured", "explain_llm_error", "llm_not_configured", "log_if_llm_missing"]

_OPENAI_BARE = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_AUTH_ERRORS = {"AuthenticationError", "PermissionDeniedError"}
_CONNECTION_ERRORS = {"APIConnectionError", "APITimeoutError", "Timeout", "ServiceUnavailableError"}
_MISSING_KEY_TEXT = ("missing credentials", "api_key", "incorrect api key")
# LiteLLM often wraps a refused connection, a DNS failure or a timeout in a bare InternalServerError
# (``OpenAIException - Connection error.``); only these phrases make it "unreachable". A genuine
# provider 500 carries none of them, so it is left unexplained rather than mislabelled.
_UNREACHABLE_TEXT = (
    "connection error",
    "connection refused",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "timed out",
    "max retries exceeded",
)


class LlmNotConfigured(Exception):
    """Ingest was asked for, but no LLM provider key or endpoint is configured."""


def _is_openai(model: str) -> bool:
    name = model.strip().lower()
    if "/" in name:
        return name.split("/", 1)[0] == "openai"
    return name.startswith(_OPENAI_BARE)


def llm_not_configured(settings: Settings, environ: Mapping[str, str] | None = None) -> str | None:
    """A message on how to configure an LLM, or ``None`` when ingest can run (or we cannot tell).

    Refuses only when the coref or KG-build model is an OpenAI one and neither a key nor an
    ``OPENAI_API_BASE`` is set. With ``require_tenant_llm_key`` on, tenants bring their own keys
    and the per-tenant check applies instead, so this answers ``None``.
    """
    env = os.environ if environ is None else environ
    if settings.require_tenant_llm_key:
        return None
    if not any(_is_openai(m) for m in (settings.coref_model, settings.kg_build_model)):
        return None
    if (settings.openai_api_key or "").strip() or env.get("OPENAI_API_KEY", "").strip():
        return None
    if env.get("OPENAI_API_BASE", "").strip():
        return None
    return (
        "Ingest needs an LLM and none is configured. Set OPENAI_API_KEY, or point "
        "OPENAI_API_BASE at an OpenAI-compatible endpoint (a hosted provider, or a local "
        "server such as Ollama or vLLM), then restart. Querying needs neither. "
        "See docs/QUICKSTART.md, 'Give ingest an LLM'."
    )


def log_if_llm_missing(
    settings: Settings, logger: logging.Logger, environ: Mapping[str, str] | None = None
) -> bool:
    """Log one ERROR line when ingest would fail every document; True when it did."""
    problem = llm_not_configured(settings, environ)
    if problem is not None:
        logger.error("this process will fail every ingested document: %s", problem)
    return problem is not None


def _host(environ: Mapping[str, str]) -> str:
    base = environ.get("OPENAI_API_BASE", "").strip()
    return (urlsplit(base).hostname or "") if base else ""


def _chain_text(error: BaseException) -> str:
    """The lower-cased text of an error and the errors it was raised from, outermost first."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        parts.append(str(current).lower())
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def explain_llm_error(error: BaseException, environ: Mapping[str, str] | None = None) -> str | None:
    """A readable reason for a provider auth or connection failure, or ``None`` for any other.

    Matches LiteLLM's exceptions by class name along the MRO (so LiteLLM need not be imported).
    The text never carries the key or the URL, only the endpoint's host.
    """
    env = os.environ if environ is None else environ
    names = {cls.__name__ for cls in type(error).__mro__}
    text = str(error).lower()
    if names & _AUTH_ERRORS:
        return "The LLM provider rejected the API key (check OPENAI_API_KEY)."
    if "InternalServerError" in names and any(t in text for t in _MISSING_KEY_TEXT):
        return "No LLM API key is configured (set OPENAI_API_KEY or OPENAI_API_BASE)."
    if names & _CONNECTION_ERRORS or (
        "InternalServerError" in names and any(t in _chain_text(error) for t in _UNREACHABLE_TEXT)
    ):
        host = _host(env)
        where = f"at {host} " if host else ""
        return f"The LLM endpoint {where}is unreachable (check OPENAI_API_BASE)."
    return None
