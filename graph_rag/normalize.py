"""Shared name normalization for entity linking (V4, ADR-0004).

Entity linking blocks candidates by *type + normalized name* (ADR-0004). The
normalization used to build that blocking key MUST be identical everywhere it is
computed — the fast-suite fake (:class:`~graph_rag.fakes.InMemoryEntityStore`),
the real Elasticsearch ``EntityStore`` adapter, and the EL stage that queries
them. This module is the single source of truth for that rule so those three
never drift.

The rule (deliberately simple + deterministic): case-fold, replace every
non-alphanumeric character with a space, collapse runs of whitespace, and strip.
So ``"Apple, Inc."``, ``"apple inc"`` and ``"APPLE   INC"`` all normalize to the
same key ``"apple inc"``.
"""

from __future__ import annotations

import re

__all__ = ["normalize_name"]

# Any run of characters that are neither word characters (letters/digits/``_``)
# nor whitespace — i.e. punctuation — is replaced with a single space.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
# Any run of whitespace collapses to a single space.
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_name(name: str) -> str:
    """Return the normalized blocking key for an entity surface form.

    Case-folds ``name``, turns punctuation into spaces, collapses whitespace and
    strips. Deterministic and pure, so it is safe to use as a stable blocking
    key across ingestion runs (ADR-0004).

    Args:
        name: A raw entity name / surface form (e.g. ``"Apple, Inc."``).

    Returns:
        The normalized key (e.g. ``"apple inc"``); ``""`` for an all-punctuation
        or empty input.
    """
    lowered = name.casefold()
    despunct = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", despunct)
    return collapsed.strip()
