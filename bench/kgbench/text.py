"""Plain-text matching shared by the answer scorers: sentences, whole-word names, past markers."""

from __future__ import annotations

import re
from collections.abc import Sequence

SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
PAST = re.compile(
    r"\b(formerly|previously|no longer|used to|until|was|were|former|withdrawn|retracted|"
    r"replaced|instead of|rather than|not|never|from)\b",
    re.IGNORECASE,
)


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in SENTENCES.split(text) if s.strip()]


def mentions(sentence: str, names: Sequence[str]) -> bool:
    """Whether any of ``names`` appears in ``sentence`` as a whole word, in any case."""
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", sentence, re.IGNORECASE) for n in names)
