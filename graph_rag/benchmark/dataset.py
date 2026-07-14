"""2WikiMultihopQA dataset loader + fixed named subsets (B8, ADR-0009).

Loads examples in the **2WikiMultihopQA JSON shape** — each example carries a
``question``, an ``answer`` (+ optional ``answer_aliases``), a ``context`` (a
list of ``[title, [sentence, ...]]`` paragraphs) and ``supporting_facts`` (a list
of ``[title, sentence_index]``) — into typed :class:`BenchmarkExample` records.

The real corpus lives **outside git** (``datasets/`` is gitignored, B8): the
loader takes a file path (a JSON array, or newline-delimited JSON) or a directory
(the first ``*.json``/``*.jsonl`` in it). A TINY hand-authored fixture ships in
``tests/fixtures/wiki2_mini.json`` so the fast suite + CLI run with NO download.

**Named subsets are FIXED and deterministic (B8, reproducibility — ADR-0009):**
:func:`select_subset` sorts the examples by ``id`` and takes a fixed prefix, so
the same subset is selected on every run regardless of file order. ``small`` is a
demo-sized slice; the real benchmark targets ~100–200 questions (``medium``).
``--limit`` narrows further (still deterministic, applied after the sort).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Paragraph",
    "BenchmarkExample",
    "SUBSETS",
    "load_examples",
    "select_subset",
]

# Fixed subset sizes (B8). Selection is a deterministic sorted-by-id prefix, so
# each named subset is stable across runs. ``full`` means "every example". The
# real 2Wiki benchmark uses ``medium`` (~100–200 questions, ADR-0009); ``small``
# is the fast demo slice (the mini fixture has fewer than 20, so ``small`` takes
# all of it — still deterministic).
SUBSETS: dict[str, int | None] = {
    "small": 20,
    "medium": 200,
    "full": None,
}


@dataclass(frozen=True)
class Paragraph:
    """One context paragraph — a titled list of sentences (2Wiki ``context`` item).

    ``title`` names the source article (the supporting-fact key); ``sentences``
    are its ordered sentences, indexed by their position (the supporting-fact
    ``sentence_index``).
    """

    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkExample:
    """One 2WikiMultihopQA example — question, gold answer, context, supporting facts.

    ``answer_aliases`` are extra acceptable gold surfaces (so a correct but
    differently-phrased answer is credited, ADR-0009). ``supporting_facts`` are
    ``(title, sentence_index)`` identifiers into ``context`` — the gold evidence
    the retrieval P/R/F1 is scored against.
    """

    id: str
    question: str
    answer: str
    context: tuple[Paragraph, ...]
    supporting_facts: tuple[tuple[str, int], ...]
    answer_aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gold_answers(self) -> list[str]:
        """The acceptable gold answer surfaces: the answer plus every alias."""
        return [self.answer, *self.answer_aliases]


def _parse_example(raw: dict) -> BenchmarkExample:
    """Parse one raw 2Wiki JSON object into a :class:`BenchmarkExample`.

    Tolerant of the id key (``_id`` or ``id``) and of ``supporting_facts`` /
    ``context`` entries given as lists or tuples, matching the published 2Wiki
    shape.
    """
    example_id = str(raw.get("_id") or raw.get("id") or "")
    context = tuple(
        Paragraph(title=str(title), sentences=tuple(str(s) for s in sentences))
        for title, sentences in raw.get("context", [])
    )
    supporting_facts = tuple(
        (str(title), int(sentence_index))
        for title, sentence_index in raw.get("supporting_facts", [])
    )
    aliases = tuple(str(a) for a in raw.get("answer_aliases", []) or [])
    return BenchmarkExample(
        id=example_id,
        question=str(raw["question"]),
        answer=str(raw.get("answer", "")),
        context=context,
        supporting_facts=supporting_facts,
        answer_aliases=aliases,
    )


def _read_raw(path: Path) -> list[dict]:
    """Read a JSON array or newline-delimited JSON file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return list(json.loads(text))
    # Newline-delimited JSON (one example per line).
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_examples(path: str | Path) -> list[BenchmarkExample]:
    """Load 2WikiMultihopQA examples from a file or directory.

    Args:
        path: A ``.json`` (array) / ``.jsonl`` file, or a directory whose first
            ``*.json`` / ``*.jsonl`` file (sorted by name) is read. The real
            dataset lives outside git (``datasets/``, B8); the shipped fixture is
            ``tests/fixtures/wiki2_mini.json``.

    Returns:
        The parsed examples in FILE order (call :func:`select_subset` to pick a
        deterministic, sorted subset).

    Raises:
        FileNotFoundError: If ``path`` (or a dataset file within it) is absent.
    """
    resolved = Path(path)
    if resolved.is_dir():
        candidates = sorted(
            [*resolved.glob("*.json"), *resolved.glob("*.jsonl")], key=lambda p: p.name
        )
        if not candidates:
            raise FileNotFoundError(f"no *.json / *.jsonl dataset file in directory {resolved}")
        resolved = candidates[0]
    if not resolved.is_file():
        raise FileNotFoundError(f"dataset file not found: {resolved}")
    return [_parse_example(raw) for raw in _read_raw(resolved)]


def select_subset(
    examples: list[BenchmarkExample],
    subset: str = "small",
    *,
    limit: int | None = None,
) -> list[BenchmarkExample]:
    """Pick a FIXED, deterministic subset of ``examples`` (B8, reproducibility).

    Examples are sorted by ``id`` (stable, order-independent of the source file)
    and the named ``subset`` prefix is taken; ``limit`` narrows further. This is
    the reproducible selection the benchmark pins so scores are comparable across
    runs (ADR-0009).

    Args:
        examples: The loaded examples (any order).
        subset: A key of :data:`SUBSETS` (``small`` / ``medium`` / ``full``).
        limit: Optional hard cap applied after the subset prefix.

    Returns:
        The selected examples, sorted by ``id``.

    Raises:
        KeyError: If ``subset`` is not a known subset name.
    """
    if subset not in SUBSETS:
        raise KeyError(f"unknown subset {subset!r}; choose from {sorted(SUBSETS)}")
    ordered = sorted(examples, key=lambda e: e.id)
    size = SUBSETS[subset]
    if size is not None:
        ordered = ordered[:size]
    if limit is not None:
        ordered = ordered[:limit]
    return ordered
