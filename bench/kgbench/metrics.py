"""Pure, non-LLM scoring core for the V8 benchmark (ADR-0009, R8).

Every function here is a **pure function of simple types** — no I/O, no pipeline
dependencies — so the scoring guarantees (R8) are trivially unit-testable and the
benchmark scores are deterministic and reproducible. Two families of metric:

* **Answer quality** (:func:`exact_match`, :func:`token_f1`) — the standard
  SQuAD / HotpotQA / 2WikiMultihopQA string metrics, under the standard answer
  :func:`normalize_answer` (lowercase; strip punctuation; strip the articles
  ``a``/``an``/``the``; collapse whitespace). A prediction is scored against a
  LIST of acceptable gold surfaces so a differently-phrased-but-correct entity
  (matching an alias) is not unfairly scored wrong (ADR-0009).
* **Supporting-fact retrieval** (:func:`supporting_fact_prf`) — precision /
  recall / F1 over supporting-fact identifiers (e.g. ``(title, sentence_index)``
  pairs): did retrieval surface the gold evidence sentences?

:func:`aggregate` averages a list of per-question metric dicts over a run.

The scoring is **non-LLM throughout** — plain string/set comparison against gold
(ADR-0009, Q32): the whole point is a benchmark that re-runs at ~$0.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence

__all__ = [
    "normalize_answer",
    "exact_match",
    "token_f1",
    "supporting_fact_prf",
    "aggregate",
]

# Whole-word articles stripped by the standard normalization (SQuAD/2Wiki).
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Return the STANDARD normalized form of an answer string (SQuAD / 2Wiki).

    The canonical normalization used by SQuAD, HotpotQA and 2WikiMultihopQA
    answer scoring, applied in the canonical order: lowercase → strip
    punctuation → strip the articles ``a``/``an``/``the`` → collapse whitespace.
    So ``"The United States."`` and ``"united states"`` both normalize to
    ``"united states"``.

    Args:
        s: A raw answer / surface string.

    Returns:
        The normalized comparison key (may be ``""`` for empty/all-punctuation
        input).
    """
    lowered = s.lower()
    depunct = lowered.translate(_PUNCT_TABLE)
    dearticled = _ARTICLES_RE.sub(" ", depunct)
    return _WHITESPACE_RE.sub(" ", dearticled).strip()


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """Return ``1.0`` if ``prediction`` normalizes-equal to ANY gold, else ``0.0``.

    Standard exact-match under :func:`normalize_answer`. ``golds`` is the list of
    acceptable gold surfaces (e.g. the answer's ``name`` plus each alias), so a
    differently-phrased-but-correct answer that matches an alias scores as correct
    (ADR-0009).

    Args:
        prediction: The predicted answer string.
        golds: Acceptable gold surfaces (answer name + aliases).

    Returns:
        ``1.0`` on a match against any gold, ``0.0`` otherwise (``0.0`` if
        ``golds`` is empty).
    """
    normalized_prediction = normalize_answer(prediction)
    return 1.0 if any(normalized_prediction == normalize_answer(g) for g in golds) else 0.0


def _token_f1_single(prediction: str, gold: str) -> float:
    """SQuAD token-overlap F1 of one prediction against one gold string.

    Follows the SQuAD convention for the degenerate cases: if either side has no
    tokens after normalization, the score is ``1.0`` only when both are empty
    (an exact empty match), else ``0.0``.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        # SQuAD: only credit when both are empty (they are exactly equal).
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Return the MAX SQuAD token-F1 of ``prediction`` over all ``golds``.

    Token-overlap F1 (the standard SQuAD partial-credit metric): the harmonic
    mean of token precision and recall over the normalized token multisets, taken
    as the best score across the acceptable gold surfaces.

    Args:
        prediction: The predicted answer string.
        golds: Acceptable gold surfaces (answer name + aliases).

    Returns:
        The best token-F1 in ``[0.0, 1.0]`` (``0.0`` if ``golds`` is empty).
    """
    return max((_token_f1_single(prediction, g) for g in golds), default=0.0)


def supporting_fact_prf(
    predicted: Iterable[Hashable], gold: Iterable[Hashable]
) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` over supporting-fact identifiers.

    Set-based P/R/F1 over supporting-fact identifiers — e.g. ``(title,
    sentence_index)`` pairs: the retrieved supporting sentences (``predicted``)
    versus the gold supporting facts (``gold``). Duplicates are ignored (compared
    as sets). Matches the HotpotQA supporting-fact convention: an empty side
    yields ``0.0`` for the affected ratio (no division by zero).

    Args:
        predicted: The retrieved supporting-fact identifiers.
        gold: The gold supporting-fact identifiers.

    Returns:
        ``(precision, recall, f1)``, each in ``[0.0, 1.0]``.
    """
    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positives = len(predicted_set & gold_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average a list of per-question metric dicts into one run-level dict.

    Every row is expected to carry the same metric keys; the union of keys is
    averaged, treating a missing key on a row as ``0.0`` so a ragged input still
    aggregates sensibly.

    Args:
        rows: The per-question metric dicts.

    Returns:
        A dict mapping each metric name to its mean over ``rows`` (empty dict for
        no rows).
    """
    if not rows:
        return {}
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}
