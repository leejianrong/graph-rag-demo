"""Unit tests for the pure benchmark scoring core (R8 scoring guarantees, ADR-0009).

Fast, offline, no fixtures: the metric functions are pure functions of simple
types. Covers the standard answer normalization, exact match (including an alias
match and a differently-phrased-but-correct entity), token-F1 partial overlap, and
supporting-fact P/R/F1 over crafted predicted-vs-gold sets (empty / perfect /
partial). No LLM is involved — scoring is string/set comparison (ADR-0009, Q32).
"""

from __future__ import annotations

import pytest

from graph_rag.benchmark.metrics import (
    aggregate,
    exact_match,
    normalize_answer,
    supporting_fact_prf,
    token_f1,
)


class TestNormalizeAnswer:
    """The standard SQuAD/2Wiki normalization (lowercase, strip articles/punct/ws)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The United States", "united states"),
            ("united states", "united states"),
            ("A. B. C.", "b c"),  # 'a' is an article and is stripped
            ("  Ada   Lovelace  ", "ada lovelace"),
            ("The Analytical Engine!", "analytical engine"),
            ("", ""),
            (".,;:!?", ""),
        ],
    )
    def test_normalization_is_standard(self, raw: str, expected: str) -> None:
        assert normalize_answer(raw) == expected

    def test_articles_stripped_only_as_whole_words(self) -> None:
        # 'a'/'an'/'the' inside a token must NOT be stripped.
        assert normalize_answer("Thebes Panama") == "thebes panama"


class TestExactMatch:
    """Exact match under normalization, over a list of acceptable golds."""

    def test_direct_match(self) -> None:
        assert exact_match("Ada Lovelace", ["Ada Lovelace"]) == 1.0

    def test_differently_phrased_but_correct_scores_correct(self) -> None:
        # Articles + casing + punctuation differ but the answer is the same entity.
        assert exact_match("the United States.", ["United States"]) == 1.0

    def test_alias_match_scores_correct(self) -> None:
        # The prediction matches an ALIAS, not the primary name — still correct.
        golds = ["Ada Lovelace", "Augusta Ada King", "Countess of Lovelace"]
        assert exact_match("Countess of Lovelace", golds) == 1.0

    def test_wrong_answer_scores_zero(self) -> None:
        assert exact_match("Charles Babbage", ["Ada Lovelace"]) == 0.0

    def test_empty_golds_scores_zero(self) -> None:
        assert exact_match("anything", []) == 0.0


class TestTokenF1:
    """SQuAD token-overlap F1, taking the best over the golds."""

    def test_perfect_overlap(self) -> None:
        assert token_f1("Ada Lovelace", ["Ada Lovelace"]) == 1.0

    def test_partial_overlap(self) -> None:
        # pred={analytical, engine}, gold={the, analytical, engine} -> 'the' is an
        # article, stripped by normalization -> gold={analytical, engine} -> F1=1.0.
        assert token_f1("Analytical Engine", ["The Analytical Engine"]) == 1.0

    def test_genuine_partial_overlap(self) -> None:
        # pred={charles, babbage, jr}, gold={charles, babbage}: 2 common.
        # precision 2/3, recall 2/2 -> F1 = 2*(2/3)/(2/3+1) = 0.8.
        assert token_f1("Charles Babbage Jr", ["Charles Babbage"]) == pytest.approx(0.8)

    def test_no_overlap(self) -> None:
        assert token_f1("London", ["Ada Lovelace"]) == 0.0

    def test_best_over_multiple_golds(self) -> None:
        # Matches the second gold exactly -> 1.0 even though the first is disjoint.
        assert token_f1("Ada Lovelace", ["Charles Babbage", "Ada Lovelace"]) == 1.0


class TestSupportingFactPrf:
    """Supporting-fact precision/recall/F1 over identifier sets."""

    def test_perfect(self) -> None:
        gold = {("Doc", 0), ("Doc", 1)}
        precision, recall, f1 = supporting_fact_prf(gold, gold)
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)

    def test_empty_prediction(self) -> None:
        precision, recall, f1 = supporting_fact_prf(set(), {("Doc", 0)})
        assert (precision, recall, f1) == (0.0, 0.0, 0.0)

    def test_partial(self) -> None:
        # predicted 2 (1 correct, 1 spurious); gold 2 (1 missed).
        predicted = [("Doc", 0), ("Doc", 9)]
        gold = [("Doc", 0), ("Doc", 1)]
        precision, recall, f1 = supporting_fact_prf(predicted, gold)
        assert precision == pytest.approx(0.5)  # 1 of 2 predicted is correct
        assert recall == pytest.approx(0.5)  # 1 of 2 gold retrieved
        assert f1 == pytest.approx(0.5)

    def test_duplicates_ignored(self) -> None:
        # A duplicate retrieved fact does not inflate precision (set semantics).
        precision, recall, f1 = supporting_fact_prf([("Doc", 0), ("Doc", 0)], [("Doc", 0)])
        assert (precision, recall, f1) == (1.0, 1.0, 1.0)


class TestAggregate:
    """Averaging per-question metric dicts over a run."""

    def test_averages_each_key(self) -> None:
        rows = [
            {"exact_match": 1.0, "token_f1": 1.0},
            {"exact_match": 0.0, "token_f1": 0.5},
        ]
        assert aggregate(rows) == {"exact_match": 0.5, "token_f1": 0.75}

    def test_empty_rows(self) -> None:
        assert aggregate([]) == {}
