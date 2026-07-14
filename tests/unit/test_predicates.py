"""Unit tests for the closed predicate set + raw→closed mapping (V5, ADR-0006).

Exercises :func:`graph_rag.predicates.map_predicate` — clean maps to closed-set
members, the ``RELATED_TO`` open fallback preserving the raw phrase, and the
case/space/underscore-insensitivity + determinism the KG-build stage relies on.
Fast, offline, ``$0``.
"""

from __future__ import annotations

from graph_rag.predicates import CLOSED_PREDICATES, Predicate, map_predicate


def test_closed_set_is_the_expected_twelve() -> None:
    """The closed set is exactly the ~12 members from ADR-0006 / ARCHITECTURE §5c."""
    assert CLOSED_PREDICATES == {
        "LOCATED_IN",
        "PART_OF",
        "MEMBER_OF",
        "WORKS_FOR",
        "HAS_ROLE",
        "FOUNDED",
        "OWNS",
        "PRODUCES",
        "PARTICIPATED_IN",
        "OCCURRED_ON",
        "AFFILIATED_WITH",
        "RELATED_TO",
    }


def test_exact_member_maps_cleanly_with_no_raw() -> None:
    """A raw string equal to a member maps to it with ``raw_predicate`` None."""
    for member in Predicate:
        predicate, raw = map_predicate(member.value)
        assert predicate == member.value
        assert raw is None


def test_synonym_maps_to_primary_member() -> None:
    """Common phrasings resolve to the intended primary predicate, dropping raw."""
    assert map_predicate("employed by") == ("WORKS_FOR", None)
    assert map_predicate("headquartered in") == ("LOCATED_IN", None)
    assert map_predicate("acquired") == ("OWNS", None)
    assert map_predicate("subsidiary of") == ("PART_OF", None)
    assert map_predicate("took part in") == ("PARTICIPATED_IN", None)


def test_unknown_predicate_falls_back_to_related_to_preserving_raw() -> None:
    """No match → RELATED_TO + the original phrase kept for edge provenance."""
    predicate, raw = map_predicate("has a crush on")
    assert predicate == "RELATED_TO"
    assert raw == "has a crush on"


def test_explicit_related_to_is_a_clean_map_not_a_fallback() -> None:
    """An explicit 'related to' cleanly maps to RELATED_TO with no raw phrase."""
    assert map_predicate("related to") == ("RELATED_TO", None)
    assert map_predicate("connected to") == ("RELATED_TO", None)


def test_case_space_underscore_punctuation_insensitive() -> None:
    """WORKS_FOR / works for / Works-For / 'WORKS   FOR' all map identically."""
    variants = ["WORKS_FOR", "works for", "Works-For", "WORKS   FOR", " works_for "]
    results = {map_predicate(v) for v in variants}
    assert results == {("WORKS_FOR", None)}


def test_deterministic_across_repeated_calls() -> None:
    """Repeated calls on the same input return identical results (deterministic)."""
    for phrase in ["employed by", "unmapped phrase", "PART_OF"]:
        assert map_predicate(phrase) == map_predicate(phrase)


def test_predicate_members_are_strings() -> None:
    """``Predicate`` is a StrEnum so a member slots straight into ``Triple.predicate``."""
    assert Predicate.WORKS_FOR == "WORKS_FOR"
    assert isinstance(Predicate.WORKS_FOR, str)
