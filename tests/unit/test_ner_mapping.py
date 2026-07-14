"""Unit test for the spaCy-label -> curated-type mapping (TESTING §2, no model).

Pure function under test (:func:`graph_rag.stages.ner.curated_type`): asserts the
curated narrowing from ADR-0002, in particular that ``GPE`` and ``LOC`` both merge
to ``LOCATION`` and that out-of-set labels are dropped. No spaCy model is loaded.
"""

from __future__ import annotations

import pytest

from graph_rag.stages.ner import curated_type


@pytest.mark.parametrize(
    ("spacy_label", "expected"),
    [
        ("PERSON", "PERSON"),
        ("ORG", "ORG"),
        ("GPE", "LOCATION"),  # geo-political entity -> LOCATION
        ("LOC", "LOCATION"),  # non-GPE location    -> LOCATION (merge, ADR-0002)
        ("DATE", "DATE"),
        ("EVENT", "EVENT"),
        ("NORP", "NORP"),
        ("PRODUCT", "PRODUCT"),
    ],
)
def test_curated_labels_map_to_curated_types(spacy_label: str, expected: str) -> None:
    """Each in-set spaCy label maps to its curated type; GPE+LOC both -> LOCATION."""
    assert curated_type(spacy_label) == expected


def test_gpe_and_loc_merge_to_same_location_type() -> None:
    """The GPE/LOC merge is a single target, not two distinct types."""
    assert curated_type("GPE") == curated_type("LOC") == "LOCATION"


@pytest.mark.parametrize(
    "spacy_label",
    ["CARDINAL", "MONEY", "PERCENT", "ORDINAL", "TIME", "QUANTITY", "LANGUAGE", "WORK_OF_ART", ""],
)
def test_out_of_set_labels_are_dropped(spacy_label: str) -> None:
    """Labels outside the curated set map to None (the mention is dropped)."""
    assert curated_type(spacy_label) is None
