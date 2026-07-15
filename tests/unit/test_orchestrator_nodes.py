"""Unit tests for the orchestrator's graph-node derivation (fast, offline, ``$0``).

Pins the single chokepoint that turns per-document EL links into graph nodes /
the KG-build entity map (:meth:`~graph_rag.orchestrator.Orchestrator._doc_canonical_entities`),
in particular that a ``DATE`` entity is excluded — a date is an edge qualifier
(:attr:`~graph_rag.models.Triple.date`), never a standalone node (ADR-0006).
"""

from __future__ import annotations

from graph_rag.models import EntityLink
from graph_rag.orchestrator import Orchestrator


def _link(surface: str, canonical_id: str, entity_type: str) -> EntityLink:
    return EntityLink(
        mention_text=surface,
        canonical_id=canonical_id,
        entity_type=entity_type,  # type: ignore[arg-type]
        score=1.0,
        is_new=True,
    )


def test_date_entities_are_excluded_from_graph_nodes() -> None:
    """A ``DATE`` link never becomes a graph node (nor a KG-build entity-map entry)."""
    links = [
        _link("Aurelia Components", "e-org", "ORG"),
        _link("2023", "e-date", "DATE"),
        _link("Berlin", "e-loc", "LOCATION"),
    ]

    entities = Orchestrator._doc_canonical_entities(links)  # noqa: SLF001 — chokepoint under test

    ids = {entity.canonical_id for entity in entities}
    assert ids == {"e-org", "e-loc"}  # the DATE entity is filtered out
    assert all(entity.type != "DATE" for entity in entities)


def test_non_date_entities_are_deduplicated_by_canonical_id() -> None:
    """Several links resolving to one canonical yield a single node (unchanged)."""
    links = [
        _link("Aurelia", "e-org", "ORG"),
        _link("Aurelia Components", "e-org", "ORG"),
    ]

    entities = Orchestrator._doc_canonical_entities(links)  # noqa: SLF001 — chokepoint under test

    assert [entity.canonical_id for entity in entities] == ["e-org"]
