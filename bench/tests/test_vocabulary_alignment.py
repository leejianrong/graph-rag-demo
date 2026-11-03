"""kggen's vocabulary must stay expressible in graph-rag's schema.

kggen is standalone and defines its own entity types and relations. The bench is where the
two meet, so the check lives here: every type and relation kggen can emit must exist in
graph-rag, or scoring would compare against labels the pipeline can never produce.
"""

from __future__ import annotations

from typing import get_args

from kggen.spec import ENTITY_TYPES, PREDICATES, VAGUE_PREDICATE

from graph_rag.models import CuratedType
from graph_rag.predicates import CLOSED_PREDICATES, Predicate


def test_every_kggen_entity_type_is_a_graph_rag_type() -> None:
    assert set(ENTITY_TYPES) <= set(get_args(CuratedType))


def test_every_kggen_relation_is_in_the_closed_predicate_set() -> None:
    assert set(PREDICATES) <= {str(p) for p in CLOSED_PREDICATES}


def test_the_vague_relation_is_graph_rags_fallback() -> None:
    assert VAGUE_PREDICATE == str(Predicate.RELATED_TO)
