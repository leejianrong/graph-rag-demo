"""The closed predicate set for the knowledge graph + raw→closed mapping (V5).

Pins backlog item B7 (ADR-0006, ARCHITECTURE §5c): the KG-build stage's LLM emits
triples whose predicate is a free-text phrase; every such phrase is mapped here to
the closest member of a **closed ~12-predicate set** so the graph stays clean,
queryable and benchmark-consistent. When nothing fits, the edge becomes the open
fallback :attr:`Predicate.RELATED_TO` and the model's original phrase is preserved
as the edge property ``raw_predicate`` — so no relation is ever silently lost.

Both Wave-2 agents (the Neo4j adapter and the KG-build stage) code against this:
the stage calls :func:`map_predicate` on each raw LLM predicate; the resulting
member string becomes :attr:`~graph_rag.models.Triple.predicate` and the returned
``raw_predicate`` becomes :attr:`~graph_rag.models.EdgeProvenance.raw_predicate`.

The mapping is deterministic and case/space/underscore/punctuation-insensitive:
raw phrases are normalized (lower-cased, non-alphanumerics collapsed to single
spaces, stripped) before lookup, so ``"WORKS_FOR"``, ``"works for"`` and
``"Works-For"`` all resolve to :attr:`Predicate.WORKS_FOR`. A small, explicit
synonym table maps common phrasings; anything unmatched falls back to
``RELATED_TO``.
"""

from __future__ import annotations

import re
from enum import StrEnum

__all__ = ["Predicate", "CLOSED_PREDICATES", "map_predicate"]


class Predicate(StrEnum):
    """The closed knowledge-graph predicate set (ADR-0006 / ARCHITECTURE §5c).

    Eleven curated primary relations plus :attr:`RELATED_TO`, the open fallback
    used when a raw predicate maps to none of the primaries. Members are plain
    strings (``StrEnum``), so ``Predicate.WORKS_FOR == "WORKS_FOR"`` and a member
    slots straight into :attr:`~graph_rag.models.Triple.predicate`.
    """

    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    MEMBER_OF = "MEMBER_OF"
    WORKS_FOR = "WORKS_FOR"
    HAS_ROLE = "HAS_ROLE"
    FOUNDED = "FOUNDED"
    OWNS = "OWNS"
    PRODUCES = "PRODUCES"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    OCCURRED_ON = "OCCURRED_ON"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    RELATED_TO = "RELATED_TO"  # open fallback — preserve raw phrase in provenance


# The full closed set as a frozenset of the member strings (order-independent
# membership checks for the adapter/stage).
CLOSED_PREDICATES: frozenset[str] = frozenset(p.value for p in Predicate)

# Any run of characters that are neither word characters nor whitespace — plus
# the underscore, which ``\w`` would otherwise keep — collapses to a space, so
# the match is underscore/punctuation-insensitive.
_SEP_RE = re.compile(r"[^0-9a-z]+")


def _norm(raw: str) -> str:
    """Normalize a raw predicate phrase for deterministic lookup.

    Lower-cases, replaces every non-alphanumeric run (spaces, underscores,
    hyphens, punctuation) with a single space and strips. ``"WORKS_FOR"``,
    ``"works  for"`` and ``"Works-For"`` all normalize to ``"works for"``.
    """
    return _SEP_RE.sub(" ", raw.casefold()).strip()


# Synonym table: normalized phrase -> closed predicate. The canonical normalized
# form of each member (e.g. ``"works for"``) is added automatically below, so
# this table only needs the *extra* phrasings the LLM is likely to emit. Kept
# deliberately small and explicit; extend here as the corpus surfaces new phrasings
# (B7 is "extend before building the corpus").
_SYNONYMS: dict[str, Predicate] = {
    # LOCATED_IN
    "location": Predicate.LOCATED_IN,
    "located": Predicate.LOCATED_IN,
    "based in": Predicate.LOCATED_IN,
    "headquartered in": Predicate.LOCATED_IN,
    "is in": Predicate.LOCATED_IN,
    "situated in": Predicate.LOCATED_IN,
    # PART_OF
    "part of": Predicate.PART_OF,
    "subsidiary of": Predicate.PART_OF,
    "division of": Predicate.PART_OF,
    "belongs to": Predicate.PART_OF,
    # MEMBER_OF
    "member": Predicate.MEMBER_OF,
    "belongs to group": Predicate.MEMBER_OF,
    # WORKS_FOR
    "employed by": Predicate.WORKS_FOR,
    "employee of": Predicate.WORKS_FOR,
    "works at": Predicate.WORKS_FOR,
    "works for": Predicate.WORKS_FOR,
    # HAS_ROLE
    "role": Predicate.HAS_ROLE,
    "has title": Predicate.HAS_ROLE,
    "serves as": Predicate.HAS_ROLE,
    "position": Predicate.HAS_ROLE,
    # FOUNDED
    "founder of": Predicate.FOUNDED,
    "co founded": Predicate.FOUNDED,
    "established": Predicate.FOUNDED,
    "created": Predicate.FOUNDED,
    # OWNS
    "owner of": Predicate.OWNS,
    "acquired": Predicate.OWNS,
    "controls": Predicate.OWNS,
    # PRODUCES
    "makes": Predicate.PRODUCES,
    "manufactures": Predicate.PRODUCES,
    "develops": Predicate.PRODUCES,
    "produced": Predicate.PRODUCES,
    # PARTICIPATED_IN
    "took part in": Predicate.PARTICIPATED_IN,
    "attended": Predicate.PARTICIPATED_IN,
    "involved in": Predicate.PARTICIPATED_IN,
    "participant in": Predicate.PARTICIPATED_IN,
    # OCCURRED_ON
    "happened on": Predicate.OCCURRED_ON,
    "took place on": Predicate.OCCURRED_ON,
    "dated": Predicate.OCCURRED_ON,
    "on date": Predicate.OCCURRED_ON,
    # AFFILIATED_WITH
    "associated with": Predicate.AFFILIATED_WITH,
    "affiliation": Predicate.AFFILIATED_WITH,
    "allied with": Predicate.AFFILIATED_WITH,
    # RELATED_TO (explicit — a clean map to the fallback member)
    "relates to": Predicate.RELATED_TO,
    "connected to": Predicate.RELATED_TO,
    "linked to": Predicate.RELATED_TO,
}

# Full lookup: the canonical normalized form of every member (``"works for"`` for
# ``WORKS_FOR``) plus the synonym phrasings. Built once at import.
_LOOKUP: dict[str, Predicate] = {_norm(p.value): p for p in Predicate}
_LOOKUP.update(_SYNONYMS)


def map_predicate(raw: str) -> tuple[str, str | None]:
    """Map a raw LLM predicate phrase to a closed-set member (ADR-0006, B7).

    Deterministic and case/space/underscore/punctuation-insensitive: ``raw`` is
    normalized then looked up against the closed members and a small synonym
    table.

    Args:
        raw: The free-text predicate the KG-build LLM emitted (e.g.
            ``"is employed by"``, ``"WORKS_FOR"``, ``"acquired"``).

    Returns:
        ``(predicate, raw_predicate)`` where ``predicate`` is a member of
        :class:`Predicate` (as its string value):

        * **Clean map** — ``raw`` matches a primary member or a known synonym:
          returns ``(<PREDICATE>, None)``. The mapped member fully captures the
          relation, so no raw phrase needs preserving (an explicit ``"related
          to"`` therefore returns ``("RELATED_TO", None)``).
        * **No match** — returns ``("RELATED_TO", raw)``: the edge uses the open
          fallback and the caller stores the original ``raw`` phrase in the edge's
          ``raw_predicate`` provenance so nothing is lost.
    """
    match = _LOOKUP.get(_norm(raw))
    if match is not None:
        return (match.value, None)
    return (Predicate.RELATED_TO.value, raw)
