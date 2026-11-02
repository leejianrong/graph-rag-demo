"""The declarative ``WorldSpec`` that controls how complex a generated world is (ADR-0011).

A spec plus its seed fully determines a world. Knobs fall into groups: graph size and
mix (entity count, type mix, predicate mix), graph shape (average degree, degree skew,
communities, cycles). Ambiguity and temporal churn knobs arrive in a later slice.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph_rag.models import CuratedType
from graph_rag.predicates import Predicate

# DATE is an edge qualifier in graph-rag, never a node, so the generator never places it.
ENTITY_TYPES: tuple[str, ...] = tuple(t for t in get_args(CuratedType) if t != "DATE")

# Which (subject type, object type) pairs each predicate may connect. Only predicates
# with an unambiguous typing are generated: OCCURRED_ON (a date qualifier), HAS_ROLE and
# the RELATED_TO fallback are left out.
PREDICATE_SIGNATURES: dict[str, tuple[tuple[str, str], ...]] = {
    Predicate.LOCATED_IN: (
        ("ORG", "LOCATION"),
        ("EVENT", "LOCATION"),
        ("PERSON", "LOCATION"),
        ("LOCATION", "LOCATION"),
    ),
    Predicate.PART_OF: (("ORG", "ORG"), ("LOCATION", "LOCATION"), ("PRODUCT", "PRODUCT")),
    Predicate.MEMBER_OF: (("PERSON", "ORG"), ("PERSON", "NORP"), ("ORG", "ORG")),
    Predicate.WORKS_FOR: (("PERSON", "ORG"),),
    Predicate.FOUNDED: (("PERSON", "ORG"),),
    Predicate.OWNS: (("ORG", "ORG"), ("ORG", "PRODUCT"), ("PERSON", "PRODUCT")),
    Predicate.PRODUCES: (("ORG", "PRODUCT"),),
    Predicate.PARTICIPATED_IN: (("PERSON", "EVENT"), ("ORG", "EVENT")),
    Predicate.AFFILIATED_WITH: (("PERSON", "ORG"), ("ORG", "ORG"), ("PERSON", "NORP")),
}
PREDICATES: tuple[str, ...] = tuple(str(p) for p in PREDICATE_SIGNATURES)

_DEFAULT_TYPE_MIX = {
    "PERSON": 0.35,
    "ORG": 0.25,
    "LOCATION": 0.15,
    "EVENT": 0.10,
    "PRODUCT": 0.10,
    "NORP": 0.05,
}
_DEFAULT_PREDICATE_MIX = {p: 1.0 for p in PREDICATES}


def _check_weights(name: str, weights: dict[str, float], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(weights) - set(allowed))
    if unknown:
        raise ValueError(f"{name} has unknown keys {unknown}; allowed: {list(allowed)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{name} weights must be >= 0")
    if sum(weights.values()) <= 0:
        raise ValueError(f"{name} needs at least one positive weight")


class WorldSpec(BaseModel):
    """Every knob that shapes a generated world. Same spec and seed give the same world."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    n_entities: int = Field(ge=2, description="Number of entities (nodes).")
    type_mix: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_TYPE_MIX),
        description="Relative weight of each entity type.",
    )
    predicate_mix: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_PREDICATE_MIX),
        description="Relative weight of each relation (relations without their types are skipped).",
    )
    avg_degree: float = Field(
        gt=0, description="Average total degree; edges = n_entities * avg_degree / 2."
    )
    degree_skew: float = Field(
        default=0.0,
        ge=0,
        description="0 = uniform degrees; larger = a few hub entities (Zipf exponent).",
    )
    n_communities: int = Field(default=1, ge=1, description="Number of entity communities.")
    community_mixing: float = Field(
        default=0.1, ge=0, le=1, description="Chance an edge joins two different communities."
    )
    cycle_rate: float = Field(
        default=0.0, ge=0, le=1, description="Share of edges that close a 3-cycle."
    )

    @model_validator(mode="after")
    def _check(self) -> WorldSpec:
        _check_weights("type_mix", self.type_mix, ENTITY_TYPES)
        _check_weights("predicate_mix", self.predicate_mix, PREDICATES)
        if self.n_communities > self.n_entities:
            raise ValueError("n_communities cannot exceed n_entities")
        pairs = self.feasible_pairs()
        if not pairs:
            raise ValueError("no relation can connect the entity types in type_mix")
        used = {t for ps in pairs.values() for pair in ps for t in pair}
        isolated = sorted(t for t, w in self.type_mix.items() if w > 0 and t not in used)
        if isolated:
            raise ValueError(
                f"entity types {isolated} cannot take part in any selected relation "
                "(they would be isolated)"
            )
        return self

    def feasible_pairs(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Relations with weight > 0 mapped to the type pairs that exist in ``type_mix``."""
        present = {t for t, w in self.type_mix.items() if w > 0}
        out: dict[str, tuple[tuple[str, str], ...]] = {}
        for pred, weight in self.predicate_mix.items():
            if weight <= 0:
                continue
            pairs = tuple(
                (s, d)
                for s, d in PREDICATE_SIGNATURES[Predicate(pred)]
                if s in present and d in present
            )
            if pairs:
                out[pred] = pairs
        return out

    def with_overrides(self, **changes: Any) -> WorldSpec:
        """A copy with some knobs changed (re-validated)."""
        return WorldSpec.model_validate({**self.model_dump(), **changes})


PRESET_NAMES: tuple[str, ...] = ("tiny", "small", "medium", "large", "stress")


def load_spec(path: str | Path) -> WorldSpec:
    """Load and validate a spec from a YAML file."""
    return WorldSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def load_preset(name: str) -> WorldSpec:
    """Load one of the named presets shipped with kgbench."""
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESET_NAMES)}")
    text = (resources.files("kgbench") / "presets" / f"{name}.yaml").read_text()
    return WorldSpec.model_validate(yaml.safe_load(text))
