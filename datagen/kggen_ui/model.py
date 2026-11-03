"""The UI's logic, with no indah import: knob definitions, building a spec, summarising a world."""

from __future__ import annotations

from dataclasses import dataclass

from kggen.spec import PRESET_NAMES, WorldSpec, load_preset
from kggen.stats import check_against_spec
from kggen.world import Edge, Entity, World, WorldGenerationError, generate
from pydantic import ValidationError

MAX_ENTITIES = 20_000  # what the graph view can hold (ADR-0048); larger worlds go through the CLI


@dataclass(frozen=True)
class Knob:
    name: str
    label: str
    min: float
    max: float
    step: float
    integer: bool = False
    group: str = "shape"  # which card of the form it sits in


GROUPS = {"shape": "Graph shape", "ambiguity": "Ambiguity", "churn": "Change over time"}
KNOBS: tuple[Knob, ...] = (
    Knob("n_entities", "Entities", 2, MAX_ENTITIES, 10, integer=True),
    Knob("avg_degree", "Average degree", 0.5, 20, 0.5),
    Knob("degree_skew", "Degree skew (hubs)", 0, 3, 0.1),
    Knob("n_communities", "Communities", 1, 100, 1, integer=True),
    Knob("community_mixing", "Community mixing", 0, 1, 0.05),
    Knob("cycle_rate", "Cycle rate", 0, 1, 0.01),
    Knob("seed", "Seed", 0, 1_000_000, 1, integer=True),
    Knob("alias_rate", "Entities with aliases", 0, 1, 0.05, group="ambiguity"),
    Knob("homonym_rate", "Homonyms (shared names)", 0, 1, 0.01, group="ambiguity"),
    Knob("distractor_rate", "Distractor entities", 0, 1, 0.01, group="ambiguity"),
    Knob("noise_rate", "Vague relations", 0, 1, 0.05, group="ambiguity"),
    Knob("churn_rate", "Facts replaced later", 0, 1, 0.05, group="churn"),
    Knob("retraction_rate", "Facts retracted later", 0, 1, 0.05, group="churn"),
)
KNOB_NAMES = tuple(k.name for k in KNOBS)
DEFAULT_PRESET = "small"


def values_from_spec(spec: WorldSpec) -> dict[str, float]:
    """The knob values a spec carries, for filling the form."""
    return {name: getattr(spec, name) for name in KNOB_NAMES}


def preset_values(name: str) -> dict[str, float]:
    return values_from_spec(load_preset(name))


def spec_from_values(values: dict[str, float], base: WorldSpec | None = None) -> WorldSpec:
    """Build a validated spec from form values; a ``ValueError`` explains what is wrong."""
    base = base or load_preset(DEFAULT_PRESET)
    clean: dict[str, float] = {}
    for knob in KNOBS:
        raw = values.get(knob.name)
        if raw is None or raw == "":
            raise ValueError(f"{knob.label} needs a value")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{knob.label} must be a number") from None
        if not knob.min <= number <= knob.max:
            raise ValueError(f"{knob.label} must be between {knob.min:g} and {knob.max:g}")
        clean[knob.name] = int(round(number)) if knob.integer else number
    try:
        return base.with_overrides(**clean)
    except ValidationError as exc:
        raise ValueError(exc.errors()[0]["msg"].removeprefix("Value error, ")) from None


def generate_world(values: dict[str, float], base: WorldSpec | None = None) -> World:
    """Generate the world the form describes. Same values and seed give the same world."""
    try:
        return generate(spec_from_values(values, base))
    except WorldGenerationError as exc:
        raise ValueError(str(exc)) from None


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"


def check_rows(world: World) -> list[list[str]]:
    """Each knob's target and realized value with a pass mark, for the stats panel."""
    return [
        [c.knob, _fmt(c.target), _fmt(c.realized), "ok" if c.ok else "off target"]
        for c in check_against_spec(world)
    ]


# What the graph view can hold (the indah Graph component's ceiling, ADR-0048).
MAX_GRAPH_NODES = 20_000
MAX_GRAPH_EDGES = 60_000
DISTRACTOR_GROUP = "Distractor"


def graph_limits_message(world: World) -> str | None:
    """Why this world is too big to draw, or ``None`` when it fits the graph view."""
    edges = len(world.current_edges())
    if len(world.entities) > MAX_GRAPH_NODES or edges > MAX_GRAPH_EDGES:
        return (
            f"This world has {len(world.entities):,} entities and {edges:,} facts; the graph view "
            f"draws up to {MAX_GRAPH_NODES:,} and {MAX_GRAPH_EDGES:,}. Lower the entities or the "
            "average degree to see it."
        )
    return None


def graph_data(world: World) -> tuple[list[dict], list[dict]]:
    """The world as the Graph component wants it: nodes coloured by type (distractors apart) and
    the facts still true at the end, each edge carrying its fact id."""
    nodes = [
        {"id": e.id, "label": e.name, "group": DISTRACTOR_GROUP if e.distractor else e.type}
        for e in world.entities
    ]
    edges = [{"source": f.subject, "target": f.object, "id": f.id} for f in world.current_edges()]
    return nodes, edges


@dataclass(frozen=True)
class WorldIndex:
    """Look-ups for the detail panel: an entity by id, and every fact that touches it."""

    entities: dict[str, Entity]
    touching: dict[str, list[Edge]]


def index_world(world: World) -> WorldIndex:
    touching: dict[str, list[Edge]] = {}
    for fact in world.edges:
        touching.setdefault(fact.subject, []).append(fact)
        touching.setdefault(fact.object, []).append(fact)
    return WorldIndex({e.id: e for e in world.entities}, touching)


def entity_summary(index: WorldIndex | None, entity_id: str | None) -> str:
    """One line about the selected entity, or a prompt when nothing is selected."""
    if index is None or entity_id is None or entity_id not in index.entities:
        return "Click a node to see its facts."
    e = index.entities[entity_id]
    parts = [f"{e.name} ({e.id})", "a distractor" if e.distractor else e.type.lower()]
    parts.append(f"community {e.community}")
    if e.aliases:
        parts.append("also written " + ", ".join(e.aliases))
    return ", ".join(parts)


FACT_COLUMNS = ["Relation", "", "Other entity", "From", "Until"]


def fact_rows(index: WorldIndex | None, entity_id: str | None) -> list[list[str]]:
    """Every fact that touches the entity, historical ones included, oldest first."""
    if index is None or entity_id is None or entity_id not in index.touching:
        return []
    rows = []
    for f in sorted(index.touching[entity_id], key=lambda f: (f.valid_from, f.id)):
        outgoing = f.subject == entity_id
        other = index.entities[f.object if outgoing else f.subject]
        rows.append(
            [
                f.predicate,
                "to" if outgoing else "from",
                f"{other.name} ({other.id})",
                str(f.valid_from),
                "now" if f.valid_to is None else str(f.valid_to),
            ]
        )
    return rows


@dataclass(frozen=True)
class Summary:
    entities: int
    facts: int
    seed: int


def summarize(world: World) -> Summary:
    return Summary(len(world.entities), len(world.edges), world.spec.seed)


__all__ = [
    "DEFAULT_PRESET",
    "GROUPS",
    "KNOBS",
    "KNOB_NAMES",
    "FACT_COLUMNS",
    "MAX_GRAPH_EDGES",
    "MAX_GRAPH_NODES",
    "MAX_ENTITIES",
    "PRESET_NAMES",
    "Knob",
    "Summary",
    "check_rows",
    "entity_summary",
    "fact_rows",
    "graph_data",
    "graph_limits_message",
    "index_world",
    "generate_world",
    "preset_values",
    "spec_from_values",
    "summarize",
    "values_from_spec",
]
