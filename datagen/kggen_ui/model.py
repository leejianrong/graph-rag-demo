"""The UI's logic, with no indah import: knob definitions, building a spec, summarising a world."""

from __future__ import annotations

from dataclasses import dataclass

from kggen.questions import MAX_HOPS, Question, QuestionSpec, generate_questions
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


def read_knob(knob: Knob, raw: object) -> float:
    """One form value as a number, or a ``ValueError`` that says what is wrong with it."""
    if raw is None or raw == "":
        raise ValueError(f"{knob.label} needs a value")
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{knob.label} must be a number") from None
    if not knob.min <= number <= knob.max:
        raise ValueError(f"{knob.label} must be between {knob.min:g} and {knob.max:g}")
    return int(round(number)) if knob.integer else number


def spec_from_values(values: dict[str, float], base: WorldSpec | None = None) -> WorldSpec:
    """Build a validated spec from form values; a ``ValueError`` explains what is wrong."""
    base = base or load_preset(DEFAULT_PRESET)
    clean = {knob.name: read_knob(knob, values.get(knob.name)) for knob in KNOBS}
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


# The question form: how many, and the relative weight of each hop count.
QUESTION_KNOBS: tuple[Knob, ...] = (
    Knob("n_questions", "Questions", 1, 500, 1, integer=True, group="questions"),
    *(
        Knob(f"hops_{h}", f"{h}-hop weight", 0, 10, 1, integer=True, group="questions")
        for h in range(1, MAX_HOPS + 1)
    ),
)
QUESTION_DEFAULTS: dict[str, float] = {
    "n_questions": 20,
    **{f"hops_{h}": 1 for h in range(1, MAX_HOPS + 1)},
}


def question_spec_from_values(values: dict[str, float]) -> QuestionSpec:
    """Build the question spec from the form; a ``ValueError`` explains what is wrong."""
    read = {k.name: read_knob(k, values.get(k.name)) for k in QUESTION_KNOBS}
    mix = {h: float(read[f"hops_{h}"]) for h in range(1, MAX_HOPS + 1) if read[f"hops_{h}"] > 0}
    if not mix:
        raise ValueError("Give at least one hop count a weight above 0")
    return QuestionSpec(n_questions=int(read["n_questions"]), hop_mix=mix)


def derive_questions(world: World, values: dict[str, float]) -> tuple[Question, ...]:
    """Questions from the world, or a ``ValueError`` saying why it cannot (for example, too few
    chains with a unique answer in a small world)."""
    try:
        return generate_questions(world, question_spec_from_values(values))
    except WorldGenerationError as exc:
        raise ValueError(str(exc)) from None


QUESTION_COLUMNS = ("id", "hops", "question", "answer")


def question_rows(questions: tuple[Question, ...] | list[Question]) -> list[dict]:
    """The question list as table rows."""
    return [
        {"id": q.id, "hops": q.hops, "question": q.question, "answer": q.answer_name}
        for q in questions
    ]


def find_question(questions: tuple[Question, ...] | list[Question], qid: str | None):
    return next((q for q in questions if q.id == qid), None)


def gold_path(index: WorldIndex, q: Question) -> tuple[list[str], list[str]]:
    """The entities along a question's chain, start to answer, and the facts that link them."""
    nodes, facts = [q.start], list(q.supporting_facts)
    for fid in facts:
        f = index.facts[fid]
        nodes.append(f.object if f.subject == nodes[-1] else f.subject)
    return nodes, facts


def gold_rows(index: WorldIndex | None, q: Question | None) -> list[list[str]]:
    """The gold path as rows: hop, fact id, the fact read out, and when it held."""
    if index is None or q is None:
        return []
    rows = []
    for hop, fid in enumerate(q.supporting_facts, start=1):
        f = index.facts[fid]
        until = "now" if f.valid_to is None else str(f.valid_to)
        rows.append(
            [
                str(hop),
                fid,
                f"{index.entities[f.subject].name} {f.predicate} {index.entities[f.object].name}",
                f"{f.valid_from} to {until}",
            ]
        )
    return rows


def answer_text(q: Question | None) -> str:
    if q is None:
        return "Pick a question to see its answer and its gold path."
    also = (
        f" (also accepted: {', '.join(q.accepted_answers[1:])})"
        if len(q.accepted_answers) > 1
        else ""
    )
    return f"Answer: {q.answer_name}{also}"


def highlight_for(index: WorldIndex | None, q: Question | None) -> dict | None:
    """What to emphasise on the graph for a question: its entities and its facts."""
    if index is None or q is None:
        return None
    nodes, edges = gold_path(index, q)
    return {"nodes": nodes, "edges": edges}


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
    facts: dict[str, Edge]


def index_world(world: World) -> WorldIndex:
    touching: dict[str, list[Edge]] = {}
    for fact in world.edges:
        touching.setdefault(fact.subject, []).append(fact)
        touching.setdefault(fact.object, []).append(fact)
    return WorldIndex({e.id: e for e in world.entities}, touching, {f.id: f for f in world.edges})


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
    "QUESTION_COLUMNS",
    "QUESTION_DEFAULTS",
    "QUESTION_KNOBS",
    "FACT_COLUMNS",
    "MAX_GRAPH_EDGES",
    "MAX_GRAPH_NODES",
    "MAX_ENTITIES",
    "PRESET_NAMES",
    "Knob",
    "Summary",
    "answer_text",
    "check_rows",
    "derive_questions",
    "find_question",
    "gold_path",
    "gold_rows",
    "highlight_for",
    "question_rows",
    "question_spec_from_values",
    "read_knob",
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
