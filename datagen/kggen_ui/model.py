"""The UI's logic, with no indah import: knob definitions, building a spec, summarising a world."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kggen.export import export_archive
from kggen.llm import CachedChat, CacheMiss, OpenAICompatible, ResponseCache
from kggen.questions import (
    MAX_HOPS,
    QUESTION_TYPES,
    Question,
    QuestionSpec,
    generate_questions,
)
from kggen.render import Document, RenderSpec, Sentence, render_documents
from kggen.render_llm import InvalidRendering, render_documents_llm
from kggen.spec import PRESET_NAMES, WorldSpec, load_preset
from kggen.stats import check_against_spec
from kggen.world import Edge, Entity, World, WorldGenerationError, generate
from pydantic import ValidationError

MAX_ENTITIES = 50_000  # the stress preset; a bigger graph is drawn as a sample (ADR-0048)


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


# What each form field controls and affects: the text of its info tip.
HELP: dict[str, str] = {
    "n_entities": (
        "How many entities (nodes) the world has. More entities mean more facts, more "
        "documents and longer generation. The graph is drawn only up to 20,000 nodes; "
        "larger worlds still give stats, questions, documents and export."
    ),
    "avg_degree": (
        "Average number of facts each entity takes part in. Facts = entities x degree / "
        "2. Higher makes denser graphs and longer multi-hop paths; very low leaves many "
        "entities disconnected."
    ),
    "degree_skew": (
        "0 gives every entity a similar number of facts. Higher values create a few hub "
        "entities with many facts and a long tail. Hubs make questions with several valid "
        "answers and stress retrieval."
    ),
    "n_communities": (
        "How many clusters entities are grouped into. Facts mostly stay inside a cluster, "
        "so more communities give more separate topics and more distinct neighbourhoods."
    ),
    "community_mixing": (
        "The chance a fact joins two different communities. 0 keeps clusters isolated; "
        "higher blurs them. It decides how easily a path leaves its topic."
    ),
    "cycle_rate": (
        "Share of facts that close a three-entity loop. Cycles give paths that return to "
        "where they started, which tests that questions and traversal do not loop "
        "forever."
    ),
    "seed": (
        "Same settings and seed give exactly the same world. Change it for a different "
        "world of the same shape."
    ),
    "alias_rate": (
        "Share of entities that also go by other names. Documents may use an alias, so a "
        "system has to link different spellings to one entity."
    ),
    "homonym_rate": (
        "Extra entities that reuse another entity's name (as a share of entities). They "
        "test whether a system keeps two people or places with the same name apart."
    ),
    "distractor_rate": (
        "Extra look-alike entities with similar names that take part in no real facts you "
        "ask about. They add plausible but wrong candidates for retrieval."
    ),
    "noise_rate": (
        "Share of facts that are a vague 'related to' link between any two entities "
        "instead of a specific relation. It adds weak, uninformative edges."
    ),
    "churn_rate": (
        "Share of facts that are replaced later by a newer value (a new employer, a new "
        "city). The old value stays in the world as history, so questions can ask about "
        "the current or an earlier value."
    ),
    "retraction_rate": (
        "Share of facts withdrawn later with nothing replacing them. After retraction the "
        "fact should no longer be answered as true."
    ),
    "n_questions": (
        "How many questions to derive from the world. Fewer are returned if the world "
        "cannot supply enough with a single correct answer."
    ),
    "alias_use_rate": (
        "How often a document mentions an entity by one of its aliases instead of its main name."
    ),
    "target_words": (
        "How long each document is, in words. 0 gives short fact-only documents; 500 reads like a "
        "short article; 5,000 like a long report. Facts are spread through background prose that "
        "states nothing, so a long document is mostly noise around few facts. The LLM writer "
        "ignores this."
    ),
    "neighbour_facts": (
        "Facts about connected entities added to each long document as context, so one document "
        "mentions several entities. 0 keeps a document to its own subject's facts."
    ),
    "max_facts_per_doc": (
        "The most facts one document states. Higher gives longer documents with fewer "
        "files; lower gives many short ones."
    ),
}


_TYPE_HELP = {
    "entity": "Questions with one entity as the answer, kept only when it is unique.",
    "set": "Questions that ask for every entity that fits, between 2 and 25 of them.",
    "count": "Questions that ask how many entities fit; the answer is a number.",
    "compare": "Questions asking which of two entities has more of some relation.",
    "superlative": "Questions asking which entity has the most of some relation. Few exist.",
    "unanswerable": "Questions about a fact the data never held; the right reply is to decline",
    "false_premise": "Questions that state a fact the data contradicts; the reply should fix it",
    "as_of": "Questions about the value in a past year. Needs facts replaced later (churn).",
    "latest": "Questions about the current value of a fact that was replaced (needs churn).",
    "retracted": "Questions about a fact that was withdrawn and has no current answer.",
    "path": "Questions asking how two entities are connected; the answer is the chain of facts.",
    "open_ended": "'Summarize what is known' questions, graded on facts covered and stale claims.",
}


def knob_help(knob: Knob) -> str:
    if knob.name.startswith("hops_"):
        return (
            f"Relative share of questions that follow {knob.name[5:]} relation(s) from the start "
            "entity. 0 leaves that length out. Longer paths are harder."
        )
    if knob.name.startswith("type_"):
        kind = knob.name[5:]
        return (
            f"Relative share of {kind} questions. {_TYPE_HELP[kind]} If the world cannot supply "
            "enough of a type, plain entity questions make up the difference."
        )
    return HELP.get(knob.name, "")


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
    *(
        Knob(f"type_{t}", f"{t} questions weight", 0, 10, 1, integer=True, group="questions")
        for t in QUESTION_TYPES
    ),
)
QUESTION_DEFAULTS: dict[str, float] = {
    "n_questions": 20,
    **{f"hops_{h}": 1 for h in range(1, MAX_HOPS + 1)},
    **{f"type_{t}": 1 for t in QUESTION_TYPES},
}


def question_spec_from_values(values: dict[str, float]) -> QuestionSpec:
    """Build the question spec from the form; a ``ValueError`` explains what is wrong."""
    read = {k.name: read_knob(k, values.get(k.name)) for k in QUESTION_KNOBS}
    mix = {h: float(read[f"hops_{h}"]) for h in range(1, MAX_HOPS + 1) if read[f"hops_{h}"] > 0}
    if not mix:
        raise ValueError("Give at least one hop count a weight above 0")
    types = {t: float(read[f"type_{t}"]) for t in QUESTION_TYPES if read[f"type_{t}"] > 0}
    if not types:
        raise ValueError("Give at least one question type a weight above 0")
    return QuestionSpec(n_questions=int(read["n_questions"]), hop_mix=mix, type_mix=types)


def derive_questions(world: World, values: dict[str, float]) -> tuple[Question, ...]:
    """Questions from the world, or a ``ValueError`` saying why it cannot (for example, too few
    chains with a unique answer in a small world)."""
    try:
        return generate_questions(world, question_spec_from_values(values))
    except WorldGenerationError as exc:
        raise ValueError(str(exc)) from None


QUESTION_COLUMNS = ("id", "hops", "question", "answer")


def question_summary(questions: tuple[Question, ...] | list[Question]) -> str:
    """How many questions there are and of which types, so a type that fell back is visible."""
    if not questions:
        return ""
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.type] = counts.get(q.type, 0) + 1
    parts = ", ".join(f"{n} {t}" for t, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{len(questions)} questions: {parts}."


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
    if q.type != "entity":  # several answers or two entities: every entity and fact involved
        facts = list(dict.fromkeys(f for path in q.gold_paths for f in path))
        nodes = [q.start]
        for fid in facts:
            f = index.facts[fid]
            nodes += [n for n in (f.subject, f.object) if n not in nodes]
        return nodes, facts
    nodes, facts = [q.start], list(q.supporting_facts)
    for fid in facts:
        f = index.facts[fid]
        nodes.append(f.object if f.subject == nodes[-1] else f.subject)
    return nodes, facts


def _all_facts(q: Question) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f for path in q.gold_paths for f in path))


def gold_rows(index: WorldIndex | None, q: Question | None) -> list[list[str]]:
    """The gold path as rows: hop, fact id, the fact read out, and when it held."""
    if index is None or q is None:
        return []
    rows = []
    facts = q.supporting_facts if q.type == "entity" else _all_facts(q)
    for hop, fid in enumerate(facts, start=1):
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
    if q.type == "set":
        return f"Answers ({len(q.answers)}): {q.answer_name}"
    if q.type == "count":
        return f"Answer: {q.number}"
    if q.type == "open_ended":
        return (
            f"A good answer covers {len(q.required_facts)} facts and does not state any of the "
            f"{len(q.forbidden_claims)} stale claims."
        )
    if q.type == "path":
        return f"Chain ({q.hops} facts): {q.answer_name}"
    if q.type in ("as_of", "latest"):
        stale = f" (not {', '.join(q.stale_values)}, which held at another time)"
        return f"Answer: {q.answer_name}{stale}"
    if q.type in ("unanswerable", "false_premise", "retracted"):
        return f"Expected reply: decline or correct ({q.answer_name})"
    return f"Answer: {q.answer_name}{also}"


def highlight_for(index: WorldIndex | None, q: Question | None) -> dict | None:
    """What to emphasise on the graph for a question: its entities and its facts."""
    if index is None or q is None:
        return None
    nodes, edges = gold_path(index, q)
    return {"nodes": nodes, "edges": edges}


# -- documents ---------------------------------------------------------------------------------

# The template renderer's two knobs: how often a mention uses an alias, and the facts per document.
RENDER_KNOBS: tuple[Knob, ...] = (
    Knob("alias_use_rate", "Mentions that use an alias", 0, 1, 0.05, group="documents"),
    Knob(
        "max_facts_per_doc", "Most facts in a document", 1, 20, 1, integer=True, group="documents"
    ),
    Knob("target_words", "Words per document", 0, 10_000, 100, integer=True, group="documents"),
    Knob("neighbour_facts", "Context facts", 0, 50, 1, integer=True, group="documents"),
)
RENDER_DEFAULTS: dict[str, float] = {
    "alias_use_rate": 0.5,
    "max_facts_per_doc": 6,
    "target_words": 500,
    "neighbour_facts": 3,
}
MAX_RENDER_WORDS = 10_000_000  # most words the UI will render at once; the CLI has no limit
MAX_DOCUMENT_ROWS = 5_000  # the table shows this many; a bigger world still renders them all


def render_spec_from_values(values: dict[str, float]) -> RenderSpec:
    read = {k.name: read_knob(k, values.get(k.name)) for k in RENDER_KNOBS}
    return RenderSpec(
        alias_use_rate=float(read["alias_use_rate"]),
        max_facts_per_doc=int(read["max_facts_per_doc"]),
        target_words=int(read["target_words"]),
        neighbour_facts=int(read["neighbour_facts"]),
    )


def render_world(world: World, values: dict[str, float]) -> tuple[Document, ...]:
    """The world's facts as template documents: deterministic, offline, no LLM."""
    spec = render_spec_from_values(values)
    if len(world.entities) * spec.target_words > MAX_RENDER_WORDS:
        raise ValueError(
            f"{len(world.entities):,} entities at {spec.target_words:,} words each is more than "
            f"the {MAX_RENDER_WORDS:,} words the UI renders at once. Lower the words per document "
            "or use `kggen docs generate --words` from the command line."
        )
    return render_documents(world, spec)


LLM_DEFAULT_MODEL = "claude-haiku"
# Answers are cached here, so rendering the same world again calls nothing. Not user-editable: the
# server writes into it. The API key is read from KGGEN_LLM_API_KEY only, never typed into the UI.
LLM_CACHE_DIR = Path(os.environ.get("KGGEN_LLM_CACHE", ".cache/kggen-llm"))


@dataclass(frozen=True)
class LLMRendered:
    documents: tuple[Document, ...]
    calls: int  # requests that went to the endpoint
    hits: int  # requests answered from the cache


def render_world_llm(
    world: World,
    values: dict[str, float],
    base_url: str,
    model_name: str,
    cache_dir: Path | None = None,
) -> LLMRendered:
    """The world's facts as LLM-written documents. A blank ``base_url`` uses the cache only.

    Answers that were already valid stay cached, so after an error a retry carries on from there.
    """
    if not model_name.strip():
        raise ValueError("Give the model name the endpoint (or the cache) was built with.")
    base_url = base_url.strip()
    client = (
        OpenAICompatible(base_url, model_name.strip(), os.environ.get("KGGEN_LLM_API_KEY"))
        if base_url
        else None
    )
    chat = CachedChat(ResponseCache(cache_dir or LLM_CACHE_DIR), client, model_name.strip())
    try:
        documents = render_documents_llm(world, chat, render_spec_from_values(values))
    except CacheMiss:
        raise ValueError(
            "Not everything is in the cache yet. Enter an endpoint to write the rest."
        ) from None
    except (InvalidRendering, OSError, KeyError, IndexError) as exc:
        raise ValueError(
            f"The LLM renderer stopped: {exc}. Valid answers so far are cached."
        ) from exc
    return LLMRendered(documents, chat.calls, chat.hits)


def llm_note(rendered: LLMRendered, model_name: str) -> str:
    return (
        f"{len(rendered.documents):,} documents written by {model_name.strip()}: "
        f"{rendered.calls:,} endpoint calls, {rendered.hits:,} answered from the cache."
    )


def document_rows(documents: tuple[Document, ...] | list[Document]) -> list[dict]:
    """The document list as table rows (the first ``MAX_DOCUMENT_ROWS``)."""
    return [
        {
            "id": d.id,
            "title": d.title,
            "sentences": len(d.sentences),
            "words": len(d.text.split()),
            "kind": "distractor" if d.distractor else "",
        }
        for d in documents[:MAX_DOCUMENT_ROWS]
    ]


def find_document(documents: tuple[Document, ...] | list[Document], doc_id: str | None):
    return next((d for d in documents if d.id == doc_id), None)


def sentence_rows(document: Document | None) -> list[dict]:
    """One row per fact sentence of a document, keyed by the fact it expresses. Filler sentences
    state no fact, so they are not rows (the document text shows them)."""
    if document is None:
        return []
    return [{"fact": s.fact_ids[0], "sentence": s.text} for s in document.sentences if s.fact_ids]


def document_text(document: Document | None) -> str:
    return "" if document is None else document.text


def sentence_lookup(documents: tuple[Document, ...] | list[Document]) -> dict[str, Sentence]:
    """The sentence that expresses each fact: the one in the document about the fact's subject, or
    else the first. A long document also states a few facts about connected entities as context."""
    found: dict[str, Sentence] = {}
    own: set[str] = set()
    for d in documents:
        for s in d.sentences:
            about_subject = bool(s.mentions) and s.mentions[0].entity_id == d.subject
            for f in s.fact_ids:
                if f not in found or (about_subject and f not in own):
                    found[f] = s
                    if about_subject:
                        own.add(f)
    return found


def mention_rows(
    index: WorldIndex | None, lookup: dict[str, Sentence], fact_id: str | None
) -> list[list[str]]:
    """The entity mentions in a sentence: the surface form, who it is, and the id."""
    sentence = lookup.get(fact_id) if fact_id else None
    if index is None or sentence is None:
        return []
    return [[m.surface, index.entities[m.entity_id].name, m.entity_id] for m in sentence.mentions]


def fact_highlight(index: WorldIndex | None, fact_id: str | None) -> dict | None:
    """Emphasise a fact on the graph: both its entities, and the edge when the graph draws it. The
    graph holds only facts still true at the end, so a fact that ended has no edge to show."""
    if index is None or fact_id is None or fact_id not in index.facts:
        return None
    fact = index.facts[fact_id]
    return {
        "nodes": [fact.subject, fact.object],
        "edges": [fact.id] if fact.valid_to is None else [],
    }


def fact_note(index: WorldIndex | None, fact_id: str | None) -> str:
    """What the picked sentence says, as a fact, and whether the graph can show its edge."""
    if index is None or fact_id is None or fact_id not in index.facts:
        return "Click a sentence to see the fact it expresses on the graph."
    f = index.facts[fact_id]
    subject, obj = index.entities[f.subject].name, index.entities[f.object].name
    held = f"{f.valid_from} to {'now' if f.valid_to is None else f.valid_to}"
    text = f"{fact_id}: {subject} {f.predicate} {obj}, held {held}."
    if f.valid_to is not None:
        text += " It ended, so the graph (the final state) draws its two entities but no edge."
    return text


# -- export ------------------------------------------------------------------------------------


def export_name(world: World) -> str:
    """The file name for a world's dataset: what and which, so downloads do not collide."""
    return f"kggen-{world.spec.n_entities}-entities-seed{world.spec.seed}.zip"


def export_dataset(
    world: World, questions: QuestionSpec | None, render: RenderSpec | None
) -> tuple[bytes, str]:
    """The reproducible dataset zip for the world, with the question and render specs that made the
    questions and documents on screen (their defaults when the form was not valid)."""
    data = export_archive(world, questions or QuestionSpec(), render or RenderSpec())
    return data, export_name(world)


def export_summary(data: bytes, name: str) -> str:
    return f"{name}: {len(data) / 1e6:.1f} MB. Re-run it with the command in its README."


# What the graph view can hold (the indah Graph component's ceiling, ADR-0048).
MAX_GRAPH_NODES = 20_000
MAX_GRAPH_EDGES = 60_000
DISTRACTOR_GROUP = "Distractor"


def graph_data(world: World) -> tuple[list[dict], list[dict]]:
    """The world as the Graph component wants it: nodes coloured by type (distractors apart) and
    the facts still true at the end, each edge carrying its fact id."""
    nodes = [
        {"id": e.id, "label": e.name, "group": DISTRACTOR_GROUP if e.distractor else e.type}
        for e in world.entities
    ]
    edges = [{"source": f.subject, "target": f.object, "id": f.id} for f in world.current_edges()]
    return nodes, edges


def graph_view(world: World) -> tuple[list[dict], list[dict], str]:
    """What to draw, and a note on it. A world within the graph view's limits is drawn whole (the
    note is empty). A bigger one is drawn as a sample: the most connected entities that fit both
    limits, with the facts between them, so the picture shows the densest part of the world."""
    nodes, edges = graph_data(world)
    if len(nodes) <= MAX_GRAPH_NODES and len(edges) <= MAX_GRAPH_EDGES:
        return nodes, edges, ""
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1
    ranked = sorted(nodes, key=lambda n: (-degree.get(n["id"], 0), n["id"]))
    rank = {n["id"]: i for i, n in enumerate(ranked)}
    # An edge is drawn once both its ends are, that is from its later end's rank on, so the nodes
    # to keep are the top k, with k as large as both limits allow.
    came_in = sorted(max(rank[e["source"]], rank[e["target"]]) for e in edges)
    k = min(MAX_GRAPH_NODES, len(nodes))
    if len(came_in) > MAX_GRAPH_EDGES:
        k = min(k, came_in[MAX_GRAPH_EDGES])
    kept = {n["id"] for n in ranked[:k]}
    shown_nodes = [n for n in nodes if n["id"] in kept]
    shown_edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
    note = (
        f"This world has {len(nodes):,} entities and {len(edges):,} facts, more than the graph "
        f"view holds ({MAX_GRAPH_NODES:,} and {MAX_GRAPH_EDGES:,}). Showing the "
        f"{len(shown_nodes):,} most connected entities and the {len(shown_edges):,} facts "
        "between them."
    )
    return shown_nodes, shown_edges, note


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
    "HELP",
    "knob_help",
    "DEFAULT_PRESET",
    "GROUPS",
    "KNOBS",
    "KNOB_NAMES",
    "MAX_DOCUMENT_ROWS",
    "RENDER_DEFAULTS",
    "RENDER_KNOBS",
    "LLM_DEFAULT_MODEL",
    "LLM_CACHE_DIR",
    "LLMRendered",
    "render_world_llm",
    "llm_note",
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
    "export_dataset",
    "export_name",
    "export_summary",
    "document_rows",
    "fact_highlight",
    "fact_note",
    "find_document",
    "mention_rows",
    "render_spec_from_values",
    "render_world",
    "sentence_lookup",
    "sentence_rows",
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
    "graph_view",
    "index_world",
    "generate_world",
    "preset_values",
    "spec_from_values",
    "summarize",
    "values_from_spec",
]
