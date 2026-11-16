"""Multi-hop questions with gold paths, derived from the world model.

A question starts at one named entity and follows 1 to 4 typed relations; the answer is the
entity at the end. Questions come from the world's facts, never from any pipeline's output,
and every question carries the ids of the facts that support it (its gold path). A question
of the plain kind is kept only if its answer is unique: following the same relations from the start
entity reaches exactly one entity in the world's final state. Other kinds (ADR-0013) ask for every
entity that fits (``set``), how many (``count``), which of two has more (``compare``) or which has
the most (``superlative``). Two kinds have no answer to find: ``unanswerable`` asks about a relation
the world holds no fact for, and ``false_premise`` states a fact the world contradicts; the right
reply is to decline or to correct. Their gold is still read off the world's facts. When a world
cannot supply enough unique-answer questions at some hop count, the rest of that count are set
questions instead of an error, because on a large skewed world most long paths end in several
valid answers.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from kggen.ambiguity import NameReader
from kggen.spec import VAGUE_PREDICATE
from kggen.world import Edge, World, WorldGenerationError, apportion

MAX_HOPS = 4
MIN_SET = 2
MAX_BFS_NODES = 30_000  # a path search that grows past this many entities is dropped
MAX_REQUIRED = 8  # an open-ended question requires at most this many of the subject's own facts
MAX_CONTEXT = 4  # and at most this many facts about the entities it connects to
MAX_PATHS = 4  # a path question is kept when at most this many shortest chains link the pair
MAX_FRONTIER = 2000  # a path fanning out past this many entities is dropped as not unique
MAX_SET = 25  # a set question is kept only when it has this many answers or fewer
QUESTION_TYPES = (
    "entity",
    "set",
    "count",
    "compare",
    "superlative",
    "unanswerable",
    "false_premise",
    "as_of",
    "latest",
    "retracted",
    "when",
    "before",
    "duration",
    "conversation",
    "common",
    "path",
    "open_ended",
)

# How each relation reads in a question (the same wording whichever side is the answer).
VERB = {
    "LOCATED_IN": "is located in",
    "PART_OF": "is part of",
    "MEMBER_OF": "is a member of",
    "WORKS_FOR": "works for",
    "FOUNDED": "founded",
    "OWNS": "owns",
    "PRODUCES": "produces",
    "PARTICIPATED_IN": "participated in",
    "AFFILIATED_WITH": "is affiliated with",
    "GOVERNED_BY": "is governed by",
}
PLURAL = {
    "PERSON": "people",
    "ORG": "organizations",
    "LOCATION": "locations",
    "EVENT": "events",
    "PRODUCT": "products",
    "NORP": "groups",
    "LAW": "laws",
}
TYPE_LABEL = {
    "PERSON": "person",
    "ORG": "organization",
    "LOCATION": "location",
    "EVENT": "event",
    "PRODUCT": "product",
    "NORP": "group",
    "LAW": "law",
}


# How each relation reads when asking the year it began: "In what year did {S} ... ?"
_BEGAN = {
    "LOCATED_IN": "come to be located in",
    "PART_OF": "become part of",
    "MEMBER_OF": "join",
    "WORKS_FOR": "start working for",
    "FOUNDED": "found",
    "OWNS": "come to own",
    "PRODUCES": "begin producing",
    "PARTICIPATED_IN": "take part in",
    "AFFILIATED_WITH": "become affiliated with",
    "GOVERNED_BY": "come under",
}


# Who is asking (GRA-187). A persona is a profile of what to ask and how: ``with_persona`` sets the
# mixes, and the wording and the conversation shape follow from ``QuestionSpec.persona``.
PERSONA_NAMES = ("none", "newcomer", "learner", "expert", "mixed")
PERSONAS: dict[str, dict] = {
    # New to the world: discovery, one step at a time, "tell me about", things that may not exist.
    "newcomer": {
        "type_mix": {"entity": 3.0, "open_ended": 2.0, "set": 1.0, "unanswerable": 1.0},
        "hop_mix": {1: 3.0, 2: 1.0},
    },
    # Knows some of it: a conversation that follows what each answer reveals.
    "learner": {
        "type_mix": {"conversation": 4.0, "entity": 1.0, "set": 1.0},
        "hop_mix": {2: 2.0, 3: 1.0},
    },
    # Knows a lot: looks for links across several entities and for what does not hold.
    "expert": {
        "type_mix": {
            "common": 3.0,
            "path": 2.0,
            "compare": 1.0,
            "superlative": 1.0,
            "false_premise": 1.0,
            "count": 1.0,
        },
        "hop_mix": {2: 2.0, 3: 2.0, 4: 1.0},
    },
}


class QuestionSpec(BaseModel):
    """How many questions to derive and how many hops each should take."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    n_questions: int = Field(default=50, ge=1)
    hop_mix: dict[int, float] = Field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        description="Relative weight of questions with each hop count (1 to 4).",
    )

    type_mix: dict[str, float] = Field(
        default_factory=lambda: {"entity": 1.0},
        description=f"Relative weight of each question type: {', '.join(QUESTION_TYPES)}.",
    )
    persona: Literal["none", "newcomer", "learner", "expert", "mixed"] = Field(
        default="none",
        description="Who asks. newcomer, learner and expert shape the wording; mixed uses all three"
        " in equal shares. Use with_persona to take a persona's mix of types and hops too.",
    )
    ambiguity: Literal["ignore", "mark", "drop"] = Field(
        default="ignore",
        description="mark: record whether names alone give a second answer. drop: avoid those.",
    )
    phrasing: Literal["plain", "varied"] = Field(
        default="plain",
        description="plain: one wording per type. varied: a seeded choice from a fixed bank.",
    )

    @model_serializer(mode="wrap")
    def _omit_the_defaults(self, handler):  # type: ignore[no-untyped-def]
        """Leave the default ``phrasing`` and ``ambiguity`` out of a spec file, so old files keep
        their bytes."""
        data = handler(self)
        if data.get("phrasing") == "plain":
            data.pop("phrasing")
        if data.get("ambiguity") == "ignore":
            data.pop("ambiguity")
        if data.get("persona") == "none":
            data.pop("persona")
        return data

    @model_validator(mode="after")
    def _check(self) -> QuestionSpec:
        if not self.type_mix or set(self.type_mix) - set(QUESTION_TYPES):
            raise ValueError(f"type_mix keys must be within {QUESTION_TYPES}")
        if any(w < 0 for w in self.type_mix.values()) or sum(self.type_mix.values()) <= 0:
            raise ValueError("type_mix needs non-negative weights with a positive sum")
        if not self.hop_mix or set(self.hop_mix) - set(range(1, MAX_HOPS + 1)):
            raise ValueError(f"hop_mix keys must be within 1..{MAX_HOPS}")
        if any(w < 0 for w in self.hop_mix.values()) or sum(self.hop_mix.values()) <= 0:
            raise ValueError("hop_mix needs non-negative weights with a positive sum")
        return self


def with_persona(spec: QuestionSpec, persona: str) -> QuestionSpec:
    """``spec`` asking as ``persona``: its mix of types and hops, and its wording."""
    if persona not in PERSONA_NAMES:
        raise ValueError(f"unknown persona {persona!r}; choose from {list(PERSONA_NAMES)}")
    profile = PERSONAS.get(persona, {})
    return spec.model_copy(update={"persona": persona, **profile})


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    hops: int
    start: str  # entity id the question names
    answer: str  # entity id
    answer_name: str
    accepted_answers: tuple[str, ...]  # the answer's name and aliases
    supporting_facts: tuple[str, ...]  # fact ids, in hop order
    # The typed gold record (ADR-0013); a plain entity question fills the first four.
    type: str = "entity"
    answer_kind: str = "entity"  # entity | set | number
    answers: tuple[str, ...] = ()  # every correct entity id
    number: int | None = None  # the answer of a count question
    gold_paths: tuple[tuple[str, ...], ...] = ()  # one chain of fact ids per correct answer
    ordered: bool = False
    tie_rule: str = ""
    grading: str = "exact"  # exact | set_f1 | numeric | abstain | path
    abstain: dict | None = None  # {expected, reason}: the right reply is to decline
    # A false premise: {asserted_fact, contradicting_fact_ids, correction}.
    premise: dict | None = None
    # Open-ended questions: facts a good answer covers (with a weight) and claims it must not make.
    required_facts: tuple[dict, ...] = ()
    forbidden_claims: tuple[dict, ...] = ()
    as_of: int | None = None  # the year a temporal question asks about
    stale_values: tuple[str, ...] = ()  # names that held earlier and must not be given as current
    answer_names: tuple[tuple[str, ...], ...] = ()  # each answer's name and aliases, as `answers`
    # A fact on the path involves an entity whose name or alias another entity also carries, so
    # the text alone cannot tell the two apart (the generator's homonym case).
    shared_name: bool = False
    # Names alone give a second answer along a chain question's path (``kggen.ambiguity``); None
    # when it was not assessed, which keeps it out of the file.
    ambiguous_under_names: bool | None = None
    # Who asked (GRA-187); None when no persona was chosen, which keeps it out of the file.
    persona: str | None = None
    # A conversation: one entry per turn (question, answer, the fact that answers it). ``question``
    # is the last turn alone, which refers back with a pronoun; ``transcript`` has them all.
    turns: tuple[dict, ...] = ()

    def transcript(self) -> str:
        """The conversation so far, each earlier turn with its answer, then the last question."""
        if not self.turns:
            return self.question
        earlier = [f"Q: {t['question']} A: {t['answer_name']}." for t in self.turns[:-1]]
        return " ".join([*earlier, f"Q: {self.turns[-1]['question']}"])

    def to_dict(self) -> dict:
        record = {
            key: value
            for key, value in vars(self).items()
            if (value is not None or key not in ("ambiguous_under_names", "persona"))
            and (value or key != "turns")
        }
        return {
            **record,
            "accepted_answers": list(self.accepted_answers),
            "supporting_facts": list(self.supporting_facts),
            "answers": list(self.answers),
            "gold_paths": [list(p) for p in self.gold_paths],
            "stale_values": list(self.stale_values),
            "answer_names": [list(n) for n in self.answer_names],
            "required_facts": list(self.required_facts),
            "forbidden_claims": list(self.forbidden_claims),
            **({"turns": list(self.turns)} if self.turns else {}),
        }


# The wordings ``phrasing="varied"`` chooses from, planned here rather than invented at generation
# time. ``{np}`` is a noun phrase that starts with "the"; ``{label}`` and ``{plural}`` are a type's
# noun, ``{tail}`` the relative clause, ``{a}``, ``{b}`` and ``{what}`` as in the plain wording.
# The first of each is the plain wording. The gold record never depends on the wording.
_NAME_STEMS = (
    "Name {np}.",
    "Identify {np}.",
    "Give the name of {np}.",
    "What is the name of {np}?",
    "Which entity is {np}?",
)
_SET_STEMS = (
    "List every {label} that {tail}.",
    "Name all the {plural} that {tail}.",
    "Give every {label} that {tail}.",
)
_COUNT_STEMS = (
    "Count the {plural} that {tail}.",
    "What is the number of {plural} that {tail}?",
    "Give the number of {plural} that {tail}.",
)
_COMPARE_STEMS = (
    "Which of {a} and {b} has more {what}?",
    "Of {a} and {b}, which has more {what}?",
    "Between {a} and {b}, which has more {what}?",
)
_CLAIM_STEMS = (
    "{claim} Name {np}.",
    "{claim} Identify {np}.",
    "{claim} What is the name of {np}?",
)
_NAME_THE_STEMS = (
    "Name the {label} that {tail}.",
    "Identify the {label} that {tail}.",
    "What is the {label} that {tail}?",
    "Give the name of the {label} that {tail}.",
)


# What a newcomer asks about something: plain curiosity, in the first person.
_DISCOVER_STEMS = (
    "{wh} is {np}?",
    "Can you tell me {wh_lower} {np} is?",
    "I'm new to this. {wh} is {np}?",
    "I'd like to find out: {wh_lower} is {np}?",
)
_DISCOVER_TELL = (
    "Tell me about {name}",
    "I'm new here, so tell me what I should know about {name}",
    "Give me an overview of {name}",
)
# The follow-up turns of a conversation, which refer back to the last answer.
_FOLLOW_STEMS = (
    "And name the {label} that {tail}.",
    "Now name the {label} that {tail}.",
    "Next, name the {label} that {tail}.",
)
_COMMON_STEMS = (
    "What do {a} and {b} have in common?",
    "Which entities are directly connected to both {a} and {b}?",
    "Is there anything that links {a} and {b}? Name what both are connected to.",
)


class _Phrasing:
    """Chooses question wording. Plain always uses the first; varied draws from its own stream.

    The draws come from a stream apart from the one that picks the questions, so plain and varied
    produce the same questions, ids, answers and gold, and only the words differ.
    """

    def __init__(self, varied: bool, rng: random.Random, persona: str = "none") -> None:
        self._varied, self._rng, self.persona = varied, rng, persona

    @property
    def varied(self) -> bool:
        return self._varied

    def _pick(self, stems: tuple[str, ...]) -> str:
        return self._rng.choice(stems) if self._varied else stems[0]

    def relative(self, label: str) -> str:
        """The word that opens a relative clause after a type's noun."""
        if not self._varied:
            return "that"
        return self._rng.choice(("that", "who" if label == "person" else "which"))

    def name(self, np: str, label: str | None = None) -> str:
        if self.persona == "newcomer" and label:
            wh = "Who" if label == "person" else "What"
            stem = self._rng.choice(_DISCOVER_STEMS)
            return stem.format(np=np, wh=wh, wh_lower=wh.lower())
        return self._pick(_NAME_STEMS).format(np=np)

    def tell_me_about(self, name: str) -> str | None:
        """A newcomer's way to open an open-ended question; None for any other asker."""
        return (
            self._rng.choice(_DISCOVER_TELL).format(name=name)
            if self.persona == "newcomer"
            else None
        )

    def follow_up(self, label: str, tail: str) -> str:
        return self._pick(_FOLLOW_STEMS).format(label=label, tail=tail)

    def common(self, a: str, b: str) -> str:
        return self._pick(_COMMON_STEMS).format(a=a, b=b)

    def name_the(self, label: str, tail: str) -> str:
        return self._pick(_NAME_THE_STEMS).format(label=label, tail=tail)

    def members(
        self, kind: str, label: str, plural: str, inner: str, verb: str, forward: bool
    ) -> str:
        """A set or count question; varied wording makes the verb agree with a plural noun."""
        stem = self._pick(_SET_STEMS if kind == "set" else _COUNT_STEMS)
        if self._varied and "{plural}" in stem:
            verb = _plural_verb(verb) if not forward else verb
        tail = f"{inner} {verb}" if forward else f"{verb} {inner}"
        return stem.format(label=label, plural=plural, tail=tail)

    def compare(self, a: str, b: str, what: str) -> str:
        return self._pick(_COMPARE_STEMS).format(a=a, b=b, what=what)

    def claim(self, claim: str, np: str) -> str:
        return self._pick(_CLAIM_STEMS).format(claim=claim, np=np)


def _plural_verb(verb: str) -> str:
    """The relation's wording for a plural subject: "is part of" becomes "are part of"."""
    if verb == "is a member of":
        return "are members of"
    if verb.startswith("is "):
        return "are " + verb[3:]
    first, _, rest = verb.partition(" ")
    return (first.removesuffix("s") + (" " + rest if rest else "")) if first.endswith("s") else verb


# The noun a relation gives the entity at its far end, read from the near one ("Abel's employer"):
# ``(predicate, forward)`` where forward means the answer is the object. A relation without a noun
# in a direction is absent (nothing natural names "something located in X").
NOUN = {
    ("LOCATED_IN", True): "location",
    ("PART_OF", True): "parent",
    ("PART_OF", False): "part",
    ("MEMBER_OF", True): "group",
    ("MEMBER_OF", False): "member",
    ("WORKS_FOR", True): "employer",
    ("WORKS_FOR", False): "employee",
    ("FOUNDED", True): "venture",
    ("FOUNDED", False): "founder",
    ("OWNS", True): "holding",
    ("OWNS", False): "owner",
    ("PRODUCES", True): "product",
    ("PRODUCES", False): "producer",
    ("PARTICIPATED_IN", True): "event",
    ("PARTICIPATED_IN", False): "participant",
    ("AFFILIATED_WITH", True): "affiliate",
    ("AFFILIATED_WITH", False): "affiliate",
    ("GOVERNED_BY", True): "governing body",
}
# What a location calls what is in it, by the type of the thing in it.
_INSIDE = {
    "PERSON": "resident",
    "ORG": "local organization",
    "LOCATION": "sublocation",
    "EVENT": "local event",
    "PRODUCT": "local product",
}
_COUNT_WORD = {2: "twice", 3: "three times", 4: "four times"}


def _noun_phrase(
    world_names: dict[str, str],
    types: dict[str, str],
    start: str,
    steps,
    relative=None,
    possessive: bool = False,
) -> str:
    """The phrase naming the entity reached after ``steps`` from ``start``.

    ``relative`` chooses the word that opens each clause from the noun before it (default "that").
    With ``possessive`` a step that has a noun reads "Abel's employer's location" instead of nesting
    a clause per hop, and a run of three or more of one relation is stated once with a count.
    """
    phrase = world_names[start]
    poss = possessive
    i = 0
    while i < len(steps):
        edge, forward = steps[i]
        run = 1
        while i + run < len(steps) and (steps[i + run][0].predicate, steps[i + run][1]) == (
            edge.predicate,
            forward,
        ):
            run += 1
        verb = VERB[edge.predicate]
        target = edge.object if forward else edge.subject
        noun = NOUN.get((edge.predicate, forward))
        if edge.predicate == "LOCATED_IN" and not forward:
            noun = _INSIDE.get(types[target])
        label = TYPE_LABEL[types[target]]
        if poss and noun and run <= 2:
            of = "'" if phrase.endswith("s") else "'s"
            phrase += of + f" {noun}"
            for _ in range(run - 1):
                phrase += f"'s {noun}"
            i += run
            continue
        if poss and run >= 2:
            how = "" if forward else " backwards"
            phrase = (
                f"the {label} you reach by following \u201c{verb}\u201d links{how} "
                f"{_COUNT_WORD.get(run, f'{run} times')} from {phrase}"
            )
            i += run
            poss = False  # one clause at most: the rest nests as before
            continue
        poss = False
        word = relative(label) if relative else "that"
        if forward:
            phrase = f"the {label} {word} {phrase} {verb}"
        else:
            phrase = f"the {label} {word} {verb} {phrase}"
        i += 1
    return phrase


class _Walker:
    """Random simple paths over the live facts, without copying a hub's whole edge list per step.

    It draws the same random numbers as choosing from the list of unvisited neighbours would, so a
    seed keeps giving the same questions; it only avoids building that list.
    """

    def __init__(self, out_edges: dict[str, list[Edge]], in_edges: dict[str, list[Edge]]) -> None:
        self._out, self._in = out_edges, in_edges
        self._cache: dict[str, tuple[list[tuple[Edge, bool]], dict[str, list[int]]]] = {}

    def _around(self, node: str):
        if node not in self._cache:
            options = [(e, True) for e in self._out.get(node, [])] + [
                (e, False) for e in self._in.get(node, [])
            ]
            by_other: dict[str, list[int]] = {}
            for i, (e, forward) in enumerate(options):
                by_other.setdefault(e.object if forward else e.subject, []).append(i)
            self._cache[node] = (options, by_other)
        return self._cache[node]

    def walk(self, rng: random.Random, hops: int, start: str):
        """Up to ``hops`` steps from ``start``: a list of (edge, followed forward)."""
        steps: list[tuple[Edge, bool]] = []
        at, visited = start, {start}
        for _ in range(hops):
            options, by_other = self._around(at)
            skipped = sorted(i for v in visited for i in by_other.get(v, ()))
            if len(options) == len(skipped):
                break
            index = rng.choice(range(len(options) - len(skipped)))
            for i in skipped:
                if i <= index:
                    index += 1
            edge, forward = options[index]
            at = edge.object if forward else edge.subject
            visited.add(at)
            steps.append((edge, forward))
        return steps


def _mixed(world: World, spec: QuestionSpec) -> tuple[Question, ...]:
    """All three askers in equal shares, each with its own mix of types and hops."""
    shares = apportion(spec.n_questions, {p: 1.0 for p in ("newcomer", "learner", "expert")})
    out: list[Question] = []
    for persona in ("newcomer", "learner", "expert"):
        if not shares.get(persona):
            continue
        part = with_persona(spec, persona).model_copy(
            update={"n_questions": shares[persona], "seed": spec.seed}
        )
        out += generate_questions(world, part)
    width = len(str(len(out) - 1))
    return tuple(Question(**{**vars(q), "id": f"Q{i:0{width}d}"}) for i, q in enumerate(out))


def generate_questions(world: World, spec: QuestionSpec | None = None) -> tuple[Question, ...]:
    """Derive ``spec.n_questions`` questions from the world's final state, by ``spec.type_mix``."""
    spec = spec or QuestionSpec()
    rng = random.Random(f"{world.spec.seed}:{spec.seed}:questions")
    if spec.persona == "mixed":
        return _mixed(world, spec)
    phrasing = _Phrasing(
        spec.phrasing == "varied",
        random.Random(f"{world.spec.seed}:{spec.seed}:phrasing"),
        spec.persona,
    )
    entities = {e.id: e for e in world.entities}
    names = {i: e.name for i, e in entities.items()}
    types = {i: e.type for i, e in entities.items()}
    name_count = Counter(e.name for e in world.entities)
    reader = NameReader(world) if spec.ambiguity != "ignore" else None
    avoid_ambiguous = spec.ambiguity == "drop"

    # Typed facts between real entities that are still true at the end.
    live = [
        e
        for e in world.current_edges()
        if e.predicate != VAGUE_PREDICATE
        and not entities[e.subject].distractor
        and not entities[e.object].distractor
    ]
    out_edges: dict[str, list[Edge]] = {}
    in_edges: dict[str, list[Edge]] = {}
    for e in live:
        out_edges.setdefault(e.subject, []).append(e)
        in_edges.setdefault(e.object, []).append(e)

    walker = _Walker(out_edges, in_edges)
    ends: dict[tuple[str, str, bool], list[str]] = {}
    for e in live:
        ends.setdefault((e.subject, e.predicate, True), []).append(e.object)
        ends.setdefault((e.object, e.predicate, False), []).append(e.subject)

    def reach(frontier: set[str], predicate: str, forward: bool) -> set[str]:
        return {x for node in frontier for x in ends.get((node, predicate, forward), ())}

    def names_of(ids) -> tuple[str, ...]:
        return tuple(n for i in ids for n in (names[i], *entities[i].aliases))

    starts = sorted(i for i in set(out_edges) | set(in_edges) if name_count[names[i]] == 1)
    if not starts:
        raise WorldGenerationError("no entity has a unique name and a typed fact to ask about")

    seen: set[tuple] = set()

    def entity_questions(hops: int, target: int) -> list[Question]:
        found: list[Question] = []
        budget = 400 * target + 2000
        while len(found) < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            steps = walker.walk(rng, hops, start)
            if len(steps) < hops:
                continue
            at = start
            frontier = {start}
            for edge, forward in steps:
                at = edge.object if forward else edge.subject
                frontier = reach(frontier, edge.predicate, forward)
                if len(frontier) > MAX_FRONTIER:
                    break
            facts = tuple(e.id for e, _ in steps)
            if frontier != {at} or ("entity", start, facts) in seen:
                continue
            if avoid_ambiguous and reader and reader.ambiguous(start, facts, (at,)):
                continue  # a reader who sees only names would find a second answer
            seen.add(("entity", start, facts))
            answer = entities[at]
            found.append(
                Question(
                    id="",
                    question=phrasing.name(
                        _noun_phrase(
                            names, types, start, steps, phrasing.relative, phrasing.varied
                        ),
                        TYPE_LABEL[types[at]],
                    ),
                    hops=hops,
                    start=start,
                    answer=at,
                    answer_name=answer.name,
                    accepted_answers=(answer.name, *answer.aliases),
                    supporting_facts=facts,
                    answers=(at,),
                    gold_paths=(facts,),
                )
            )
        return found

    def set_questions(hops: int, target: int, kind: str) -> list[Question]:
        """Questions whose last step reaches 2 to MAX_SET entities, to list or to count."""
        found: list[Question] = []
        budget = 400 * target + 2000
        while len(found) < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            steps = walker.walk(rng, hops, start)
            if len(steps) < hops:
                continue
            prefix, (last, forward) = steps[:-1], steps[-1]
            at, frontier = start, {start}
            for edge, fwd in prefix:
                at = edge.object if fwd else edge.subject
                frontier = reach(frontier, edge.predicate, fwd)
                if len(frontier) > MAX_FRONTIER:
                    break
            if frontier != {at}:
                continue
            members = sorted(reach({at}, last.predicate, forward))
            if not MIN_SET <= len(members) <= MAX_SET:
                continue
            lead = tuple(e.id for e, _ in prefix)
            key = (kind, start, lead, last.predicate, forward)
            if key in seen:
                continue
            seen.add(key)
            edge_of = {
                (e.object if forward else e.subject): e.id
                for e in (out_edges if forward else in_edges).get(at, [])
                if e.predicate == last.predicate
            }
            paths = tuple((*lead, edge_of[m]) for m in members)
            if avoid_ambiguous and reader and reader.ambiguous(start, paths[0], members):
                continue
            inner = _noun_phrase(names, types, start, prefix, phrasing.relative, phrasing.varied)
            label = TYPE_LABEL[types[members[0]]]
            plural = PLURAL[types[members[0]]]
            verb = VERB[last.predicate]
            text = phrasing.members(kind, label, plural, inner, verb, forward)
            found.append(
                Question(
                    id="",
                    question=text,
                    hops=hops,
                    start=start,
                    answer="",
                    answer_name=(
                        "; ".join(names[m] for m in members) if kind == "set" else str(len(members))
                    ),
                    accepted_answers=names_of(members),
                    supporting_facts=paths[0],
                    type=kind,
                    answer_kind="set" if kind == "set" else "number",
                    answers=tuple(members),
                    number=len(members) if kind == "count" else None,
                    gold_paths=paths,
                    grading="set_f1" if kind == "set" else "numeric",
                )
            )
        return found

    # How many facts each entity has for each relation and direction.
    counts: dict[tuple[str, bool], dict[str, list[str]]] = {}
    for e in live:
        counts.setdefault((e.predicate, True), {}).setdefault(e.subject, []).append(e.id)
        counts.setdefault((e.predicate, False), {}).setdefault(e.object, []).append(e.id)

    def phrase_of(predicate: str, forward: bool, plural: str) -> str:
        verb = VERB[predicate]
        return f"{plural} that it {verb}" if forward else f"{plural} that {_plural_verb(verb)} it"

    by_id = {e.id: e for e in live}

    def target_plural(predicate: str, forward: bool, members: dict[str, list[str]]) -> str:
        fact = by_id[next(iter(next(iter(members.values()))))]
        return PLURAL[types[fact.object if forward else fact.subject]]

    def compare_questions(target: int) -> list[Question]:
        found: list[Question] = []
        keys = sorted(counts)
        budget = 200 * target + 500
        while len(found) < target and budget > 0 and keys:
            budget -= 1
            predicate, forward = rng.choice(keys)
            members = counts[(predicate, forward)]
            unique = sorted(i for i in members if name_count[names[i]] == 1)
            if len(unique) < 2:
                continue
            a, b = rng.sample(unique, 2)
            if types[a] != types[b] or len(members[a]) == len(members[b]):
                continue
            key = ("compare", frozenset((a, b)), predicate, forward)
            if key in seen:
                continue
            seen.add(key)
            winner, loser = (a, b) if len(members[a]) > len(members[b]) else (b, a)
            plural = target_plural(predicate, forward, members)
            first, second = sorted((a, b))
            paths = tuple((f,) for f in (*members[winner], *members[loser]))
            found.append(
                Question(
                    id="",
                    question=phrasing.compare(
                        names[first], names[second], phrase_of(predicate, forward, plural)
                    ),
                    hops=1,
                    start=first,
                    answer=winner,
                    answer_name=names[winner],
                    accepted_answers=(names[winner], *entities[winner].aliases),
                    supporting_facts=paths[0],
                    type="compare",
                    answers=(winner,),
                    gold_paths=paths,
                    tie_rule="the counts differ",
                )
            )
        return found

    def superlative_questions(target: int) -> list[Question]:
        found: list[Question] = []
        for predicate, forward in sorted(counts):
            if len(found) >= target:
                break
            members = counts[(predicate, forward)]
            ranked = sorted(members, key=lambda i: (-len(members[i]), i))
            top = ranked[0]
            if len(members[top]) < 2 or (
                len(ranked) > 1 and len(members[ranked[1]]) == len(members[top])
            ):
                continue
            if name_count[names[top]] != 1 or (("superlative", predicate, forward) in seen):
                continue
            seen.add(("superlative", predicate, forward))
            label = TYPE_LABEL[types[top]]
            plural = target_plural(predicate, forward, members)
            paths = tuple((f,) for f in members[top])
            found.append(
                Question(
                    id="",
                    question=(
                        f"Which {label} has the most {phrase_of(predicate, forward, plural)}?"
                    ),
                    hops=1,
                    start=top,
                    answer=top,
                    answer_name=names[top],
                    accepted_answers=(names[top], *entities[top].aliases),
                    supporting_facts=paths[0],
                    type="superlative",
                    answers=(top,),
                    gold_paths=paths,
                    tie_rule="a single entity has the most",
                )
            )
        return found

    ever = {(e.subject, e.predicate, e.object) for e in world.edges}  # live and past facts
    held: dict[tuple[str, str, bool], bool] = {}
    for s_, p_, o_ in ever:
        held[(s_, p_, True)] = held[(o_, p_, False)] = True
    holders: dict[tuple[str, bool], set[str]] = {
        k: {types[t] for t in m} for k, m in counts.items()
    }

    def unanswerable_questions(target: int) -> list[Question]:
        found: list[Question] = []
        keys = sorted(counts)
        budget = 200 * target + 500
        while len(found) < target and budget > 0 and keys:
            budget -= 1
            start = rng.choice(starts)
            predicate, forward = rng.choice(keys)
            if types[start] not in holders[(predicate, forward)]:
                continue
            if (
                held.get((start, predicate, forward))
                or ("unanswerable", start, predicate, forward) in seen
            ):
                continue
            seen.add(("unanswerable", start, predicate, forward))
            other = by_id[next(iter(next(iter(counts[(predicate, forward)].values()))))]
            label = TYPE_LABEL[types[other.object if forward else other.subject]]
            verb = VERB[predicate]
            tail = f"{names[start]} {verb}" if forward else f"{verb} {names[start]}"
            found.append(
                Question(
                    id="",
                    question=phrasing.name_the(label, tail),
                    hops=1,
                    start=start,
                    answer="",
                    answer_name="no answer: the data holds no such fact",
                    accepted_answers=(),
                    supporting_facts=(),
                    type="unanswerable",
                    answer_kind="abstain",
                    grading="abstain",
                    abstain={"expected": True, "reason": "no_evidence"},
                )
            )
        return found

    def false_premise_questions(target: int) -> list[Question]:
        found: list[Question] = []
        budget = 400 * target + 1000
        while len(found) < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            options = [(e, True) for e in out_edges.get(start, [])] + [
                (e, False) for e in in_edges.get(start, [])
            ]
            if not options:
                continue
            edge, forward = rng.choice(options)
            true_end = edge.object if forward else edge.subject
            if reach({start}, edge.predicate, forward) != {true_end}:
                continue  # the real answer must be the only one, so the correction is exact
            fake = rng.choice(starts)
            if (
                fake in (start, true_end)
                or types[fake] != types[true_end]
                or ((start, edge.predicate, fake) if forward else (fake, edge.predicate, start))
                in ever
            ):
                continue
            nxt = [(e, True) for e in out_edges.get(fake, []) if e.object != start] + [
                (e, False) for e in in_edges.get(fake, []) if e.subject != start
            ]
            if not nxt or ("false_premise", start, edge.id, fake) in seen:
                continue
            seen.add(("false_premise", start, edge.id, fake))
            second, second_forward = rng.choice(nxt)
            verb = VERB[edge.predicate]
            claim = (
                f"{names[start]} {verb} {names[fake]}."
                if forward
                else f"{names[fake]} {verb} {names[start]}."
            )
            truth = (
                f"{names[start]} {verb} {names[true_end]}"
                if forward
                else f"{names[true_end]} {verb} {names[start]}"
            )
            asked = _noun_phrase(
                names, types, fake, [(second, second_forward)], phrasing.relative, phrasing.varied
            )
            found.append(
                Question(
                    id="",
                    question=phrasing.claim(claim, asked),
                    hops=2,
                    start=start,
                    answer="",
                    answer_name=f"the premise is false: {truth}",
                    accepted_answers=(),
                    supporting_facts=(edge.id,),
                    type="false_premise",
                    answer_kind="false_premise",
                    gold_paths=((edge.id,),),
                    grading="abstain",
                    abstain={"expected": True, "reason": "false_premise"},
                    premise={
                        "asserted_fact": {
                            "subject": start if forward else fake,
                            "predicate": edge.predicate,
                            "object": fake if forward else start,
                        },
                        "contradicting_fact_ids": [edge.id],
                        "correction": truth,
                    },
                )
            )
        return found

    # Each (subject, relation, kind of object) over time, for the questions about change. Grouping
    # by the kind of object keeps "the organization that X owns" apart from "the product X owns".
    history: dict[tuple[str, str, str], list[Edge]] = {}
    for e in sorted(world.edges, key=lambda e: (e.valid_from, e.id)):
        if (
            e.predicate != VAGUE_PREDICATE
            and not entities[e.subject].distractor
            and not entities[e.object].distractor
            and name_count[names[e.subject]] == 1
        ):
            history.setdefault((e.subject, e.predicate, types[e.object]), []).append(e)
    changed = sorted(
        k
        for k, v in history.items()
        if any(a.valid_to == b.valid_from for a in v for b in v if a is not b)
    )
    withdrawn = sorted(k for k, v in history.items() if all(e.valid_to is not None for e in v))
    axis = world.spec.axis
    last_year = axis.end

    def asked(subject: str, predicate: str, kind: str, prefix: str = "") -> str:
        return f"{prefix}name the {TYPE_LABEL[kind]} that {names[subject]} {VERB[predicate]}"

    def temporal_questions(kind: str, target: int) -> list[Question]:
        found: list[Question] = []
        keys = {"as_of": changed, "latest": changed, "retracted": withdrawn}[kind]
        budget = 200 * target + 500
        while len(found) < target and budget > 0 and keys:
            budget -= 1
            subject, predicate, kind_of = rng.choice(keys)
            versions = history[(subject, predicate, kind_of)]
            key: tuple
            if kind == "retracted":
                gone = rng.choice(versions)
                last_end = max(e.valid_to or 0 for e in versions)
                key = (kind, subject, predicate, kind_of)
                answers: tuple[str, ...] = ()
                stale = tuple(dict.fromkeys(names[e.object] for e in versions))
                text = asked(subject, predicate, kind_of, "Currently, ")
                gold = ((gone.id,),)
                shown = f"no current answer: the fact was withdrawn in {axis.label(last_end)}"
                year = None
            elif kind == "latest":
                now = [e for e in versions if e.valid_to is None]
                if len(now) != 1 or len(versions) < 2:
                    continue
                key = (kind, subject, predicate, kind_of)
                answers = (now[0].object,)
                stale = tuple(dict.fromkeys(names[e.object] for e in versions if e is not now[0]))
                text = asked(subject, predicate, kind_of, "Currently, ")
                gold = ((now[0].id,),)
                shown, year = names[now[0].object], None
            else:
                pick = rng.choice(versions)
                year = rng.randint(pick.valid_from, (pick.valid_to or last_year + 1) - 1)
                holding = {
                    e.object
                    for e in versions
                    if e.valid_from <= year and (e.valid_to is None or year < e.valid_to)
                }
                if holding != {pick.object}:
                    continue
                key = (kind, subject, predicate, kind_of, year)
                answers = (pick.object,)
                stale = tuple(
                    dict.fromkeys(names[e.object] for e in versions if e.object != pick.object)
                )
                text = asked(subject, predicate, kind_of, f"As of {axis.label(year)}, ")
                gold = ((pick.id,),)
                shown = names[pick.object]
            if key in seen:
                continue
            seen.add(key)
            text = text[0].upper() + text[1:] + "."
            found.append(
                Question(
                    id="",
                    question=text,
                    hops=1,
                    start=subject,
                    answer=answers[0] if answers else "",
                    answer_name=shown,
                    accepted_answers=names_of(answers),
                    supporting_facts=gold[0],
                    type=kind,
                    answer_kind="abstain" if kind == "retracted" else "entity",
                    answers=answers,
                    gold_paths=gold,
                    grading="abstain" if kind == "retracted" else "exact",
                    abstain=(
                        {"expected": True, "reason": "retracted"} if kind == "retracted" else None
                    ),
                    as_of=year,
                    stale_values=stale,
                )
            )
        return found

    def timeline_questions(kind: str, target: int) -> list[Question]:
        """Questions on when a fact held: the year it began or ended, what came just before it,
        and how long it lasted. Only a fact whose dates the documents state is asked about."""
        found: list[Question] = []
        start = axis.start
        facts = sorted(
            (
                e
                for versions in history.values()
                for e in versions
                if name_count[names[e.object]] == 1
            ),
            key=lambda e: e.id,
        )
        if kind == "before":
            pairs = [
                (a, b)
                for versions in history.values()
                for a in versions
                for b in versions
                if a.valid_to is not None
                and a.valid_to == b.valid_from
                and a.object != b.object
                and name_count[names[a.object]] == 1
                and name_count[names[b.object]] == 1
            ]
            pairs.sort(key=lambda p: (p[0].id, p[1].id))
            pool: list = pairs
        elif kind == "duration":
            pool = [e for e in facts if e.valid_from > start and e.valid_to is not None]
        else:
            pool = [e for e in facts if e.valid_from > start or e.valid_to is not None]
        rng.shuffle(pool)
        for item in pool:
            if len(found) >= target:
                break
            if kind == "before":
                a, b = item
                key: tuple = (kind, a.id, b.id)
                label = TYPE_LABEL[types[a.object]]
                text = (
                    f"Name the {label} that {names[a.subject]} {VERB[a.predicate]} "
                    f"immediately before {names[b.object]}."
                )
                answers: tuple[str, ...] = (a.object,)
                shown = names[a.object]
                gold: tuple[str, ...] = (a.id, b.id)
                number = None
                accepted = names_of(answers)
                number_kind = False
            else:
                e = item
                s_, o_ = names[e.subject], names[e.object]
                if kind == "duration":
                    key = (kind, e.id)
                    text = (
                        f"For how many {axis.noun(True)} was it true that "
                        f"{s_} {VERB[e.predicate]} {o_}?"
                    )
                    number = e.valid_to - e.valid_from
                else:
                    began = e.valid_from > start and (e.valid_to is None or rng.random() < 0.5)
                    key = (kind, e.id, began)
                    if began:
                        text = f"In what {axis.noun()} did {s_} {_BEGAN[e.predicate]} {o_}?"
                        number = e.valid_from
                    else:
                        text = (
                            f"Until what {axis.noun()} was it true that "
                            f"{s_} {VERB[e.predicate]} {o_}?"
                        )
                        number = e.valid_to
                answers, shown, gold = (), str(number), (e.id,)  # type: ignore[assignment]
                accepted = (shown,)
                number_kind = True
                if kind != "duration" and axis.unit != "year":
                    # A month or a day is not a number to compare: it is a date, written the way
                    # the documents write it (or as ISO), and graded like a name.
                    shown, accepted, number_kind = axis.label(number), axis.accepted(number), False
            if key in seen:
                continue
            seen.add(key)
            found.append(
                Question(
                    id="",
                    question=text,
                    hops=len(gold),
                    start=item[0].subject if kind == "before" else item.subject,
                    answer=answers[0] if answers else "",
                    answer_name=shown,
                    accepted_answers=accepted,
                    supporting_facts=gold,
                    type=kind,
                    answer_kind="number" if number_kind else "entity",
                    answers=answers,
                    number=number if number_kind else None,
                    gold_paths=(gold,),
                    grading="numeric" if number_kind else "exact",
                )
            )
        return found

    def conversation_questions(turns: int, target: int) -> list[Question]:
        """A chain asked one step at a time, each turn referring back to the last answer."""
        found: list[Question] = []
        budget = 400 * target + 2000
        while len(found) < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            steps = walker.walk(rng, turns, start)
            if len(steps) < turns:
                continue
            at, rows, texts = start, [], []
            for i, (edge, forward) in enumerate(steps):
                nxt = edge.object if forward else edge.subject
                if reach({at}, edge.predicate, forward) != {nxt} or name_count[names[nxt]] != 1:
                    break  # this turn would have more than one answer
                label, verb = TYPE_LABEL[types[nxt]], VERB[edge.predicate]
                if i == 0:
                    np = _noun_phrase(
                        names, types, start, [(edge, forward)], phrasing.relative, phrasing.varied
                    )
                    text = phrasing.name(np)
                else:
                    person = types[at] == "PERSON"
                    verb = _plural_verb(verb) if person else verb
                    pron = "they" if person else "it"
                    obj = "them" if person else "it"
                    tail = f"{pron} {verb}" if forward else f"{verb} {obj}"
                    text = phrasing.follow_up(label, tail)
                texts.append(text)
                rows.append(
                    {
                        "question": text,
                        "answer": nxt,
                        "answer_name": names[nxt],
                        "supporting_facts": [edge.id],
                    }
                )
                at = nxt
            else:
                facts = tuple(e.id for e, _ in steps)
                key = ("conversation", start, facts)
                if key in seen or len(set(names[r["answer"]] for r in rows)) != len(rows):
                    continue
                if avoid_ambiguous and reader and reader.ambiguous(start, facts, (at,)):
                    continue
                seen.add(key)
                final = entities[at]
                found.append(
                    Question(
                        id="",
                        question=texts[-1],
                        hops=turns,
                        start=start,
                        answer=at,
                        answer_name=final.name,
                        accepted_answers=(final.name, *final.aliases),
                        supporting_facts=facts,
                        type="conversation",
                        answers=(at,),
                        gold_paths=(facts,),
                        turns=tuple(rows),
                    )
                )
        return found

    neighbours: dict[str, dict[str, Edge]] = {}  # entity -> {neighbour: its lowest-id fact}
    for e in sorted(live, key=lambda e: e.id):
        neighbours.setdefault(e.subject, {}).setdefault(e.object, e)
        neighbours.setdefault(e.object, {}).setdefault(e.subject, e)

    def common_questions(target: int) -> list[Question]:
        """What two entities both connect to: a link across the graph that a chain does not show."""
        found: list[Question] = []
        budget = 200 * target + 1000
        while len(found) < target and budget > 0:
            budget -= 1
            a = rng.choice(starts)
            near = sorted(neighbours.get(a, ()))
            if not near:
                continue
            via = rng.choice(near)
            others = sorted(set(neighbours.get(via, ())) - {a})
            if not others:
                continue
            b = rng.choice(others)
            if name_count[names[b]] != 1 or b == a or b in neighbours.get(a, ()):
                continue
            shared = sorted((set(neighbours[a]) & set(neighbours[b])) - {a, b})
            if not 1 <= len(shared) <= 5 or any(name_count[names[m]] != 1 for m in shared):
                continue
            key = ("common", *sorted((a, b)))
            if key in seen:
                continue
            paths = tuple((neighbours[a][m].id, neighbours[b][m].id) for m in shared)
            if avoid_ambiguous and reader and reader.ambiguous(a, paths[0], tuple(shared)):
                continue
            seen.add(key)
            found.append(
                Question(
                    id="",
                    question=phrasing.common(names[a], names[b]),
                    hops=2,
                    start=a,
                    answer="",
                    answer_name="; ".join(names[m] for m in shared),
                    accepted_answers=names_of(shared),
                    supporting_facts=paths[0],
                    type="common",
                    answer_kind="set",
                    answers=tuple(shared),
                    gold_paths=paths,
                    grading="set_f1",
                )
            )
        return found

    def path_questions(hops: int, target: int) -> list[Question]:
        """ "How is A connected to B?" where the shortest chain of facts is ``hops`` long."""
        found: list[Question] = []
        budget = 40 * target + 200
        while len(found) < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            steps = walker.walk(rng, hops, start)
            if len(steps) < hops:
                continue
            end = steps[-1][0].object if steps[-1][1] else steps[-1][0].subject
            if name_count[names[end]] != 1 or end == start or ("path", start, end) in seen:
                continue
            layers = {start: 0}
            parents: dict[str, list[tuple[str, Edge]]] = {}
            frontier = [start]
            for depth in range(1, hops + 1):
                nxt: list[str] = []
                for node in frontier:
                    for e in (*out_edges.get(node, ()), *in_edges.get(node, ())):
                        other = e.object if e.subject == node else e.subject
                        if layers.setdefault(other, depth) == depth:
                            parents.setdefault(other, []).append((node, e))
                            if len(parents[other]) == 1:
                                nxt.append(other)
                frontier = nxt
                if len(layers) > MAX_BFS_NODES:
                    break
            if layers.get(end) != hops:
                continue  # a shorter chain exists, or the search was cut off
            walked: list[tuple[Edge, ...]] = []
            stack: list[tuple[str, tuple[Edge, ...]]] = [(end, ())]
            while stack and len(walked) <= MAX_PATHS:
                node, tail = stack.pop()
                if node == start:
                    walked.append(tail)
                    continue
                stack += [(prev, (e, *tail)) for prev, e in parents.get(node, ())]
            if not 1 <= len(walked) <= MAX_PATHS:
                continue
            seen.add(("path", start, end))
            paths = tuple(sorted(tuple(e.id for e in chain) for chain in walked))
            first = {e.id: e for chain in walked for e in chain}
            at, readout = start, []
            for fid in paths[0]:
                e = first[fid]
                nxt_id = e.object if e.subject == at else e.subject
                readout.append(f"{names[e.subject]} {VERB[e.predicate]} {names[e.object]}")
                at = nxt_id
            found.append(
                Question(
                    id="",
                    question=f"How is {names[start]} connected to {names[end]}? "
                    "Give the chain of facts.",
                    hops=hops,
                    start=start,
                    answer="",
                    answer_name="; ".join(readout),
                    accepted_answers=(),
                    supporting_facts=paths[0],
                    type="path",
                    answer_kind="path",
                    answers=(end,),
                    gold_paths=paths,
                    ordered=True,
                    grading="path",
                )
            )
        return found

    def open_ended_questions(hops: int, target: int) -> list[Question]:
        """ "Summarize what is known about X": gold is facts to cover and claims to avoid."""
        found: list[Question] = []
        around = {
            i: sorted((*out_edges.get(i, ()), *in_edges.get(i, ())), key=lambda e: e.id)
            for i in starts
        }
        eligible = [i for i in starts if len(around[i]) >= 3]
        everything = [e for e in world.edges if e.predicate != VAGUE_PREDICATE]
        budget = 200 * target + 500
        while len(found) < target and budget > 0 and eligible:
            budget -= 1
            start = rng.choice(eligible)
            if ("open_ended", start, hops) in seen:
                continue
            seen.add(("open_ended", start, hops))
            own = around[start][:MAX_REQUIRED]
            context: list[Edge] = []
            if hops >= 2:
                near = sorted({x for e in own for x in (e.subject, e.object)} - {start})
                taken = {e.id for e in own}
                context = [
                    e
                    for n in near
                    for e in sorted(
                        (*out_edges.get(n, ()), *in_edges.get(n, ())), key=lambda e: e.id
                    )
                    if e.id not in taken and start not in (e.subject, e.object)
                ][:MAX_CONTEXT]

            def record(edge: Edge, **extra) -> dict:
                return {
                    "fact_id": edge.id,
                    "text": f"{names[edge.subject]} {VERB[edge.predicate]} {names[edge.object]}",
                    "subject": edge.subject,
                    "object": edge.object,
                    "subject_names": list(names_of([edge.subject])),
                    "object_names": list(names_of([edge.object])),
                    **extra,
                }

            required = tuple(record(e, weight=1.0) for e in own) + tuple(
                record(e, weight=0.5) for e in context
            )
            # A stale claim between two entities that a current fact also links (under another
            # relation) cannot be told apart from it by names, so it is not asked about.
            linked = {frozenset((e.subject, e.object)) for e in live}
            forbidden = tuple(
                record(
                    e,
                    reason=(
                        "superseded"
                        if any(
                            o.subject == e.subject and o.predicate == e.predicate and o is not e
                            for o in live
                        )
                        else "retracted"
                    ),
                )
                for e in everything
                if start in (e.subject, e.object)
                and e.valid_to is not None
                and frozenset((e.subject, e.object)) not in linked
                and not entities[e.subject].distractor
                and not entities[e.object].distractor
            )
            lead = phrasing.tell_me_about(names[start]) or (
                f"Summarize what is known about {names[start]}"
            )
            text = lead + (" and the entities it is directly connected to." if hops >= 2 else ".")
            paths = tuple((r["fact_id"],) for r in required)
            found.append(
                Question(
                    id="",
                    question=text,
                    hops=hops,
                    start=start,
                    answer="",
                    answer_name=(
                        f"open-ended: cover {len(required)} facts, avoid {len(forbidden)} claims"
                    ),
                    accepted_answers=(),
                    supporting_facts=paths[0],
                    type="open_ended",
                    answer_kind="open_ended",
                    gold_paths=paths,
                    grading="nugget",
                    required_facts=required,
                    forbidden_claims=forbidden,
                )
            )
        return found

    quota = apportion(spec.n_questions, spec.type_mix)
    questions: list[Question] = []
    short = 0  # questions another type could not supply; plain entity questions make them up
    for kind in ("set", "count", "path", "open_ended", "conversation"):
        wanted = apportion(quota.get(kind, 0), {str(h): w for h, w in spec.hop_mix.items()})
        for hops_key in sorted(wanted):
            hops, n = int(hops_key), wanted[hops_key]
            if kind == "conversation":
                got = conversation_questions(max(2, hops), n)
            elif kind == "path":
                got = path_questions(hops, n)
            elif kind == "open_ended":
                got = open_ended_questions(min(hops, 2), n)
            else:
                got = set_questions(hops, n, kind)
            questions += got
            short += wanted[hops_key] - len(got)
    for kind, make in (
        ("compare", compare_questions),
        ("superlative", superlative_questions),
        ("unanswerable", unanswerable_questions),
        ("false_premise", false_premise_questions),
        ("common", common_questions),
        *((t, lambda n, t=t: temporal_questions(t, n)) for t in ("as_of", "latest", "retracted")),
        *((t, lambda n, t=t: timeline_questions(t, n)) for t in ("when", "before", "duration")),
    ):
        got = make(quota.get(kind, 0))
        questions += got
        short += quota.get(kind, 0) - len(got)

    entity_total = quota.get("entity", 0) + short
    wanted = apportion(entity_total, {str(h): w for h, w in spec.hop_mix.items()})
    for hops_key in sorted(wanted):
        hops, target = int(hops_key), wanted[hops_key]
        got = entity_questions(hops, target)
        if len(got) < target:  # make up the rest with sets, whose answers need not be unique
            got += set_questions(hops, target - len(got), "set")
        if len(got) < target:
            raise WorldGenerationError(
                f"could only derive {len(got)} of {target} {hops}-hop questions; "
                "ask for fewer or generate a larger world"
            )
        questions += got
    rng.shuffle(questions)
    width = len(str(len(questions) - 1))

    owners: dict[str, set[str]] = {}
    for holder in world.entities:
        for label in (holder.name, *holder.aliases):
            owners.setdefault(label.casefold(), set()).add(holder.id)
    clash = {i for ids in owners.values() if len(ids) > 1 for i in ids}
    edge_by_id = {edge.id: edge for edge in world.edges}

    def numbered(i: int, q: Question) -> Question:
        who = tuple((names[a], *entities[a].aliases) for a in q.answers)
        unclear = (
            reader.ambiguous(q.start, q.gold_paths[0], q.answers)
            if reader and q.type in {"entity", "set", "count"} and q.gold_paths
            else None
        )
        ends = {
            x for f in q.supporting_facts for x in (edge_by_id[f].subject, edge_by_id[f].object)
        }
        return Question(
            **{
                **vars(q),
                "id": f"Q{i:0{width}d}",
                "answer_names": who,
                "shared_name": bool(ends & clash),
                "ambiguous_under_names": unclear,
                "persona": spec.persona if spec.persona != "none" else None,
            }
        )

    return tuple(numbered(i, q) for i, q in enumerate(questions))
