"""Multi-hop questions with gold paths, derived from the world model.

A question starts at one named entity and follows 1 to 4 typed relations; the answer is the
entity at the end. Questions come from the world's facts, never from any pipeline's output,
and every question carries the ids of the facts that support it (its gold path). A question
is kept only if its answer is unique: following the same relations from the start entity
reaches exactly one entity in the world's final state.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kggen.spec import VAGUE_PREDICATE
from kggen.world import Edge, World, WorldGenerationError, apportion

MAX_HOPS = 4

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


class QuestionSpec(BaseModel):
    """How many questions to derive and how many hops each should take."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    n_questions: int = Field(default=50, ge=1)
    hop_mix: dict[int, float] = Field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        description="Relative weight of questions with each hop count (1 to 4).",
    )

    @model_validator(mode="after")
    def _check(self) -> QuestionSpec:
        if not self.hop_mix or set(self.hop_mix) - set(range(1, MAX_HOPS + 1)):
            raise ValueError(f"hop_mix keys must be within 1..{MAX_HOPS}")
        if any(w < 0 for w in self.hop_mix.values()) or sum(self.hop_mix.values()) <= 0:
            raise ValueError("hop_mix needs non-negative weights with a positive sum")
        return self


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

    def to_dict(self) -> dict:
        return {
            **vars(self),
            "accepted_answers": list(self.accepted_answers),
            "supporting_facts": list(self.supporting_facts),
        }


def _noun_phrase(world_names: dict[str, str], types: dict[str, str], start: str, steps) -> str:
    """The nested phrase naming the entity reached after ``steps`` from ``start``."""
    phrase = world_names[start]
    for edge, forward in steps:
        verb = VERB[edge.predicate]
        target = edge.object if forward else edge.subject
        label = TYPE_LABEL[types[target]]
        if forward:
            phrase = f"the {label} that {phrase} {verb}"
        else:
            phrase = f"the {label} that {verb} {phrase}"
    return phrase


def generate_questions(world: World, spec: QuestionSpec | None = None) -> tuple[Question, ...]:
    """Derive ``spec.n_questions`` unique-answer questions from the world's final state."""
    spec = spec or QuestionSpec()
    rng = random.Random(f"{world.spec.seed}:{spec.seed}:questions")
    entities = {e.id: e for e in world.entities}
    names = {i: e.name for i, e in entities.items()}
    types = {i: e.type for i, e in entities.items()}
    name_count = Counter(e.name for e in world.entities)

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

    def reach(frontier: set[str], predicate: str, forward: bool) -> set[str]:
        index = out_edges if forward else in_edges
        return {
            (e.object if forward else e.subject)
            for node in frontier
            for e in index.get(node, [])
            if e.predicate == predicate
        }

    starts = sorted(i for i in set(out_edges) | set(in_edges) if name_count[names[i]] == 1)
    if not starts:
        raise WorldGenerationError("no entity has a unique name and a typed fact to ask about")

    wanted = apportion(spec.n_questions, {str(h): w for h, w in spec.hop_mix.items()})
    questions: list[Question] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for hops_key in sorted(wanted):
        hops, target = int(hops_key), wanted[hops_key]
        found, budget = 0, 400 * target + 2000
        while found < target and budget > 0:
            budget -= 1
            start = rng.choice(starts)
            steps: list[tuple[Edge, bool]] = []
            at, visited = start, {start}
            for _ in range(hops):
                options = [(e, True) for e in out_edges.get(at, [])] + [
                    (e, False) for e in in_edges.get(at, [])
                ]
                options = [
                    (e, f) for e, f in options if (e.object if f else e.subject) not in visited
                ]
                if not options:
                    break
                edge, forward = rng.choice(options)
                at = edge.object if forward else edge.subject
                visited.add(at)
                steps.append((edge, forward))
            if len(steps) < hops:
                continue
            frontier = {start}
            for edge, forward in steps:
                frontier = reach(frontier, edge.predicate, forward)
            facts = tuple(e.id for e, _ in steps)
            if frontier != {at} or (start, facts) in seen:
                continue
            seen.add((start, facts))
            answer = entities[at]
            text = f"Name {_noun_phrase(names, types, start, steps)}."
            questions.append(
                Question(
                    id="",
                    question=text,
                    hops=hops,
                    start=start,
                    answer=at,
                    answer_name=answer.name,
                    accepted_answers=(answer.name, *answer.aliases),
                    supporting_facts=facts,
                )
            )
            found += 1
        if found < target:
            raise WorldGenerationError(
                f"could only derive {found} of {target} {hops}-hop questions with a unique answer; "
                "ask for fewer or generate a larger world"
            )
    rng.shuffle(questions)
    width = len(str(len(questions) - 1))
    return tuple(Question(**{**vars(q), "id": f"Q{i:0{width}d}"}) for i, q in enumerate(questions))
