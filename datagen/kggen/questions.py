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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kggen.spec import VAGUE_PREDICATE
from kggen.world import Edge, World, WorldGenerationError, apportion

MAX_HOPS = 4
MIN_SET = 2
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
    grading: str = "exact"  # exact | set_f1 | numeric | abstain
    abstain: dict | None = None  # {expected, reason}: the right reply is to decline
    premise: dict | None = (
        None  # a false premise: {asserted_fact, contradicting_fact_ids, correction}
    )

    def to_dict(self) -> dict:
        return {
            **vars(self),
            "accepted_answers": list(self.accepted_answers),
            "supporting_facts": list(self.supporting_facts),
            "answers": list(self.answers),
            "gold_paths": [list(p) for p in self.gold_paths],
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


def generate_questions(world: World, spec: QuestionSpec | None = None) -> tuple[Question, ...]:
    """Derive ``spec.n_questions`` questions from the world's final state, by ``spec.type_mix``."""
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
            seen.add(("entity", start, facts))
            answer = entities[at]
            found.append(
                Question(
                    id="",
                    question=f"Name {_noun_phrase(names, types, start, steps)}.",
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
            inner = _noun_phrase(names, types, start, prefix)
            label = TYPE_LABEL[types[members[0]]]
            plural = PLURAL[types[members[0]]]
            verb = VERB[last.predicate]
            tail = f"{inner} {verb}" if forward else f"{verb} {inner}"
            text = (
                f"List every {label} that {tail}."
                if kind == "set"
                else f"Count the {plural} that {tail}."
            )
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
        return f"{plural} that it {verb}" if forward else f"{plural} that {verb} it"

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
                    question=(
                        f"Which of {names[first]} and {names[second]} has more "
                        f"{phrase_of(predicate, forward, plural)}?"
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
        k: {t for t in map(types.get, m)} for k, m in counts.items()
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
                    question=f"Name the {label} that {tail}.",
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
            asked = _noun_phrase(names, types, fake, [(second, second_forward)])
            found.append(
                Question(
                    id="",
                    question=f"{claim} Name {asked}.",
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

    quota = apportion(spec.n_questions, spec.type_mix)
    questions: list[Question] = []
    short = 0  # questions another type could not supply; plain entity questions make them up
    for kind in ("set", "count"):
        wanted = apportion(quota.get(kind, 0), {str(h): w for h, w in spec.hop_mix.items()})
        for hops_key in sorted(wanted):
            got = set_questions(int(hops_key), wanted[hops_key], kind)
            questions += got
            short += wanted[hops_key] - len(got)
    for kind, make in (
        ("compare", compare_questions),
        ("superlative", superlative_questions),
        ("unanswerable", unanswerable_questions),
        ("false_premise", false_premise_questions),
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
    return tuple(Question(**{**vars(q), "id": f"Q{i:0{width}d}"}) for i, q in enumerate(questions))
