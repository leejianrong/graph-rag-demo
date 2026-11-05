"""Harder labelled cases for the LLM judge, with the answers written by a model (GRA-118).

``judge.build_cases`` makes plain templated answers, and a judge that scores 0 errors on them proves
little. Here a writer model is asked for each answer in natural prose, one kind at a time, and the
label is the kind's definition: what the writer was told to do. Nothing the writer returns is
trusted blindly: each answer must name the entities it is meant to name and must not name the
ones it must not, or it is rejected (and not cached). A reversal that also names the subject
before the object in some sentence is rejected too: it has stated the fact. Whether the writer
also followed the harder instructions (a real reversal, a real denial) is not checkable by code,
so the cases are meant to be read; ``docs/results/judge-hard.md`` says how many were.

Kinds (the label is what the judge should answer to "does the answer state this fact?"):

* ``paraphrase``, ``buried``, ``attributed``: the fact is stated (yes), in other words, inside a
  five-sentence answer, or as something a source reports. Counting an attributed assertion as
  stated is a labelling choice.
* ``reversed``: the relation is asserted with subject and object swapped (no). Only for relations
  that are not symmetric.
* ``denied``, ``uncertain``: the answer says it is not so, or that it is unclear (no).
* ``near_miss``: the relation is asserted with another entity as object (no).
* ``other_only``: the answer states three other facts (no).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat
from kggen.world import World

from kgbench.text import mentions, sentences_of

HARD_KINDS = {
    "paraphrase": True,
    "buried": True,
    "attributed": True,
    "reversed": False,
    "denied": False,
    "uncertain": False,
    "near_miss": False,
    "other_only": False,
}
_SYMMETRIC = {"AFFILIATED_WITH"}
_SYSTEM = (
    "You write short test passages. Reply with the passage only: no title, no quotes, "
    "no commentary."
)


def _relation(fact: dict) -> str | None:
    text, subject, obj = fact["text"], fact["subject_names"][0], fact["object_names"][0]
    if text.startswith(subject) and text.endswith(obj):
        return text[len(subject) : len(text) - len(obj)].strip() or None
    return None


def _instruction(kind: str, fact: dict, other: dict, extra: list[dict]) -> str:
    text, subject, obj = fact["text"], fact["subject_names"][0], fact["object_names"][0]
    relation = _relation(fact) or ""
    other_obj = other["object_names"][0]
    return {
        "paraphrase": f"Write one or two sentences that state this fact in your own words, "
        f"without copying its phrasing (use the two names exactly): {text}",
        "buried": f"Write a five-sentence paragraph. The third sentence states this fact: {text}. "
        "The other four sentences are generic background and name no person, organisation, place "
        "or event.",
        "attributed": f'Write one or two sentences in which a source ("according to the filing", '
        f'"the records say") reports this fact, and the passage presents it as true: {text}',
        "reversed": f"Write one or two sentences asserting that {obj} {relation} {subject}. That "
        f"is the reverse of '{text}'; assert the reverse as if it were true, naming both.",
        "denied": f"Write one or two sentences that say this is NOT the case, naming both "
        f"entities: {text}",
        "uncertain": f"Write one or two sentences that say it is unclear or disputed whether "
        f"this holds, neither confirming nor denying it: {text}",
        "near_miss": f"Write one or two sentences asserting that {subject} {relation} {other_obj}. "
        f"State only that, naming both entities.",
        "other_only": "Write a short paragraph that states these facts and nothing else: "
        + "; ".join(f["text"] for f in extra),
    }[kind]


def _checks(kind: str, fact: dict, other: dict, extra: list[dict]):
    """(names every one of which must appear, names none of which may appear)."""
    subject, obj = fact["subject_names"], fact["object_names"]
    if kind == "near_miss":
        return [subject, other["object_names"]], [obj]
    if kind == "other_only":
        return [], [obj]
    return [subject, obj], []


def _restates(answer: str, fact: dict) -> bool:
    """Whether some sentence names the subject before the object, as the true fact does. A reversal
    that does so somewhere has also stated the fact (the check is crude and drops good reversals
    too; it errs towards a clean label)."""
    subject, obj = fact["subject_names"][0], fact["object_names"][0]
    for sentence in sentences_of(answer):
        i, j = sentence.find(subject), sentence.find(obj)
        if 0 <= i < j:
            return True
    return False


def _pool(world: World, questions: Sequence) -> list[dict]:
    predicate = {e.id: e.predicate for e in world.edges}
    pool, seen = [], set()
    for q in questions:
        for fact in q.required_facts:
            usable = (
                fact["fact_id"] not in seen
                and _relation(fact)
                and predicate[fact["fact_id"]] not in _SYMMETRIC
            )
            if usable:
                seen.add(fact["fact_id"])
                pool.append({**fact, "predicate": predicate[fact["fact_id"]]})
    return pool


def build_hard_cases(
    world: World,
    questions: Sequence,
    chat: CachedChat,
    facts: int = 14,
    workers: int = 8,
) -> tuple[list[dict], int]:
    """Labelled cases written by ``chat``'s model, and how many answers were rejected."""
    pool = _pool(world, questions)[:facts]
    jobs = []
    for i, fact in enumerate(pool):
        other = pool[(i + 3) % len(pool)]
        if other["object_names"][0] == fact["object_names"][0]:
            other = pool[(i + 4) % len(pool)]
        extra = [pool[(i + k) % len(pool)] for k in (5, 6, 7)]
        extra = [f for f in extra if f["object_names"][0] != fact["object_names"][0]]
        for kind in HARD_KINDS:
            if kind == "near_miss" and other["predicate"] != fact["predicate"]:
                continue  # the swapped-in object must fit the relation
            jobs.append((kind, fact, other, extra))

    def write(job: tuple) -> dict | None:
        kind, fact, other, extra = job
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _instruction(kind, fact, other, extra)},
        ]
        answer = chat.ask(messages).strip()
        must, never = _checks(kind, fact, other, extra)
        if (
            not answer
            or not all(mentions(answer, n) for n in must)
            or any(mentions(answer, n) for n in never)
            or (kind == "reversed" and _restates(answer, fact))
        ):
            return None
        chat.store(messages, answer)
        keep = ("fact_id", "text", "subject_names", "object_names", "weight")
        return {
            "id": f"{fact['fact_id']}-{kind}",
            "kind": kind,
            "fact": {k: fact[k] for k in keep},
            "answer": answer,
            "label": HARD_KINDS[kind],
        }

    with ThreadPoolExecutor(workers) as pool_:
        written = list(pool_.map(write, jobs))
    return [c for c in written if c], sum(c is None for c in written)
