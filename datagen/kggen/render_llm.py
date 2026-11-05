"""Render documents with an LLM for realistic prose, while keeping exact gold.

One request per document. The prompt lists the facts as ``F1..Fn`` with the exact names to use, and
the model answers with one sentence per fact. Nothing the model says is trusted: every fact must
get exactly one sentence, both entity names must appear verbatim, and a fact's dates must appear.
Mention offsets are then found by this module, never reported by the model, so the gold is as exact
as the template renderer's. An invalid response is not cached and raises ``InvalidRendering``.
"""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat, Messages
from kggen.render import Document, Mention, RenderSpec, Sentence
from kggen.world import Edge, World

PROMPT_VERSION = 1
RICH_PROMPT_VERSION = 2

SYSTEM = (
    "You write short, factual encyclopedia-style paragraphs from structured facts. "
    "Reply with JSON only, no commentary and no code fences."
)

_READABLE = {  # how to say each relation in the prompt; the model words it freely
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


class InvalidRendering(ValueError):
    """The model's answer did not express every fact faithfully."""


RICH_SYSTEM = (
    "You write short, factual encyclopedia-style articles from structured facts. "
    "Reply with JSON only, no commentary and no code fences."
)


def _relation(predicate: str) -> str:
    return _READABLE.get(predicate, "is connected to")


def build_messages(
    facts: list[tuple[Edge, str, str]], start_year: int, style: str = "plain"
) -> Messages:
    lines = []
    for i, (edge, subject, obj) in enumerate(facts, 1):
        when = ""
        if edge.valid_from > start_year:
            when += f" (since {edge.valid_from})"
        if edge.valid_to is not None:
            when += f" (ended {edge.valid_to}; use the past tense)"
        lines.append(f"F{i}: {subject} | {_relation(edge.predicate)} | {obj}{when}")
    if style == "rich":
        user = (
            "Write a short article that states every fact below exactly once. You may put several "
            "facts in one sentence, reorder them, and, once an entity has been named in the "
            'article, refer back to it with a pronoun or a description ("the company", "the '
            'city"). Name each entity exactly as written the first time. A sentence that states '
            "a fact must contain that fact's object name exactly as written, and its subject "
            "must have been named exactly as written in that sentence or an earlier one. State "
            "the years given for a fact in its sentence. Add two or three background sentences "
            "of general context that name no person, organisation, place or event. Do not add "
            "facts, merge distinct facts into one claim, or name anything that is not listed.\n\n"
            + "\n".join(lines)
            + '\n\nReply as JSON in reading order: {"sentences": [{"facts": ["F1"], '
            '"text": "..."}, {"facts": [], "text": "background sentence"}, ...]}'
        )
        return [{"role": "system", "content": RICH_SYSTEM}, {"role": "user", "content": user}]
    user = (
        "Write one sentence per fact below, in the same order, in varied natural prose. "
        "Each sentence must contain the subject name and the object name exactly as written, "
        "and must state the years given for its fact. Do not add facts, merge facts, or "
        "mention anything that is not listed.\n\n"
        + "\n".join(lines)
        + '\n\nReply as JSON: {"sentences": [{"fact": "F1", "text": "..."}, ...]}'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_response(raw: str, facts: list[tuple[Edge, str, str]], start_year: int) -> list[str]:
    """The sentence texts, in fact order, or ``InvalidRendering`` saying what is wrong."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*|\s*```$", "", text)
    try:
        items = json.loads(text)["sentences"]
        labels = [i["fact"] for i in items]
        texts = [str(i["text"]).strip() for i in items]
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidRendering(f"not the requested JSON: {error}") from error
    expected = [f"F{n}" for n in range(1, len(facts) + 1)]
    if labels != expected:
        raise InvalidRendering(f"facts {labels} do not match {expected}")
    for (edge, subject, obj), sentence in zip(facts, texts, strict=True):
        spans = _find_spans(sentence, subject, obj)
        if spans is None:
            raise InvalidRendering(f"{edge.id}: names {subject!r} and {obj!r} not both present")
        years = [edge.valid_to] if edge.valid_to is not None else []
        years += [edge.valid_from] if edge.valid_from > start_year else []
        if missing := [y for y in years if str(y) not in sentence]:
            raise InvalidRendering(f"{edge.id}: year {missing[0]} missing from {sentence!r}")
    return texts


def parse_rich(
    raw: str,
    facts: list[tuple[Edge, str, str]],
    start_year: int,
    unlisted: list[str],
) -> list[tuple[str, tuple[int, ...]]]:
    """``(text, fact indices)`` per sentence in reading order, or ``InvalidRendering``."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*|\s*```$", "", text)
    try:
        items = json.loads(text)["sentences"]
        parsed = [(str(i["text"]).strip(), [str(f) for f in i["facts"]]) for i in items]
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidRendering(f"not the requested JSON: {error}") from error
    labels = [f for _, fs in parsed for f in fs]
    expected = [f"F{n}" for n in range(1, len(facts) + 1)]
    if sorted(labels) != sorted(expected):
        raise InvalidRendering(f"each of {expected} must appear exactly once, got {labels}")
    out, before = [], ""
    for sentence, fact_labels in parsed:
        if not sentence:
            raise InvalidRendering("an empty sentence")
        # a listed name may contain another entity's name as a word ("Zobi Labs"), so mask first
        bare = sentence
        for listed in sorted({n for _, a, b in facts for n in (a, b)}, key=len, reverse=True):
            bare = re.sub(rf"(?<!\w){re.escape(listed)}(?!\w)", " " * len(listed), bare)
        for name in unlisted:
            if _spans(bare, name):
                raise InvalidRendering(f"{name!r} is in the world but not in the facts")
        indices = tuple(int(label[1:]) - 1 for label in fact_labels)
        for k in indices:
            edge, subject, obj = facts[k]
            if not _spans(sentence, obj):
                raise InvalidRendering(f"{edge.id}: object {obj!r} not in {sentence!r}")
            if not _spans(before + " " + sentence, subject):
                raise InvalidRendering(f"{edge.id}: subject {subject!r} not named yet")
            years = [edge.valid_to] if edge.valid_to is not None else []
            years += [edge.valid_from] if edge.valid_from > start_year else []
            if missing := [y for y in years if str(y) not in sentence]:
                raise InvalidRendering(f"{edge.id}: year {missing[0]} missing from {sentence!r}")
        before += " " + sentence
        out.append((sentence, indices))
    return out


def _spans(sentence: str, name: str) -> list[tuple[int, int]]:
    pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
    return [m.span() for m in re.finditer(pattern, sentence)]


def _mentions(sentence: str, offset: int, surfaces: dict[str, str]) -> tuple[Mention, ...]:
    """Every whole-word occurrence of a listed surface, longest first so none overlaps another."""
    taken: list[tuple[int, int]] = []
    found = []
    for name in sorted(surfaces, key=len, reverse=True):
        for a, b in _spans(sentence, name):
            if all(b <= x or a >= y for x, y in taken):
                taken.append((a, b))
                found.append(Mention(surfaces[name], name, offset + a, offset + b))
    return tuple(sorted(found, key=lambda m: m.start))


def _find_spans(
    sentence: str, subject: str, obj: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Whole-word, non-overlapping spans of the subject and object names in ``sentence``."""

    def occurrences(name: str) -> list[tuple[int, int]]:
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        return [m.span() for m in re.finditer(pattern, sentence)]

    for s in occurrences(subject):
        for o in occurrences(obj):
            if s[1] <= o[0] or o[1] <= s[0]:
                return s, o
    return None


def render_documents_llm(
    world: World,
    chat: CachedChat,
    spec: RenderSpec | None = None,
    style: str = "plain",
    workers: int = 1,
) -> tuple[Document, ...]:
    """Documents grouped like ``render_documents`` (every fact once), but written by the model."""
    spec = spec or RenderSpec()
    rng = random.Random(f"{world.spec.seed}:{spec.seed}:render-llm")
    entities = {e.id: e for e in world.entities}
    start_year = world.spec.start_year

    def surface(entity_id: str) -> str:
        entity = entities[entity_id]
        if entity.aliases and rng.random() < spec.alias_use_rate:
            return rng.choice(entity.aliases)
        return entity.name

    by_subject: dict[str, list[Edge]] = {}
    for edge in world.edges:
        by_subject.setdefault(edge.subject, []).append(edge)

    names = {n for e in world.entities for n in (e.name, *e.aliases)}
    jobs = []  # (subject, facts, messages), drawn in a fixed order so surfaces stay reproducible
    for subject in sorted(by_subject):
        edges = by_subject[subject]
        for i in range(0, len(edges), spec.max_facts_per_doc):
            chunk = edges[i : i + spec.max_facts_per_doc]
            facts = [(e, surface(e.subject), surface(e.object)) for e in chunk]
            jobs.append((subject, facts, build_messages(facts, start_year, style)))

    def write(job: tuple) -> list[Sentence]:
        _, facts, messages = job
        raw = chat.ask(messages)
        sentences: list[Sentence] = []
        offset = 0
        if style == "rich":
            surfaces = {
                n: e for edge, a, b in facts for n, e in ((a, edge.subject), (b, edge.object))
            }
            for text, indices in parse_rich(raw, facts, start_year, sorted(names - set(surfaces))):
                ids = tuple(facts[k][0].id for k in indices)
                sentences.append(
                    Sentence(
                        text, offset, offset + len(text), ids, _mentions(text, offset, surfaces)
                    )
                )
                offset += len(text) + 1  # a single space joins sentences
            chat.store(messages, raw, prompt_version=RICH_PROMPT_VERSION)
            return sentences
        texts = parse_response(raw, facts, start_year)
        chat.store(messages, raw, prompt_version=PROMPT_VERSION)  # only once it is valid
        for (edge, name_s, name_o), text in zip(facts, texts, strict=True):
            span_s, span_o = _find_spans(text, name_s, name_o)  # type: ignore[misc]
            mentions = tuple(
                Mention(entity, name, offset + a, offset + b)
                for entity, name, (a, b) in sorted(
                    [(edge.subject, name_s, span_s), (edge.object, name_o, span_o)],
                    key=lambda m: m[2],
                )
            )
            sentences.append(Sentence(text, offset, offset + len(text), (edge.id,), mentions))
            offset += len(text) + 1
        return sentences

    # A recording endpoint holds each call until someone answers it, so ask several at once.
    # Every valid answer is stored as it arrives, so a rejected one costs only itself.
    with ThreadPoolExecutor(workers) as pool:
        written = list(pool.map(write, jobs))
    documents: list[Document] = []
    for (subject, _, _), sentences in zip(jobs, written, strict=True):
        documents.append(
            Document(
                id=f"D{len(documents):05d}",
                title=entities[subject].name,
                text=" ".join(s.text for s in sentences),
                sentences=tuple(sentences),
                distractor=entities[subject].distractor,
                subject=subject,
            )
        )
    return tuple(documents)
