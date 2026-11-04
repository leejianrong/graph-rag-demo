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

from kggen.llm import CachedChat, Messages
from kggen.render import Document, Mention, RenderSpec, Sentence
from kggen.world import Edge, World

PROMPT_VERSION = 1

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
}


class InvalidRendering(ValueError):
    """The model's answer did not express every fact faithfully."""


def _relation(predicate: str) -> str:
    return _READABLE.get(predicate, "is connected to")


def build_messages(facts: list[tuple[Edge, str, str]], start_year: int) -> Messages:
    lines = []
    for i, (edge, subject, obj) in enumerate(facts, 1):
        when = ""
        if edge.valid_from > start_year:
            when += f" (since {edge.valid_from})"
        if edge.valid_to is not None:
            when += f" (ended {edge.valid_to}; use the past tense)"
        lines.append(f"F{i}: {subject} | {_relation(edge.predicate)} | {obj}{when}")
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
    world: World, chat: CachedChat, spec: RenderSpec | None = None
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

    documents: list[Document] = []
    for subject in sorted(by_subject):
        edges = by_subject[subject]
        for i in range(0, len(edges), spec.max_facts_per_doc):
            chunk = edges[i : i + spec.max_facts_per_doc]
            facts = [(e, surface(e.subject), surface(e.object)) for e in chunk]
            messages = build_messages(facts, start_year)
            raw = chat.ask(messages)
            texts = parse_response(raw, facts, start_year)
            chat.store(messages, raw, prompt_version=PROMPT_VERSION)  # only once it is valid
            sentences, offset = [], 0
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
                offset += len(text) + 1  # a single space joins sentences
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
