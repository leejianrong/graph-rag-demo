"""Render world-model facts as documents from templates: deterministic, offline, no LLM.

Every fact becomes one sentence, and each sentence records the fact it expresses and where each
entity is mentioned (character offsets into the document), so extraction, linking and retrieval
can be scored against exact gold. Facts are grouped by subject, so a multi-hop answer needs
several documents. Entities appear under their canonical name or, at ``alias_use_rate``, an
alias. Documents about distractor entities are flagged and never mention a real entity.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from kggen.spec import VAGUE_PREDICATE
from kggen.world import Edge, World

# (present, past) template pairs per relation; several variants give some lexical variety.
TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "LOCATED_IN": (
        ("{S} is located in {O}", "{S} was located in {O}"),
        ("{S} is based in {O}", "{S} was based in {O}"),
    ),
    "PART_OF": (
        ("{S} is part of {O}", "{S} was part of {O}"),
        ("{S} belongs to {O}", "{S} belonged to {O}"),
    ),
    "MEMBER_OF": (
        ("{S} is a member of {O}", "{S} was a member of {O}"),
        ("{S} sits among the members of {O}", "{S} sat among the members of {O}"),
    ),
    "WORKS_FOR": (
        ("{S} works for {O}", "{S} worked for {O}"),
        ("{S} is employed by {O}", "{S} was employed by {O}"),
    ),
    "FOUNDED": (
        ("{S} founded {O}", "{S} founded {O}"),
        ("{S} is the founder of {O}", "{S} was the founder of {O}"),
    ),
    "OWNS": (
        ("{S} owns {O}", "{S} owned {O}"),
        ("{S} is the owner of {O}", "{S} was the owner of {O}"),
    ),
    "PRODUCES": (
        ("{S} produces {O}", "{S} produced {O}"),
        ("{S} manufactures {O}", "{S} manufactured {O}"),
    ),
    "PARTICIPATED_IN": (
        ("{S} participated in {O}", "{S} participated in {O}"),
        ("{S} took part in {O}", "{S} took part in {O}"),
    ),
    "AFFILIATED_WITH": (
        ("{S} is affiliated with {O}", "{S} was affiliated with {O}"),
        ("{S} is associated with {O}", "{S} was associated with {O}"),
    ),
    VAGUE_PREDICATE: (
        ("{S} is connected to {O}", "{S} was connected to {O}"),
        ("{S} has ties to {O}", "{S} had ties to {O}"),
    ),
}
_SLOT = re.compile(r"(\{S\}|\{O\})")


class RenderSpec(BaseModel):
    """How facts are turned into documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    max_facts_per_doc: int = Field(default=6, ge=1)
    alias_use_rate: float = Field(
        default=0.5, ge=0, le=1, description="Chance a mention uses an alias, when one exists."
    )


@dataclass(frozen=True)
class Mention:
    entity_id: str
    surface: str
    start: int  # character offsets into the document text
    end: int


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int
    fact_ids: tuple[str, ...]
    mentions: tuple[Mention, ...]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    sentences: tuple[Sentence, ...]
    distractor: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "distractor": self.distractor,
            "sentences": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "fact_ids": list(s.fact_ids),
                    "mentions": [vars(m) for m in s.mentions],
                }
                for s in self.sentences
            ],
        }


def render_documents(world: World, spec: RenderSpec | None = None) -> tuple[Document, ...]:
    """One or more documents per subject entity, covering every fact exactly once."""
    spec = spec or RenderSpec()
    rng = random.Random(f"{world.spec.seed}:{spec.seed}:render")
    entities = {e.id: e for e in world.entities}
    start_year = world.spec.start_year

    by_subject: dict[str, list[Edge]] = {}
    for edge in world.edges:  # already ordered by subject, then fact id order
        by_subject.setdefault(edge.subject, []).append(edge)

    def surface(entity_id: str) -> str:
        entity = entities[entity_id]
        if entity.aliases and rng.random() < spec.alias_use_rate:
            return rng.choice(entity.aliases)
        return entity.name

    def sentence(edge: Edge, offset: int) -> Sentence:
        present, past = rng.choice(TEMPLATES[edge.predicate])
        ended = edge.valid_to is not None
        template, suffix = (past, f" until {edge.valid_to}") if ended else (present, "")
        if edge.valid_from > start_year:
            suffix = f" since {edge.valid_from}"
        text, mentions = "", []
        for piece in _SLOT.split(template):
            if piece in ("{S}", "{O}"):
                entity_id = edge.subject if piece == "{S}" else edge.object
                name = surface(entity_id)
                mentions.append(
                    Mention(entity_id, name, offset + len(text), offset + len(text) + len(name))
                )
                text += name
            else:
                text += piece
        text += suffix + "."
        return Sentence(text, offset, offset + len(text), (edge.id,), tuple(mentions))

    documents: list[Document] = []
    for subject in sorted(by_subject):
        facts = by_subject[subject]
        for i in range(0, len(facts), spec.max_facts_per_doc):
            sentences: list[Sentence] = []
            offset = 0
            for edge in facts[i : i + spec.max_facts_per_doc]:
                s = sentence(edge, offset)
                sentences.append(s)
                offset = s.end + 1  # a single space joins sentences
            documents.append(
                Document(
                    id=f"D{len(documents):05d}",
                    title=entities[subject].name,
                    text=" ".join(s.text for s in sentences),
                    sentences=tuple(sentences),
                    distractor=entities[subject].distractor,
                )
            )
    return tuple(documents)
