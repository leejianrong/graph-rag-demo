"""Render world-model facts as documents from templates: deterministic, offline, no LLM.

Every fact becomes one sentence, and each sentence records the fact it expresses and where each
entity is mentioned (character offsets into the document), so extraction, linking and retrieval
can be scored against exact gold. Facts are grouped by subject, so a multi-hop answer needs
several documents. Entities appear under their canonical name or, at ``alias_use_rate``, an
alias. Documents about distractor entities are flagged and never mention a real entity.

With ``target_words`` set, a document is a realistic article of about that many words: the
subject's facts and a few facts about connected entities are spread through paragraphs of filler
prose (``kggen.filler``) that states no fact and names no entity except, now and then, the
subject, which is recorded as a mention. Every fact is still expressed with exact offsets.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from kggen.filler import filler_sentence
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
_SLOT_S = re.compile(r"(\{S\})")


class RenderSpec(BaseModel):
    """How facts are turned into documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    max_facts_per_doc: int = Field(default=6, ge=1)
    alias_use_rate: float = Field(
        default=0.5, ge=0, le=1, description="Chance a mention uses an alias, when one exists."
    )
    target_words: int = Field(
        default=0,
        ge=0,
        le=20_000,
        description="Words per document; 0 keeps the short fact-only documents.",
    )
    neighbour_facts: int = Field(
        default=3,
        ge=0,
        le=50,
        description="Facts about connected entities added to a long document as context.",
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
    subject: str = ""  # the entity id the document is about

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "subject": self.subject,
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

    def shifted(s: Sentence, delta: int) -> Sentence:
        moved = tuple(
            Mention(m.entity_id, m.surface, m.start + delta, m.end + delta) for m in s.mentions
        )
        return Sentence(s.text, s.start + delta, s.end + delta, s.fact_ids, moved)

    def filler(subject: str, used: set[str]) -> Sentence:
        """A filler sentence, naming the subject (a mention with no fact) when its frame does."""
        entity = entities[subject]
        text, mentions = "", []
        for piece in _SLOT_S.split(filler_sentence(rng, entity.type, avoid=used)):
            if piece == "{S}":
                name = surface(subject)
                mentions.append(Mention(subject, name, len(text), len(text) + len(name)))
                text += name
            else:
                text += piece
        return Sentence(text, 0, len(text), (), tuple(mentions))

    def article(subject: str, own: list[Edge]) -> tuple[list[Sentence], list[str]]:
        """Sentences and the separator after each, for one long document about ``subject``."""
        entity = entities[subject]
        context: list[Edge] = []
        if not entity.distractor and spec.neighbour_facts:
            pool = {
                f.id: f for e in own for f in by_subject.get(e.object, []) if f.subject != subject
            }
            ids = sorted(pool)
            context = [pool[i] for i in rng.sample(ids, min(len(ids), spec.neighbour_facts))]
        facts = [sentence(e, 0) for e in (*own, *context)]
        rng.shuffle(facts)
        words = sum(len(f.text.split()) for f in facts)
        used: set[str] = set()
        padding = [filler(subject, used)]
        while not any(m.entity_id == subject for m in padding[0].mentions):
            padding[0] = filler(subject, used)  # open by naming the subject, as an article would
        words += len(padding[0].text.split())
        while words < spec.target_words:
            padding.append(filler(subject, used))
            words += len(padding[-1].text.split())
        slots = sorted(rng.sample(range(1, len(padding) + len(facts)), len(facts)))
        merged: list[Sentence] = []
        pad, fact = iter(padding), iter(facts)
        for i in range(len(padding) + len(facts)):
            merged.append(next(fact) if i in slots else next(pad))
        out, seps, offset, run = [], [], 0, 0
        for s in merged:
            out.append(shifted(s, offset))
            run += 1
            if run >= 4 and rng.random() < 0.3:
                seps.append("\n\n")
                run = 0
            else:
                seps.append(" ")
            offset = out[-1].end + len(seps[-1])
        seps[-1] = ""
        return out, seps

    documents: list[Document] = []
    for subject in sorted(by_subject):
        facts = by_subject[subject]
        if spec.target_words:
            for i in range(0, len(facts), spec.max_facts_per_doc):
                sentences, seps = article(subject, facts[i : i + spec.max_facts_per_doc])
                text = "".join(s.text + sep for s, sep in zip(sentences, seps, strict=True))
                documents.append(
                    Document(
                        id=f"D{len(documents):05d}",
                        title=entities[subject].name,
                        text=text,
                        sentences=tuple(sentences),
                        distractor=entities[subject].distractor,
                        subject=subject,
                    )
                )
            continue
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
                    subject=subject,
                )
            )
    return tuple(documents)
