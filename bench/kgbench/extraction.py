"""Score extraction against the generator's gold: mentions, entities and triples (ADR-0011).

Gold comes from a ``kggen`` world and its rendered documents, never from the pipeline under test.
The scorer works on plain data (``Extraction``), so a test can score a perfect or a deliberately bad
extractor without a pipeline; ``run_pipeline`` makes an ``Extraction`` from a real or offline one.

* **Mentions**: a predicted mention is right when its exact character span is a gold mention.
* **Entities**: each predicted canonical entity is aligned to the gold entity most of its
  span-matched mentions refer to. TP is the number of distinct gold entities recovered this way, out
  of the canonicals predicted (so duplicates and ungrounded ones lower precision) and the gold
  entities (so merged homonyms lower recall). Cluster quality proper is the linking score (GRA-81).
* **Triples**: predicted ``(subject, predicate, object)`` over canonical ids are mapped through that
  alignment. *Strict* needs the same subject, predicate and object; *edge* only the same pair of
  entities, ignoring predicate and direction, which shows whether a relation was found even when it
  was not named. Both are reported per predicate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from kggen.render import Document
from kggen.world import World

from graph_rag.models import IngestTrigger

Span = tuple[str, int, int]  # (document id, start, end)
Triple = tuple[str, str, str]  # (subject, predicate, object)


@dataclass(frozen=True)
class Gold:
    mentions: dict[Span, str]  # span -> entity id
    entities: frozenset[str]  # entities mentioned at least once
    triples: frozenset[Triple]  # the facts the documents express


@dataclass(frozen=True)
class PredictedMention:
    doc: str
    start: int
    end: int
    canonical_id: str | None  # None when the pipeline linked it to nothing


@dataclass(frozen=True)
class Extraction:
    mentions: tuple[PredictedMention, ...]
    triples: tuple[Triple, ...]  # over the pipeline's own canonical ids


@dataclass(frozen=True)
class PRF:
    """Precision is ``correct / predicted``; recall is ``found / gold``.

    They differ when several predictions land on one gold item (duplicates), so recall can never
    exceed 1 and duplicates cost precision only.
    """

    correct: int  # predictions that are right
    predicted: int
    found: int  # gold items that some prediction recovered
    gold: int

    @classmethod
    def of_sets(cls, hits: int, predicted: int, gold: int) -> PRF:
        return cls(hits, predicted, hits, gold)

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.found / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "found": self.found,
            "gold": self.gold,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class PredicateScore:
    gold: int  # gold facts with this predicate
    predicted: int  # predicted triples with this predicate
    strict: PRF
    edge_recall: float  # gold facts whose entity pair was found under any predicate


@dataclass(frozen=True)
class ExtractionReport:
    mentions: PRF
    entities: PRF
    triples: PRF  # strict
    edges: PRF  # pair of entities only
    by_predicate: dict[str, PredicateScore]

    def to_dict(self) -> dict:
        return {
            "mentions": self.mentions.to_dict(),
            "entities": self.entities.to_dict(),
            "triples": self.triples.to_dict(),
            "edges": self.edges.to_dict(),
            "by_predicate": {
                p: {
                    "gold": s.gold,
                    "predicted": s.predicted,
                    **s.strict.to_dict(),
                    "edge_recall": round(s.edge_recall, 4),
                }
                for p, s in sorted(self.by_predicate.items())
            },
        }

    def format_table(self) -> str:
        def row(name: str, prf: PRF) -> str:
            return f"  {name:<10} P {prf.precision:.3f}  R {prf.recall:.3f}  F1 {prf.f1:.3f}"

        lines = ["Extraction vs gold", "------------------"]
        lines += [row("mentions", self.mentions), row("entities", self.entities)]
        lines += [row("triples", self.triples), row("edges", self.edges), ""]
        lines.append(f"  {'predicate':<16} {'gold':>5} {'pred':>5}   P      R      F1     edge R")
        for name, s in sorted(self.by_predicate.items()):
            lines.append(
                f"  {name:<16} {s.gold:>5} {s.predicted:>5}   {s.strict.precision:.3f}  "
                f"{s.strict.recall:.3f}  {s.strict.f1:.3f}  {s.edge_recall:.3f}"
            )
        return "\n".join(lines)


def gold_from(world: World, documents: tuple[Document, ...]) -> Gold:
    facts = {e.id: e for e in world.edges}
    mentions = {
        (d.id, m.start, m.end): m.entity_id
        for d in documents
        for s in d.sentences
        for m in s.mentions
    }
    triples = frozenset(
        (facts[f].subject, facts[f].predicate, facts[f].object)
        for d in documents
        for s in d.sentences
        for f in s.fact_ids
    )
    return Gold(mentions, frozenset(mentions.values()), triples)


def oracle(gold: Gold) -> Extraction:
    """A perfect extractor over ``gold`` (canonical ids are the gold entity ids)."""
    return Extraction(
        tuple(PredictedMention(d, a, b, e) for (d, a, b), e in sorted(gold.mentions.items())),
        tuple(sorted(gold.triples)),
    )


def align(gold: Gold, extraction: Extraction) -> dict[str, str]:
    """Canonical id -> the gold entity most of its span-matched mentions refer to (a tie goes to
    the lower entity id, so the scores do not depend on the order mentions arrive in)."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for m in extraction.mentions:
        entity = gold.mentions.get((m.doc, m.start, m.end))
        if m.canonical_id is not None and entity is not None:
            votes[m.canonical_id][entity] += 1
    return {c: min(counts, key=lambda e: (-counts[e], e)) for c, counts in votes.items()}


def _pair(s: str | None, o: str | None) -> frozenset[str] | None:
    """An unordered pair of two different entities, or None if either end is unknown."""
    return frozenset((s, o)) if s and o and s != o else None


def score(gold: Gold, extraction: Extraction) -> ExtractionReport:
    spans = {(m.doc, m.start, m.end) for m in extraction.mentions}
    mentions = PRF.of_sets(len(spans & gold.mentions.keys()), len(spans), len(gold.mentions))

    aligned = align(gold, extraction)
    canonicals = {m.canonical_id for m in extraction.mentions if m.canonical_id is not None}
    entities = PRF.of_sets(len(set(aligned.values())), len(canonicals), len(gold.entities))

    predicted = set(extraction.triples)
    mapped = {t: (aligned.get(t[0]), t[1], aligned.get(t[2])) for t in predicted}
    correct = {t for t, m in mapped.items() if m in gold.triples}
    found = {m for m in mapped.values() if m in gold.triples}
    triples = PRF(len(correct), len(predicted), len(found), len(gold.triples))

    gold_pairs = {_pair(s, o) for s, _, o in gold.triples} - {None}
    predicted_pairs = {_pair(m[0], m[2]) for m in mapped.values()} - {None}
    hit_pairs = predicted_pairs & gold_pairs
    correct_pairs = {t for t, m in mapped.items() if _pair(m[0], m[2]) in gold_pairs}
    edges = PRF(len(correct_pairs), len(predicted), len(hit_pairs), len(gold_pairs))

    by_predicate: dict[str, PredicateScore] = {}
    for name in sorted({t[1] for t in gold.triples} | {t[1] for t in predicted}):
        gold_p = {t for t in gold.triples if t[1] == name}
        predicted_p = {t for t in predicted if t[1] == name}
        strict = PRF(
            len(predicted_p & correct),
            len(predicted_p),
            len({t for t in gold_p if t in found}),
            len(gold_p),
        )
        seen = sum(_pair(s, o) in predicted_pairs for s, _, o in gold_p)
        edge_recall = seen / len(gold_p) if gold_p else 0.0
        by_predicate[name] = PredicateScore(len(gold_p), len(predicted_p), strict, edge_recall)
    return ExtractionReport(mentions, entities, triples, edges, by_predicate)


def ingest(components, documents: tuple[Document, ...]):
    """Ingest each document through ``graph_rag.wiring`` components, yielding (document, result).

    ``result`` is the pipeline's ``PipelineResult``, or None when the pipeline dropped the document.
    """
    for doc in documents:
        key = f"{doc.id}.txt"
        components.object_store.put(components.bucket, key, doc.text.encode("utf-8"))
        yield (
            doc,
            components.orchestrator.process_document(
                IngestTrigger(bucket=components.bucket, object_key=key)
            ),
        )


def run_pipeline(components, documents: tuple[Document, ...]) -> Extraction:
    """Ingest ``documents`` through ``graph_rag.wiring`` components; collect what they extract."""
    mentions: list[PredictedMention] = []
    triples: list[Triple] = []
    for doc, result in ingest(components, documents):
        if result is None:  # the pipeline dropped it: everything in it counts as missed
            continue
        link = {el.mention_text: el.canonical_id for el in result.el_result}
        canonical_surface = {m: c.canonical for c in result.coref_clusters for m in c.members}
        for m in result.mentions:
            surface = canonical_surface.get(m.text, m.text)
            canonical = link.get(m.text) or link.get(surface)
            mentions.append(PredictedMention(doc.id, m.char_start, m.char_end, canonical))
        triples += [(t.subject_id, t.predicate, t.object_id) for t in result.triples]
    return Extraction(tuple(mentions), tuple(triples))
