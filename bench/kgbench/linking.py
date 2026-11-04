"""Score entity linking with B-cubed cluster metrics against the generator's identities (ADR-0011).

A pipeline's canonical entities cluster mentions; the world model says which mentions truly refer to
the same entity. B-cubed scores each mention by the overlap of its predicted cluster with its gold
cluster, so one mistake costs in proportion to the clusters it touches:

* merging two entities (homonyms) puts foreign mentions in a cluster and lowers **precision**;
* splitting one entity (aliases) leaves its mentions in separate clusters and lowers **recall**.

Only mentions the pipeline found by exact span are scored, so a missed mention (an extraction error,
scored by ``kgbench.extraction``) does not also count as a linking error. A found mention the
pipeline did not link is a cluster of one. Results are also broken down by what makes a mention
hard: ``homonym`` (its entity's name is shared by another entity), ``alias`` (written other than
as the entity's name) and ``plain``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from kggen.render import Document
from kggen.world import World

from kgbench.extraction import Extraction, Span

CATEGORIES = ("homonym", "alias", "plain")

# Named difficulty settings: knobs layered over a base world spec, from no ambiguity to a lot.
DIFFICULTY: dict[str, dict[str, float]] = {
    "clean": {},
    "aliases": {"alias_rate": 0.7, "aliases_per_entity": 2},
    "ambiguous": {
        "alias_rate": 0.7,
        "aliases_per_entity": 2,
        "homonym_rate": 0.15,
        "distractor_rate": 0.2,
    },
}


@dataclass(frozen=True)
class BCubed:
    precision: float
    recall: float
    mentions: int

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {
            "mentions": self.mentions,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class LinkingReport:
    overall: BCubed
    by_category: dict[str, BCubed]
    clusters_predicted: int  # canonical entities holding at least one scored mention
    clusters_gold: int  # entities with at least one scored mention

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "by_category": {c: self.by_category[c].to_dict() for c in CATEGORIES},
            "clusters_predicted": self.clusters_predicted,
            "clusters_gold": self.clusters_gold,
        }

    def format_table(self) -> str:
        lines = ["Entity linking vs gold (B-cubed)", "--------------------------------"]
        rows = [("overall", self.overall)] + [(c, self.by_category[c]) for c in CATEGORIES]
        for name, b in rows:
            if not b.mentions:
                lines.append(f"  {name:<9} n/a  (no such mentions in this world)")
                continue
            lines.append(
                f"  {name:<9} P {b.precision:.3f}  R {b.recall:.3f}  F1 {b.f1:.3f}"
                f"  ({b.mentions} mentions)"
            )
        lines.append(f"  clusters: {self.clusters_predicted} predicted, {self.clusters_gold} gold")
        return "\n".join(lines)


@dataclass(frozen=True)
class LinkingGold:
    entity: dict[Span, str]  # span -> the entity it truly refers to
    category: dict[Span, str]  # span -> homonym | alias | plain


def linking_gold(world: World, documents: tuple[Document, ...]) -> LinkingGold:
    names = {e.id: e.name for e in world.entities}
    shared = {n for n, c in Counter(names.values()).items() if c > 1}
    entity: dict[Span, str] = {}
    category: dict[Span, str] = {}
    for d in documents:
        for s in d.sentences:
            for m in s.mentions:
                span = (d.id, m.start, m.end)
                entity[span] = m.entity_id
                name = names[m.entity_id]
                category[span] = (
                    "homonym" if name in shared else "alias" if m.surface != name else "plain"
                )
    return LinkingGold(entity, category)


def score_linking(gold: LinkingGold, extraction: Extraction) -> LinkingReport:
    predicted: dict[Span, str] = {}
    for i, m in enumerate(extraction.mentions):
        span = (m.doc, m.start, m.end)
        if span in gold.entity:
            predicted[span] = m.canonical_id if m.canonical_id is not None else f"_unlinked{i}"

    pred_members: dict[str, set[Span]] = defaultdict(set)
    gold_members: dict[str, set[Span]] = defaultdict(set)
    for span, cluster in predicted.items():
        pred_members[cluster].add(span)
        gold_members[gold.entity[span]].add(span)

    def bcubed(spans: list[Span]) -> BCubed:
        if not spans:
            return BCubed(0.0, 0.0, 0)
        precision = recall = 0.0
        for span in spans:
            p, g = pred_members[predicted[span]], gold_members[gold.entity[span]]
            overlap = len(p & g)
            precision += overlap / len(p)
            recall += overlap / len(g)
        return BCubed(precision / len(spans), recall / len(spans), len(spans))

    spans = sorted(predicted)
    by_category = {c: bcubed([s for s in spans if gold.category[s] == c]) for c in CATEGORIES}
    return LinkingReport(bcubed(spans), by_category, len(pred_members), len(gold_members))
