"""The entity-linking stage (N8) — corpus-local cross-document unification (ADR-0004).

The third enrichment stage and the **heart of R3**: it turns the same real-world
entity mentioned across different documents into ONE canonical entity, which
later becomes one graph node. Entity linking is entity resolution against the
corpus-local ``ES-Entities`` store (no external knowledge base, ADR-0004); the
algorithm, per doc-level entity, is **block → score → decide**:

1. **Doc-level entities.** The unit of linking is a *doc-level entity*, derived
   from the V3 coref clusters: each cluster is one doc-level entity whose surface
   is the cluster's in-document ``canonical`` and whose type comes from its member
   mentions. A mention that no cluster covers is treated as its own doc-level
   entity (surface + type taken straight from the mention).
2. **Embed mention-in-context.** The mention vector is the embedding of the
   surface form plus the sentence(s) it appears in — context disambiguates
   ("Apple" the company vs. the fruit).
3. **Block + kNN.** Candidates are gathered two ways: ``block_candidates`` (same
   type + same :func:`~graph_rag.normalize.normalize_name` key) — the corpus-local
   identity key — and ``knn`` (the nearest entity vectors of the same type) for
   fuzzy, differently-phrased surfaces.
4. **Decide (deterministic — R6.4).** An **exact-key block match** is decisive: the
   doc-level entity **merges** into that existing ``canonical_id`` (reusing it,
   growing its ``aliases``) regardless of the context-cosine — so the same entity
   named identically across documents stays ONE node even when its per-document
   mention-in-context embeddings drift below the threshold. With no block match, the
   fuzzy kNN candidates are scored by cosine and merge only if the best is at or
   above the fixed ``el_threshold`` (B2); otherwise a **new** canonical entity is
   minted and upserted (the normal always-on path). Genuine same-type/same-name
   homonyms are the province of the gated LLM tie-breaker. Linking is
   **order-sensitive**: the first document to mention an entity seeds its canonical
   record (name + vector).

Two credit-conserving refinements are **wired but gated OFF by default**
(ADR-0004): an LLM tie-breaker for near-threshold decisions and NIL retention for
very-low-confidence entities. When their flags are off the branches are inert and
**no LLM is called** — the default path is $0.

Like the NER and coref stages, this stage runs behind the :class:`ELStage`
interface and is constructor-injected into the orchestrator (the testability
seam, ADR-0010): the fast suite injects it over
:class:`~graph_rag.fakes.InMemoryEntityStore` + :class:`~graph_rag.fakes.FakeEmbedder`
(no Docker, no model download); the real stack injects it over the Elasticsearch
``EntityStore`` + the sentence-transformer ``Embedder``.

``canonical_id`` minting scheme (deterministic + stable, ADR-0004): a create-new
id is ``"e-" + sha256("el:{type}:{normalized_name}")[:16]``. It is derived purely
from the first-seen normalized name + type, so re-ingesting the same corpus (with
the same embedder + fixed order, B8) re-derives the same ids and merges rather
than duplicating. In the rare case where a genuinely-different entity shares a
type + normalized name with an existing one (blocked but scoring below threshold),
the id is disambiguated deterministically by mixing in a hash of the mention
vector, so it stays unique and reproducible.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from graph_rag.logging import get_logger
from graph_rag.models import CanonicalEntity, EntityLink
from graph_rag.normalize import normalize_name

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import CorefCluster, CuratedType, Mention, Sentence
    from graph_rag.ports import Embedder, EntityStore, LLMClient

__all__ = [
    "ELStage",
    "ELResult",
    "EntityLinkingStage",
]

_logger = get_logger(__name__)


@dataclass
class ELResult:
    """The entity-linking stage's output — per-doc links + sentence vectors.

    A plain carrier (like :class:`~graph_rag.stages.ner.NerResult`) so the
    orchestrator can persist both at the EL checkpoint: ``links`` populate the
    :class:`~graph_rag.models.DocumentRecord`'s ``el_result`` and
    ``sentence_vectors`` its passage vectors for query-side seeding (B5).
    """

    links: list[EntityLink]
    sentence_vectors: list[list[float]]


@dataclass
class _DocEntity:
    """One doc-level entity derived from the coref clusters / uncovered mentions.

    ``surface`` is the linking surface form (a cluster's in-document canonical, or
    a lone mention's text); ``entity_type`` is its curated type; ``spans`` are the
    char offsets of its mentions, used to gather mention-in-context sentences.
    """

    surface: str
    entity_type: CuratedType
    spans: list[tuple[int, int]] = field(default_factory=list)


@runtime_checkable
class ELStage(Protocol):
    """The entity-linking stage seam: doc enrichment → per-doc EL result.

    Constructor-injected into the orchestrator; the fast suite injects the real
    :class:`EntityLinkingStage` over in-memory fakes instead of live ES/model.
    """

    def link(
        self,
        text: str,
        mentions: list[Mention],
        sentences: list[Sentence],
        coref_clusters: list[CorefCluster],
    ) -> ELResult:
        """Resolve the document's doc-level entities to canonical ids."""
        ...


class EntityLinkingStage:
    """Real :class:`ELStage` — corpus-local block/score/merge EL (ADR-0004).

    Constructor-injected with the :class:`~graph_rag.ports.EntityStore` it resolves
    against and the :class:`~graph_rag.ports.Embedder` it scores with. Deterministic
    for a fixed store state + embedder + threshold (R6.4).
    """

    def __init__(
        self,
        entity_store: EntityStore,
        embedder: Embedder,
        *,
        threshold: float = 0.82,
        knn_top_k: int = 5,
        tiebreaker_enabled: bool = False,
        nil_enabled: bool = False,
        tiebreaker_margin: float = 0.05,
        nil_floor: float = 0.5,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Wire the stage to its ports + fixed policy.

        Args:
            entity_store: The canonical-entity store to block/score/upsert against.
            embedder: Produces the mention-in-context + sentence vectors.
            threshold: Cosine merge-vs-create-new threshold (B2, ``el_threshold``).
            knn_top_k: How many nearest entities to pull as extra candidates.
            tiebreaker_enabled: Gate for the LLM tie-breaker (default OFF —
                ADR-0004). When off, the tie-breaker branch is inert and no LLM is
                called.
            nil_enabled: Gate for NIL retention of very-low-confidence entities
                (default OFF). When off, low-confidence always create-new.
            tiebreaker_margin: Half-width of the near-threshold band the (gated)
                tie-breaker would arbitrate.
            nil_floor: The (gated) NIL floor; below it an entity would be retained
                as NIL rather than created.
            llm_client: Optional client for the gated tie-breaker; only used when
                ``tiebreaker_enabled`` is true. Never called on the default path.
        """
        self._entity_store = entity_store
        self._embedder = embedder
        self._threshold = threshold
        self._knn_top_k = knn_top_k
        self._tiebreaker_enabled = tiebreaker_enabled
        self._nil_enabled = nil_enabled
        self._tiebreaker_margin = tiebreaker_margin
        self._nil_floor = nil_floor
        self._llm = llm_client

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        entity_store: EntityStore,
        embedder: Embedder,
        *,
        llm_client: LLMClient | None = None,
    ) -> EntityLinkingStage:
        """Build the stage from :class:`~graph_rag.config.Settings` + injected ports.

        The ports (store + embedder) are injected rather than constructed here so
        the composition root owns adapter lifetimes; only the fixed EL policy
        (threshold B2, kNN fan-out, gated flags) is read from settings.
        """
        return cls(
            entity_store,
            embedder,
            threshold=settings.el_threshold,
            knn_top_k=settings.el_knn_top_k,
            tiebreaker_enabled=settings.el_tiebreaker_enabled,
            nil_enabled=settings.el_nil_enabled,
            llm_client=llm_client,
        )

    def link(
        self,
        text: str,
        mentions: list[Mention],
        sentences: list[Sentence],
        coref_clusters: list[CorefCluster],
    ) -> ELResult:
        """Resolve every doc-level entity to a canonical id (merge or create-new).

        Args:
            text: The raw document text (unused directly; offsets index into it).
            mentions: The NER mentions (types + char spans).
            sentences: The segmented sentences (mention-in-context + passage vectors).
            coref_clusters: The within-document coref cluster map (V3) — each
                cluster is one doc-level entity.

        Returns:
            An :class:`ELResult` with one :class:`~graph_rag.models.EntityLink` per
            doc-level entity and the document's sentence vectors.
        """
        links: list[EntityLink] = []
        for entity in self._doc_level_entities(mentions, coref_clusters):
            context = self._build_context(entity, sentences)
            vector = self._embedder.embed([context])[0]
            normalized = normalize_name(entity.surface)

            # Exact-key block match (same curated type + same normalized name/alias)
            # is the corpus-local identity key (ADR-0004): a hit is the SAME entity,
            # so unify into it DETERMINISTICALLY, independent of the context-cosine.
            # This is the fix for cross-document splitting: the mention-in-context
            # embedding of, e.g., "Berlin" drifts to ~0.70 cosine between documents —
            # below the 0.82 threshold — which used to mint a divergent duplicate
            # node and break the multi-hop bridge. The lowest canonical_id is chosen
            # for order-independent stability. Genuine same-type/same-name homonyms
            # are left to the gated LLM tie-breaker on the fuzzy path below.
            block = self._entity_store.block_candidates(
                entity_type=entity.entity_type, normalized_name=normalized
            )
            if block:
                best_entity = min(block, key=lambda candidate: candidate.canonical_id)
                scored, cosine = self._best_candidate(vector, [best_entity])
                best_score = cosine if scored is not None else 1.0
                merge = True
            else:
                # No exact-key match: fuzzy kNN candidates scored by cosine, merging
                # only at/above the threshold (this is how a differently-phrased
                # surface of one entity still merges). The gated tie-breaker
                # arbitrates near-threshold decisions here.
                candidates = self._knn_candidates(entity.entity_type, vector)
                best_entity, best_score = self._best_candidate(vector, candidates)
                merge = best_entity is not None and best_score >= self._threshold
                merge = self._maybe_tiebreak(
                    merge=merge,
                    score=best_score,
                    surface=entity.surface,
                    candidate=best_entity,
                    context=context,
                )

            if merge and best_entity is not None:
                self._merge(best_entity, entity.surface, vector)
                links.append(
                    EntityLink(
                        mention_text=entity.surface,
                        canonical_id=best_entity.canonical_id,
                        entity_type=entity.entity_type,
                        score=best_score,
                        is_new=False,
                    )
                )
                continue

            # Create-new (the normal always-on path). The gated NIL branch would
            # divert very-low-confidence entities here; off by default it never does.
            self._maybe_nil(best_score)
            canonical_id = self._mint_id(entity.entity_type, normalized, vector)
            self._entity_store.upsert(
                CanonicalEntity(
                    canonical_id=canonical_id,
                    name=entity.surface,
                    type=entity.entity_type,
                    aliases=[],
                    vector=vector,
                )
            )
            links.append(
                EntityLink(
                    mention_text=entity.surface,
                    canonical_id=canonical_id,
                    entity_type=entity.entity_type,
                    score=best_score,
                    is_new=True,
                )
            )

        return ELResult(links=links, sentence_vectors=self.embed_sentences(sentences))

    def embed_sentences(self, sentences: list[Sentence]) -> list[list[float]]:
        """Embed each sentence's text (one vector per sentence, in order).

        Used at the EL checkpoint to persist passage vectors for query-side
        seeding (B5). Returns ``[]`` for a document with no sentences.
        """
        if not sentences:
            return []
        return self._embedder.embed([sentence.text for sentence in sentences])

    # --- doc-level entity derivation ----------------------------------------

    def _doc_level_entities(
        self, mentions: list[Mention], coref_clusters: list[CorefCluster]
    ) -> list[_DocEntity]:
        """Derive the doc-level entities from coref clusters + uncovered mentions.

        Each cluster becomes one doc-level entity (surface = its canonical, type +
        spans from its member mentions). Every mention not consumed by a cluster
        becomes its own doc-level entity. A cluster whose surface forms match no
        typed mention cannot be typed and is skipped (logged).
        """
        entities: list[_DocEntity] = []
        covered: set[int] = set()

        for cluster in coref_clusters:
            member_texts = {cluster.canonical, *cluster.members}
            matched = [i for i, m in enumerate(mentions) if m.text in member_texts]
            entity_type = self._cluster_type(cluster, mentions, matched)
            if entity_type is None:
                _logger.debug(
                    "skipping untyped coref cluster %r (no matching NER mention)",
                    cluster.canonical,
                )
                continue
            covered.update(matched)
            spans = [(mentions[i].char_start, mentions[i].char_end) for i in matched]
            entities.append(
                _DocEntity(surface=cluster.canonical, entity_type=entity_type, spans=spans)
            )

        for i, mention in enumerate(mentions):
            if i in covered:
                continue
            entities.append(
                _DocEntity(
                    surface=mention.text,
                    entity_type=mention.type,
                    spans=[(mention.char_start, mention.char_end)],
                )
            )
        return entities

    @staticmethod
    def _cluster_type(
        cluster: CorefCluster, mentions: list[Mention], matched: list[int]
    ) -> CuratedType | None:
        """Pick a cluster's type: the mention equal to its canonical, else the first match."""
        for i in matched:
            if mentions[i].text == cluster.canonical:
                return mentions[i].type
        if matched:
            return mentions[matched[0]].type
        return None

    @staticmethod
    def _build_context(entity: _DocEntity, sentences: list[Sentence]) -> str:
        """Build the mention-in-context string: the surface + its sentence(s).

        A sentence is included if any of the entity's mention spans starts within
        it. Falls back to the bare surface when there are no sentences/overlap. This
        vector is the entity's stored embedding — used for query-side seeding (B5) —
        so keeping the sentence context makes it richer for retrieval. It is NOT the
        sole basis for the merge decision: an exact type+normalized-name block match
        unifies deterministically (see :meth:`link`), so a context that drifts
        across documents no longer splits the same entity into duplicate nodes.
        """
        parts = [entity.surface]
        seen: set[int] = set()
        for start, _end in entity.spans:
            for sentence in sentences:
                if sentence.index in seen:
                    continue
                if sentence.char_start <= start < sentence.char_end:
                    seen.add(sentence.index)
                    parts.append(sentence.text)
        return " ".join(parts)

    # --- candidate gathering + scoring --------------------------------------

    def _knn_candidates(
        self, entity_type: CuratedType, vector: list[float]
    ) -> list[CanonicalEntity]:
        """The fuzzy candidate set: nearest entities of the same type by vector kNN.

        Used only when there is no exact-key block match (:meth:`link` handles that
        decisively). Returns ``[]`` when kNN is disabled (``knn_top_k <= 0``).
        """
        if self._knn_top_k <= 0:
            return []
        return [
            candidate
            for candidate, _score in self._entity_store.knn(
                vector=vector, entity_type=entity_type, top_k=self._knn_top_k
            )
        ]

    def _best_candidate(
        self, vector: list[float], candidates: list[CanonicalEntity]
    ) -> tuple[CanonicalEntity | None, float]:
        """Return the highest-cosine candidate and its score (``(None, 0.0)`` if none)."""
        best: CanonicalEntity | None = None
        best_score = 0.0
        for candidate in candidates:
            if candidate.vector is None:
                continue
            score = _cosine(vector, candidate.vector)
            if best is None or score > best_score:
                best, best_score = candidate, score
        return best, best_score

    # --- decision helpers ----------------------------------------------------

    def _merge(self, entity: CanonicalEntity, surface: str, vector: list[float]) -> None:
        """Merge ``surface`` into ``entity``: grow aliases, keep the seed vector.

        The canonical record's ``name`` + ``vector`` are kept as first-seeded
        (order-sensitivity, ADR-0004); a genuinely new surface form is appended to
        ``aliases``. The upsert is idempotent by ``canonical_id`` (no duplicate).
        """
        aliases = list(entity.aliases)
        normalized = normalize_name(surface)
        known = {normalize_name(entity.name), *(normalize_name(a) for a in aliases)}
        if normalized and normalized not in known:
            aliases.append(surface)
        self._entity_store.upsert(
            CanonicalEntity(
                canonical_id=entity.canonical_id,
                name=entity.name,
                type=entity.type,
                aliases=aliases,
                vector=entity.vector if entity.vector is not None else vector,
            )
        )

    def _mint_id(self, entity_type: CuratedType, normalized_name: str, vector: list[float]) -> str:
        """Mint a deterministic, stable ``canonical_id`` for a create-new entity.

        Derived from ``type`` + ``normalized_name`` so a re-ingest of the same
        corpus re-derives it (and merges instead of duplicating). Disambiguated by
        a vector hash only in the rare case that id already belongs to a distinct
        entity of the same type + normalized name.
        """
        base = f"el:{entity_type}:{normalized_name}"
        canonical_id = "e-" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
        if self._entity_store.get(canonical_id) is None:
            return canonical_id
        vector_key = hashlib.sha256(
            ",".join(f"{x:.6f}" for x in vector).encode("utf-8")
        ).hexdigest()[:8]
        return "e-" + hashlib.sha256(f"{base}|{vector_key}".encode()).hexdigest()[:16]

    def _maybe_tiebreak(
        self,
        *,
        merge: bool,
        score: float,
        surface: str,
        candidate: CanonicalEntity | None,
        context: str,
    ) -> bool:
        """Gated LLM tie-breaker for near-threshold decisions (OFF by default).

        When the gate is off (default) this returns the deterministic ``merge``
        decision unchanged and **never calls the LLM**. When enabled, a decision
        whose score falls within ``tiebreaker_margin`` of the threshold would be
        arbitrated by the LLM; wired here as a clearly-gated branch.
        """
        if not self._tiebreaker_enabled or self._llm is None or candidate is None:
            return merge
        if abs(score - self._threshold) > self._tiebreaker_margin:
            return merge
        # Gated-on path (not exercised by the default fast suite): ask the LLM
        # whether ``surface``-in-``context`` is the same entity as ``candidate``.
        verdict = self._llm.complete(
            "Are these the same real-world entity? Answer yes or no.\n"
            f"A: {surface}\nContext: {context}\n"
            f"B: {candidate.name} (aliases: {', '.join(candidate.aliases)})"
        )
        return verdict.strip().lower().startswith("y")

    def _maybe_nil(self, score: float) -> bool:
        """Gated NIL retention for very-low-confidence entities (OFF by default).

        Returns ``False`` when the gate is off (default), so the create-new path
        proceeds normally. When enabled, a score below ``nil_floor`` signals the
        entity should be retained as NIL rather than created.
        """
        if not self._nil_enabled:
            return False
        return score < self._nil_floor


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
