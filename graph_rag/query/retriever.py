"""The V6 query retriever (N16) — the deterministic, ``$0`` retrieval path (ADR-0007).

Composes the four query-side ports — :class:`~graph_rag.ports.Embedder`,
:class:`~graph_rag.ports.EntityStore` (kNN), :class:`~graph_rag.ports.DocumentStore`
(sentence kNN) and :class:`~graph_rag.ports.GraphStore` (k-hop) — with the pinned
B4 ranker (:mod:`graph_rag.query.ranking`) into one ``retrieve`` call. It answers a
multi-hop, cross-document question with a connected subgraph, a ranked list of
candidate answer nodes and supporting sentences with provenance — **no LLM call**
(the ``synthesize`` flag is the V7 gate and is ignored here, ADR-0007).

The flow is **seed → expand → rank → answer** (ADR-0007 retrieval mode):

1. **Embed** the question with the local sentence-transformer (reused from EL).
2. **Seed** (B5): entity-anchored kNN over ``ES-Entities`` for the seed entities +
   their cosine scores, and passage-anchored kNN over ``ES-Documents`` sentence
   vectors for the supporting-sentence candidates.
3. **Expand**: k-hop BFS from the seed entities in the graph store → the connected
   :class:`~graph_rag.models.Subgraph`.
4. **Rank**: score every node with the B4 formula (``w_seed`` seed cosine +
   ``w_prox`` graph proximity); the top node is the predicted entity answer.

Every collaborator is constructor-injected (ADR-0010) so the fast suite drives the
retriever over in-memory fakes with no Docker, no model download and no LLM.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from graph_rag.logging import get_logger
from graph_rag.models import QueryResponse
from graph_rag.query.ranking import rank_nodes, select_answer

if TYPE_CHECKING:
    from graph_rag.config import Settings
    from graph_rag.models import QueryRequest, Subgraph
    from graph_rag.ports import DocumentStore, Embedder, EntityStore, GraphStore
    from graph_rag.query.synthesis import AnswerSynthesizer

__all__ = ["QueryRetriever"]

_logger = get_logger(__name__)


class QueryRetriever:
    """The non-LLM query retriever (N16) — seed → expand → rank → answer.

    Constructor-injected with the four query-side ports plus the settings-derived
    seeding depths, k-hop depth and ranking weights (ADR-0010). Use
    :meth:`from_settings` to build one from a :class:`~graph_rag.config.Settings`.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        entity_store: EntityStore,
        document_store: DocumentStore,
        graph_store: GraphStore,
        seed_top_k_entities: int,
        seed_top_k_sentences: int,
        khop_depth: int,
        rank_weight_seed: float,
        rank_weight_proximity: float,
        synthesizer: AnswerSynthesizer | None = None,
    ) -> None:
        """Wire the ports and the pinned retrieval knobs.

        Args:
            embedder: Embeds the question (reused EL sentence-transformer, B1).
            entity_store: Entity-anchored kNN seeding over ``ES-Entities`` (B5).
            document_store: Passage-anchored sentence kNN over ``ES-Documents`` (B5).
            graph_store: k-hop subgraph expansion in the knowledge graph.
            seed_top_k_entities: How many entity seeds to pull (B5).
            seed_top_k_sentences: How many supporting-sentence seeds to pull (B5).
            khop_depth: BFS expansion depth from the entity seeds (B3).
            rank_weight_seed: B4 seed-similarity weight (default ``0.7``).
            rank_weight_proximity: B4 graph-proximity weight (default ``0.3``).
            synthesizer: OPTIONAL V7 prose synthesizer (N17). ``None`` (the default)
                keeps the path deterministic and LLM-free; when wired, it runs ONLY
                if a request sets ``synthesize=true`` (the gate defaults OFF).
        """
        self._embedder = embedder
        self._entity_store = entity_store
        self._document_store = document_store
        self._graph_store = graph_store
        self._seed_top_k_entities = seed_top_k_entities
        self._seed_top_k_sentences = seed_top_k_sentences
        self._khop_depth = khop_depth
        self._rank_weight_seed = rank_weight_seed
        self._rank_weight_proximity = rank_weight_proximity
        self._synthesizer = synthesizer

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: Embedder,
        entity_store: EntityStore,
        document_store: DocumentStore,
        graph_store: GraphStore,
        synthesizer: AnswerSynthesizer | None = None,
    ) -> QueryRetriever:
        """Build a retriever from :class:`~graph_rag.config.Settings` + injected ports.

        Reads the B3/B4/B5 knobs (``seed_top_k_entities``, ``seed_top_k_sentences``,
        ``khop_depth``, ``rank_weight_seed``, ``rank_weight_proximity``) off
        ``settings``; the ports are injected so the caller reuses the SAME embedder
        / stores it built for ingestion. ``synthesizer`` is the OPTIONAL V7 prose
        synthesizer — pass one to enable gated ``synthesize=true`` synthesis; omit
        it to keep the path deterministic and LLM-free.
        """
        return cls(
            embedder=embedder,
            entity_store=entity_store,
            document_store=document_store,
            graph_store=graph_store,
            seed_top_k_entities=settings.seed_top_k_entities,
            seed_top_k_sentences=settings.seed_top_k_sentences,
            khop_depth=settings.khop_depth,
            rank_weight_seed=settings.rank_weight_seed,
            rank_weight_proximity=settings.rank_weight_proximity,
            synthesizer=synthesizer,
        )

    def retrieve(self, request: QueryRequest) -> QueryResponse:
        """Answer ``request`` via the deterministic retrieval path (+ optional V7 prose).

        Runs seed → expand → rank → answer over the injected ports and the B4
        ranker, returning the connected subgraph, the ranked candidate nodes, the
        predicted entity answer and the supporting sentences with provenance — the
        deterministic, ``$0`` V6 result, computed with NO LLM call.

        The ``request.synthesize`` flag is the V7 gate (ADR-0009) and defaults OFF:
        the LLM is touched ONLY when ``request.synthesize`` is true AND a
        :class:`~graph_rag.query.synthesis.AnswerSynthesizer` is wired, in which
        case the retrieved evidence is synthesized into ``response.prose``. With the
        flag off (the default) or no synthesizer wired, ``prose`` stays ``None`` and
        the response is byte-for-byte the V6 shape — no LLM call is made.

        Args:
            request: The natural-language query (its ``question`` + ``synthesize``).

        Returns:
            The :class:`~graph_rag.models.QueryResponse` for the question.
        """
        # 1. Embed the question once (local sentence-transformer, reused from EL).
        question_vector = self._embedder.embed([request.question])[0]

        # 2. Seed (B5). Entity-anchored: kNN over ES-Entities -> seed entities +
        #    their cosine. Passage-anchored: kNN over the ES-Documents sentence
        #    vectors -> supporting-sentence candidates (already scored + ordered).
        entity_seeds = self._entity_store.knn(
            vector=question_vector, top_k=self._seed_top_k_entities
        )
        supporting_sentences = self._document_store.search_sentences(
            vector=question_vector, top_k=self._seed_top_k_sentences
        )

        # canonical_id -> best seed cosine (a node can be returned once, but keep
        # the max defensively so re-seeding never lowers a score).
        seed_scores: dict[str, float] = {}
        for entity, score in entity_seeds:
            prior = seed_scores.get(entity.canonical_id)
            if prior is None or score > prior:
                seed_scores[entity.canonical_id] = score

        # 3. Expand: k-hop BFS from the seed entities -> the connected subgraph.
        subgraph = self._graph_store.khop(seed_ids=list(seed_scores), hops=self._khop_depth)

        # 4. Hop distance: BFS over the RETURNED subgraph's edges from the seeds
        #    (seeds = 0; unreachable nodes are omitted -> the ranker treats them as
        #    inf -> 0.0 proximity).
        hop_distance = self._hop_distances(subgraph, seed_scores)

        # 5. Rank (B4) + pick the top node as the entity answer (no type filter, V6).
        ranked_nodes = rank_nodes(
            subgraph,
            seed_scores,
            hop_distance=hop_distance,
            w_seed=self._rank_weight_seed,
            w_prox=self._rank_weight_proximity,
        )
        answer_entity = select_answer(ranked_nodes)
        answer = answer_entity.name if answer_entity is not None else None

        _logger.info(
            "query answered: %d entity seed(s), %d sentence seed(s), "
            "subgraph=%d node(s)/%d edge(s), answer=%r",
            len(seed_scores),
            len(supporting_sentences),
            len(subgraph.nodes),
            len(subgraph.edges),
            answer,
        )

        # 6. Supporting sentences already come back scored + ordered by the sentence
        #    kNN (SupportingSentence carries no vector to re-rank), so keep that
        #    order and its provenance (doc_id + offsets).
        response = QueryResponse(
            answer=answer,
            answer_entity=answer_entity,
            subgraph=subgraph,
            ranked_nodes=ranked_nodes,
            supporting_sentences=supporting_sentences,
        )

        # 7. V7 GATE (default OFF): synthesize prose ONLY when the request asks for
        #    it AND a synthesizer is wired. Otherwise prose stays None and no LLM is
        #    touched — the response is byte-for-byte the V6 shape (ADR-0009).
        if request.synthesize and self._synthesizer is not None:
            response.prose = self._synthesizer.synthesize(
                question=request.question, response=response
            )
            _logger.info("query synthesized prose (%d char(s))", len(response.prose))

        return response

    @staticmethod
    def _hop_distances(subgraph: Subgraph, seed_scores: dict[str, float]) -> dict[str, float]:
        """BFS hop distance from the nearest seed over the subgraph edges.

        Treats the subgraph edges as undirected. Seeds present in the subgraph are
        distance ``0``; each edge crossed adds one hop. Nodes unreachable from any
        seed are omitted (the ranker reads a missing entry as ``inf`` -> ``0.0``
        proximity). Deterministic: the result depends only on the node/edge sets.

        Args:
            subgraph: The k-hop traversal result to measure over.
            seed_scores: ``canonical_id`` -> seed cosine; its keys are the BFS roots.

        Returns:
            ``canonical_id`` -> hop distance (float) for every reachable node.
        """
        node_ids = {node.canonical_id for node in subgraph.nodes}

        # Undirected adjacency restricted to nodes actually in the subgraph.
        adjacency: dict[str, set[str]] = {cid: set() for cid in node_ids}
        for edge in subgraph.edges:
            if edge.subject_id in node_ids and edge.object_id in node_ids:
                adjacency[edge.subject_id].add(edge.object_id)
                adjacency[edge.object_id].add(edge.subject_id)

        distances: dict[str, float] = {}
        queue: deque[str] = deque()
        for seed_id in seed_scores:
            if seed_id in node_ids and seed_id not in distances:
                distances[seed_id] = 0.0
                queue.append(seed_id)

        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1.0
                    queue.append(neighbour)
        return distances
