"""The pipeline shell (N4) — the in-process orchestrator (ADR-0001).

The shell runs the *read* stage (N5) and the ingestion checkpoint (N11): fetch the
document bytes via :class:`~graph_rag.ports.ObjectStore`, create the bare
``ES-Documents`` record with **raw text at ingestion, before processing**, and
persist it via :class:`~graph_rag.ports.DocumentStore`.

V2 adds the first enrichment stage, NER (N6), behind this same shell. V3 adds the
coref stage (N7) — the pipeline's first LLM use — right after it. V4 adds the
entity-linking stage (N8) and the **EL checkpoint**. All stages are
constructor-injected collaborators (like the ports, ADR-0010) so the fast suite
injects canned fakes and the real stack injects
:class:`~graph_rag.stages.ner.SpacyNerStage` +
:class:`~graph_rag.stages.coref.LLMCorefStage` +
:class:`~graph_rag.stages.entity_linking.EntityLinkingStage`.

The NER + coref output — typed mentions + char spans + sentences (N6) and a
non-destructive within-document coref cluster map (N7) — is carried **in-memory**
on the returned :class:`~graph_rag.models.PipelineResult`. Through V3 it is NOT
persisted to ES (the raw record holds text only). **V4's entity-linking stage
resolves each doc-level entity to a corpus-wide ``canonical_id`` (merge or
create-new, upserting canonicals to ``ES-Entities``) and then runs the EL
checkpoint (ADR-0001/0005, ARCHITECTURE §4/§5): the SAME ``ES-Documents`` record
is enriched in place — raw text + NER mentions + coref clusters + per-doc EL
result + sentence vectors — and re-upserted, a second idempotent write to the
same ``document_id`` that overwrites the raw record.**

The EL stage is **opt-in via injection**: when it is not supplied the shell keeps
the raw-only V1–V3 write model (no EL, no checkpoint). This is deliberate — unlike
NER/coref, the EL checkpoint changes what is persisted, so it runs only when an EL
stage is wired (the real stack wires it in ``main.py``; the fast suite injects the
real stage over in-memory fakes). Later slices add KG-build to this shell.

Error handling is **log-and-drop per document** (ADR-0001): any exception while
processing one document is logged and swallowed (``process_document`` returns
``None``) so a single bad document never wedges the Kafka consumer loop — the
next trigger is processed normally. Idempotency comes from the deterministic
``document_id``: reprocessing overwrites (R1.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_rag.ids import document_id
from graph_rag.logging import get_logger
from graph_rag.models import CanonicalEntity, DocumentRecord, EntityLink, PipelineResult, Triple
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.ner import SpacyNerStage

if TYPE_CHECKING:
    from graph_rag.models import IngestTrigger
    from graph_rag.ports import DocumentStore, GraphStore, ObjectStore
    from graph_rag.stages.coref import CorefStage
    from graph_rag.stages.entity_linking import ELStage
    from graph_rag.stages.kg_build import KgStage
    from graph_rag.stages.ner import NerStage

__all__ = ["Orchestrator"]

_logger = get_logger(__name__)


class Orchestrator:
    """The single in-process pipeline shell, constructor-injected with its ports.

    Uses ``object_store`` (read stage), ``document_store`` (ingestion checkpoint),
    ``ner_stage`` (V2 enrichment) and ``coref_stage`` (V3 enrichment, first LLM
    use). The remaining ports (EntityStore/GraphStore/Embedder) plug into this
    same shell in later slices without changing the contract.
    """

    def __init__(
        self,
        object_store: ObjectStore,
        document_store: DocumentStore,
        ner_stage: NerStage | None = None,
        coref_stage: CorefStage | None = None,
        entity_linking_stage: ELStage | None = None,
        graph_store: GraphStore | None = None,
        kg_build_stage: KgStage | None = None,
    ) -> None:
        """Wire the active ports and stages.

        Args:
            object_store: Reads a document's raw bytes (N5 / MinIO).
            document_store: Writes the ``ES-Documents`` record (N11 / Elasticsearch).
            ner_stage: The NER stage (N6). Defaults to a real
                :class:`~graph_rag.stages.ner.SpacyNerStage`; the fast suite
                injects :class:`~graph_rag.fakes.FakeNerStage` so no model loads.
            coref_stage: The coref stage (N7). Defaults to a real
                :class:`~graph_rag.stages.coref.LLMCorefStage` (LiteLLM); the fast
                suite injects :class:`~graph_rag.stages.coref.FakeCorefStage` (or an
                ``LLMCorefStage`` over ``FakeLLMClient``) so no provider is called.
            entity_linking_stage: The entity-linking stage (N8). **Opt-in**: when
                ``None`` the shell keeps the raw-only V1–V3 write model (no EL, no
                checkpoint). Supplied, it resolves canonical entities and drives
                the EL checkpoint. The real stack wires
                :class:`~graph_rag.stages.entity_linking.EntityLinkingStage` in
                ``main.py``; the fast suite injects it over in-memory fakes.
            graph_store: The knowledge-graph store (N13, V5). **Opt-in**, paired
                with ``kg_build_stage``: when either is ``None`` no graph is built
                or written (V1–V4 behaviour is unaffected).
            kg_build_stage: The KG-build stage (N9, V5). **Opt-in**: supplied
                together with ``graph_store``, it builds triples over the doc's
                canonical IDs and the shell runs the **graph checkpoint** (upsert
                nodes → delete this doc's prior edges → write its edges, so
                re-ingest replaces). The real stack wires
                :class:`~graph_rag.stages.kg_build.KgBuildStage` in ``main.py``; the
                fast suite injects it over a ``FakeLLMClient`` + ``InMemoryGraphStore``.
        """
        self._object_store = object_store
        self._document_store = document_store
        self._ner_stage: NerStage = ner_stage if ner_stage is not None else SpacyNerStage()
        self._coref_stage: CorefStage = coref_stage if coref_stage is not None else LLMCorefStage()
        self._entity_linking_stage: ELStage | None = entity_linking_stage
        self._graph_store: GraphStore | None = graph_store
        self._kg_build_stage: KgStage | None = kg_build_stage

    def process_document(self, trigger: IngestTrigger) -> PipelineResult | None:
        """Process one ingest trigger end-to-end, log-and-drop on failure.

        Steps: read bytes (N5) → compute deterministic ``document_id`` → decode to
        text → build the raw :class:`~graph_rag.models.DocumentRecord` → upsert at
        the ingestion checkpoint (N11, raw text only) → run the NER stage (N6) →
        run the coref stage (N7) and carry the typed mentions + sentences + the
        non-destructive coref cluster map **in-memory** on the returned
        :class:`~graph_rag.models.PipelineResult` (NOT persisted until V4).

        Any exception is logged and swallowed so the consumer loop keeps going
        (ADR-0001); on failure this returns ``None`` instead of raising.

        Args:
            trigger: The ``{bucket, object_key}`` payload for one document.

        Returns:
            The :class:`~graph_rag.models.PipelineResult` (raw record + in-memory
            enrichment) on success, or ``None`` if this document failed (dropped).
        """
        try:
            # 1. Read stage (N5): fetch the raw bytes from the object store.
            data = self._object_store.get(trigger.bucket, trigger.object_key)

            # 2. Deterministic identity (ADR-0001): same location -> same id -> overwrite.
            doc_id = document_id(trigger.bucket, trigger.object_key)

            # 3. Decode to text. errors="replace" keeps a malformed byte from
            #    wedging the pipeline; the raw text is what we persist.
            text = data.decode("utf-8", errors="replace")

            # 4. Build the bare record (raw text only — no enrichment persisted yet)
            #    and upsert it at the ingestion checkpoint (N11). The ES write model
            #    is unchanged from V1: NER output is NOT written here (ADR-0001).
            record = DocumentRecord(
                document_id=doc_id,
                bucket=trigger.bucket,
                object_key=trigger.object_key,
                text=text,
            )
            self._document_store.upsert(record)

            # 5. NER stage (N6): typed mentions + char spans + sentences in one
            #    pass, carried in-memory on the result (persisted at V4, not here).
            ner = self._ner_stage.analyze(text)

            # 6. Coref stage (N7, first LLM use): a non-destructive within-doc
            #    cluster map over the raw text + mentions, also carried in-memory.
            #    A cached/identical run costs $0.
            coref_clusters = self._coref_stage.resolve(text, ner.mentions)

            # 7. Entity-linking stage (N8) + EL checkpoint (V4, ADR-0004/0005).
            #    Opt-in: only when an EL stage is wired. It resolves each doc-level
            #    entity to a corpus-wide canonical_id (merge / create-new, upserting
            #    canonicals to ES-Entities), then the checkpoint enriches the SAME
            #    ES-Documents record in place and re-upserts it (a 2nd idempotent
            #    write to the same document_id, overwriting the raw record).
            el_result: list[EntityLink] = []
            triples: list[Triple] = []
            if self._entity_linking_stage is not None:
                el = self._entity_linking_stage.link(
                    text, ner.mentions, ner.sentences, coref_clusters
                )
                el_result = el.links
                record.mentions = ner.mentions
                record.coref_clusters = coref_clusters
                record.el_result = el.links
                record.sentences = ner.sentences  # per-sentence offsets for V6 passage search
                record.sentence_vectors = el.sentence_vectors
                self._document_store.upsert(record)

                # 8. KG-build stage (N9) + graph checkpoint (V5, ADR-0006). Opt-in,
                #    paired: only when a graph store + KG-build stage are wired.
                #    Derive this doc's canonical entities from the EL links (id +
                #    surface + type), build triples over those canonical IDs, then
                #    write the graph. The checkpoint DELETES this doc's prior edges
                #    BEFORE writing its new ones so RE-INGESTING a document REPLACES
                #    its edges rather than duplicating them (graph idempotency,
                #    TESTING gap #1). Nodes are idempotent by canonical_id.
                if self._kg_build_stage is not None and self._graph_store is not None:
                    canonical_entities = self._doc_canonical_entities(el.links)
                    triples = self._kg_build_stage.build(
                        doc_id, text, ner.sentences, el, canonical_entities
                    )
                    self._graph_store.upsert_entities(canonical_entities)
                    self._graph_store.delete_document_edges(doc_id)
                    self._graph_store.write_triples(triples)

            _logger.info(
                "ingested document %s (%s/%s): %d mention(s), %d sentence(s), "
                "%d coref cluster(s), %d entity link(s), %d triple(s)",
                doc_id,
                trigger.bucket,
                trigger.object_key,
                len(ner.mentions),
                len(ner.sentences),
                len(coref_clusters),
                len(el_result),
                len(triples),
            )
            # 9. Return the in-memory carry so callers/tests can assert on it.
            return PipelineResult(
                record=record,
                mentions=ner.mentions,
                sentences=ner.sentences,
                coref_clusters=coref_clusters,
                el_result=el_result,
                triples=triples,
            )
        except Exception:  # noqa: BLE001 — log-and-drop per document (ADR-0001).
            _logger.exception(
                "dropping document %s/%s after processing error",
                trigger.bucket,
                trigger.object_key,
            )
            return None

    @staticmethod
    def _doc_canonical_entities(links: list[EntityLink]) -> list[CanonicalEntity]:
        """Derive this document's canonical entities (graph nodes) from its EL links.

        Each :class:`~graph_rag.models.EntityLink` carries the ``canonical_id`` the
        doc-level entity resolved to plus its surface (``mention_text``) and
        ``entity_type`` — enough to identify the node and hand the KG-build LLM the
        id↔name/type map. Deduplicated by ``canonical_id`` (several links can
        resolve to one canonical). These IDs are exactly the allowed subject/object
        IDs for this document's triples, and the nodes upserted at the graph
        checkpoint — so the graph stays grounded in the EL store (ADR-0006).

        ``DATE`` entities are excluded here: a date is an **edge qualifier**
        (:attr:`~graph_rag.models.Triple.date`), never a standalone node (ADR-0006).
        This is the single chokepoint feeding both the graph nodes and the KG-build
        entity map, so filtering here keeps a spuriously-extracted ``DATE`` mention
        (e.g. "2023") from ever becoming a node or a self-referential edge.
        """
        by_id: dict[str, CanonicalEntity] = {}
        for link in links:
            if link.entity_type == "DATE":
                continue
            by_id.setdefault(
                link.canonical_id,
                CanonicalEntity(
                    canonical_id=link.canonical_id,
                    name=link.mention_text,
                    type=link.entity_type,
                ),
            )
        return list(by_id.values())
