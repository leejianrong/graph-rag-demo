"""The pipeline shell (N4) — the in-process orchestrator (ADR-0001).

The shell runs the *read* stage (N5) and the ingestion checkpoint (N11): fetch the
document bytes via :class:`~graph_rag.ports.ObjectStore`, create the bare
``ES-Documents`` record with **raw text at ingestion, before processing**, and
persist it via :class:`~graph_rag.ports.DocumentStore`.

V2 adds the first enrichment stage, NER (N6), behind this same shell. V3 adds the
coref stage (N7) — the pipeline's first LLM use — right after it. Both stages are
constructor-injected collaborators (like the ports, ADR-0010) so the fast suite
injects canned fakes and the real stack injects
:class:`~graph_rag.stages.ner.SpacyNerStage` +
:class:`~graph_rag.stages.coref.LLMCorefStage`. Their output — typed mentions +
char spans + sentences (N6) and a non-destructive within-document coref cluster
map (N7) — is carried **in-memory** on the returned
:class:`~graph_rag.models.PipelineResult` and is NOT persisted to ES in V2/V3 (the
ES record still stores raw text only); persistence lands at the V4 EL checkpoint
(ARCHITECTURE §4). Later slices add EL/KG-build to this shell.

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
from graph_rag.models import DocumentRecord, PipelineResult
from graph_rag.stages.coref import LLMCorefStage
from graph_rag.stages.ner import SpacyNerStage

if TYPE_CHECKING:
    from graph_rag.models import IngestTrigger
    from graph_rag.ports import DocumentStore, ObjectStore
    from graph_rag.stages.coref import CorefStage
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
        """
        self._object_store = object_store
        self._document_store = document_store
        self._ner_stage: NerStage = ner_stage if ner_stage is not None else SpacyNerStage()
        self._coref_stage: CorefStage = coref_stage if coref_stage is not None else LLMCorefStage()

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
            #    cluster map over the raw text + mentions, also carried in-memory
            #    (persisted at V4, not here). A cached/identical run costs $0.
            coref_clusters = self._coref_stage.resolve(text, ner.mentions)

            _logger.info(
                "ingested document %s (%s/%s): %d mention(s), %d sentence(s), %d coref cluster(s)",
                doc_id,
                trigger.bucket,
                trigger.object_key,
                len(ner.mentions),
                len(ner.sentences),
                len(coref_clusters),
            )
            # 7. Return the in-memory carry so callers/tests can assert on it.
            return PipelineResult(
                record=record,
                mentions=ner.mentions,
                sentences=ner.sentences,
                coref_clusters=coref_clusters,
            )
        except Exception:  # noqa: BLE001 — log-and-drop per document (ADR-0001).
            _logger.exception(
                "dropping document %s/%s after processing error",
                trigger.bucket,
                trigger.object_key,
            )
            return None
