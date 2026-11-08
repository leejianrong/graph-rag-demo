"""FastAPI app exposing ``POST /ingest`` (U1/N1), ``POST /query`` (U3/N16) and ``GET /health``.

The ingest endpoint is a **thin** entry point (ADR-0001): it stores the uploaded
bytes via the ``ObjectStore`` port, computes the deterministic ``document_id``, and
publishes an :class:`~graph_rag.models.IngestTrigger` via the ``TriggerPublisher``
port. It does **not** run the pipeline — the Kafka consumer + orchestrator do that
downstream.

The query endpoint (V6) is a **synchronous** read path (ADR-0007): it hands a
:class:`~graph_rag.models.QueryRequest` to the injected
:class:`~graph_rag.query.retriever.QueryRetriever` and returns its
:class:`~graph_rag.models.QueryResponse` directly — no Kafka, no LLM (the
deterministic, ``$0`` retrieval mode). When no retriever is wired it returns 503.

Dependencies are injected through :func:`create_app` so the real composition root
(main.py) passes live adapters while tests pass in-memory fakes; the endpoint never
constructs an adapter itself (ADR-0010).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from graph_rag.agent_api import add_agent_routes
from graph_rag.auth import ApiKeys, AuthError, JsonFileKeyStore
from graph_rag.config import Settings, get_settings
from graph_rag.ids import document_id
from graph_rag.jobs import Job, JobStore
from graph_rag.logging import get_logger
from graph_rag.lookup import EntityLookup
from graph_rag.models import DeletedDocument, IngestTrigger, QueryRequest, QueryResponse
from graph_rag.ports import ObjectStore, TriggerPublisher
from graph_rag.scope import DEFAULT_SCOPE, RESERVED_KEY_PREFIX, Scope

if TYPE_CHECKING:
    from graph_rag.query.retriever import QueryRetriever

__all__ = ["create_app"]

_logger = get_logger(__name__)


def _holds(store: ObjectStore, scope: Scope, bucket: str, object_key: str, data: bytes) -> bool:
    """Whether ``(bucket, object_key)`` already holds exactly ``data`` in ``scope``."""
    try:
        return store.get(scope, bucket, object_key) == data
    except FileNotFoundError:
        return False


# Multipart framing (boundaries, headers) added to a file's bytes, so the declared length of a file
# at exactly the cap does not trip the early check; the read in /ingest is the exact limit.
_MULTIPART_SLACK = 64 * 1024


def _too_large(limit: int) -> HTTPException:
    return HTTPException(status_code=413, detail=f"The upload is over the {limit}-byte limit.")


def create_app(
    object_store: ObjectStore,
    publisher: TriggerPublisher,
    settings: Settings | None = None,
    *,
    retriever: QueryRetriever | None = None,
    eraser: Callable[[str, Scope], DeletedDocument | None] | None = None,
    keys: ApiKeys | None = None,
    lookup: EntityLookup | None = None,
    jobs: JobStore | None = None,
) -> FastAPI:
    """Build the FastAPI app with its ports injected.

    Args:
        object_store: Where uploaded bytes are stored (MinIO in prod, fake in tests).
        publisher: Where the ingest trigger is published (Kafka in prod, fake in tests).
        settings: Runtime settings; falls back to :func:`~graph_rag.config.get_settings`.
        retriever: The V6 query retriever backing ``POST /query`` (real ports in
            prod, fakes in tests). **Optional**: when ``None`` the query endpoint
            responds ``503`` — ``/ingest`` + ``/health`` are unaffected.
        eraser: Forgets one document by id (``Orchestrator.delete_document``) and returns
            what it removed, or ``None`` for an unknown id. **Optional**: when ``None``
            ``DELETE /documents/{id}`` responds ``503``.
        jobs: Where ingest progress is read from for ``GET /jobs/{id}``, and where ``accept``
            records a document as queued. **Optional**: when ``None`` the jobs endpoint
            responds ``503``. Share the instance with the orchestrator.
        lookup: Backs ``POST /v1/lookup``. **Optional**: when ``None`` it responds ``unavailable``.
        keys: Verifies API keys. **Optional**: when ``None`` and ``settings.require_auth`` is on,
            keys are read from ``settings.api_keys_path``. With ``require_auth`` off, no key is
            checked and every request runs in the default tenant.

    Returns:
        A configured :class:`fastapi.FastAPI` app exposing ``POST /ingest``,
        ``POST /query`` and ``GET /health``.
    """
    resolved_settings = settings or get_settings()
    bucket = resolved_settings.minio_bucket

    app = FastAPI(title="Graph RAG Demo — Ingest + Query API")
    if resolved_settings.require_auth and keys is None:
        keys = ApiKeys(JsonFileKeyStore(resolved_settings.api_keys_path))
    verifier = keys if resolved_settings.require_auth else None
    max_upload = resolved_settings.max_upload_bytes

    @app.middleware("http")
    async def refuse_oversize_upload(request: Request, call_next: Callable) -> Response:  # type: ignore[type-arg]
        """Turn away a declared-oversize /ingest before the server spools the body to disk."""
        declared = request.headers.get("content-length", "")
        if (
            request.method == "POST"
            and request.url.path == "/ingest"
            and declared.isdigit()
            and int(declared) > max_upload + _MULTIPART_SLACK
        ):
            return JSONResponse(status_code=413, content={"detail": _too_large(max_upload).detail})
        return await call_next(request)  # type: ignore[no-any-return]

    def tenant_of(request: Request) -> str:
        """The tenant a request acts for: its key's tenant, or the default when auth is off."""
        if verifier is None:
            return DEFAULT_SCOPE.tenant
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("Send the API key as 'Authorization: Bearer <key>'.")
        return verifier.verify(token.strip()).tenant

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe (dev-playbook observability floor)."""
        return {"status": "ok"}

    def accept(
        object_key: str, data: bytes, force: bool, scope: Scope = DEFAULT_SCOPE
    ) -> dict[str, str]:
        """Store ``data`` under ``object_key`` and queue it, unless the source already holds it."""
        doc_id = document_id(bucket, object_key, scope)
        if not force and _holds(object_store, scope, bucket, object_key, data):
            _logger.info("unchanged document_id=%s object_key=%s", doc_id, object_key)
            return {
                "document_id": doc_id,
                "bucket": bucket,
                "object_key": object_key,
                "status": "unchanged",
            }

        object_store.put(scope, bucket, object_key, data)
        if jobs is not None:  # before publishing, so a fast worker's "done" is not overwritten
            jobs.queued(scope, doc_id, object_key)
        publisher.publish(
            IngestTrigger(
                bucket=bucket,
                object_key=object_key,
                tenant=scope.tenant,
                namespace=scope.namespace,
            )
        )

        _logger.info(
            "ingested document_id=%s bucket=%s object_key=%s bytes=%d",
            doc_id,
            bucket,
            object_key,
            len(data),
        )
        return {
            "document_id": doc_id,
            "bucket": bucket,
            "object_key": object_key,
            "status": "accepted",
        }

    @app.post("/ingest")
    async def ingest(
        file: Annotated[UploadFile, File()],
        source_url: Annotated[str | None, Form()] = None,
        force: bool = False,
        tenant: str = Depends(tenant_of),
    ) -> dict[str, str]:
        """Store an uploaded document and publish its ingest trigger.

        The source is keyed by ``source_url`` when given, else by the uploaded filename;
        the bucket comes from settings. Returns the deterministic ``document_id``, the
        resolved location and a ``status``: ``accepted`` when the bytes were stored and a
        trigger published, or ``unchanged`` when this source already holds exactly these
        bytes, in which case nothing is stored or published, so the pipeline (and any LLM
        behind it) does not run again. ``force=true`` ingests it anyway, for a source whose
        first attempt failed. Does NOT run the pipeline — that happens downstream.
        """
        data = await file.read(max_upload + 1)  # one byte over is enough to know it is too big
        if len(data) > max_upload:
            raise _too_large(max_upload)
        object_key = source_url or file.filename or "upload"
        if object_key.startswith(RESERVED_KEY_PREFIX):
            raise HTTPException(
                status_code=400, detail=f"Keys may not start with {RESERVED_KEY_PREFIX!r}."
            )
        return accept(object_key, data, force, Scope(tenant))

    @app.get("/jobs/{doc_id}")
    def job(doc_id: str, tenant: str = Depends(tenant_of)) -> Job:
        """Where a document is in the pipeline: queued, running, done or failed.

        The id is the ``document_id`` that ``/ingest`` and ``/v1/remember`` returned. A failed
        job says which stage failed. An id this tenant never queued is a 404, whoever else
        has queued it, and a process restart forgets jobs (they are held in memory).
        """
        if jobs is None:
            raise HTTPException(status_code=503, detail="Job status is not configured here.")
        found = jobs.get(Scope(tenant), doc_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No job {doc_id}.")
        return found

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, tenant: str = Depends(tenant_of)) -> DeletedDocument:
        """Forget a document: its edges, its record, its raw bytes, and entities only it used.

        Runs in this process, not through Kafka, so the answer says what was removed. A
        document that is not stored is a 404. Entities that another document still
        mentions are kept.
        """
        if eraser is None:
            raise HTTPException(status_code=503, detail="Delete is not configured on this service.")
        result = eraser(doc_id, Scope(tenant))
        if result is None:
            raise HTTPException(status_code=404, detail=f"No document {doc_id}.")
        return result

    @app.post("/query")
    def query(request: QueryRequest, tenant: str = Depends(tenant_of)) -> QueryResponse:
        """Answer a question via the deterministic, ``$0`` retrieval path (V6, ADR-0007).

        Synchronous read path: hands the parsed :class:`~graph_rag.models.QueryRequest`
        to the injected retriever and returns its
        :class:`~graph_rag.models.QueryResponse` (connected subgraph + ranked nodes
        + predicted entity answer + supporting sentences with provenance). No Kafka,
        no LLM. Responds ``503`` when no retriever is wired.
        """
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail="Query retriever is not configured on this service.",
            )
        response = retriever.retrieve(request, Scope(tenant))
        _logger.info(
            "query question=%r answer=%r ranked=%d",
            request.question,
            response.answer,
            len(response.ranked_nodes),
        )
        return response

    add_agent_routes(
        app,
        accept=accept,
        retriever=retriever,
        eraser=eraser,
        bucket=bucket,
        max_text_bytes=resolved_settings.max_remember_bytes,
        tenant_of=tenant_of,
        lookup=lookup,
    )
    return app
