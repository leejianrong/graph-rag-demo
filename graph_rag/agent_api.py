"""The ``/v1`` agent API: ``remember``, ``recall`` and ``forget`` as JSON calls (ADR-0014, GRA-132).

A thin facade. Each verb reshapes a request onto the endpoint logic that already exists (ingest,
the retriever, the eraser) and answers failures in the contract's one shape,
``{"error": {"code", "message", "details?"}}``. The routes under ``/v1`` use that shape; the older
endpoints keep FastAPI's ``{"detail": ...}``. ``lookup`` and the planned fields (``token_budget``,
``as_of``) are not served; a body that carries one has it ignored. ``namespace`` partitions memory
inside the tenant (ADR-0015): each verb runs in the scope it names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from graph_rag.auth import AuthError
from graph_rag.ids import document_id
from graph_rag.logging import get_logger
from graph_rag.models import DeletedDocument, QueryRequest, QueryResponse
from graph_rag.scope import RESERVED_KEY_PREFIX, Scope

if TYPE_CHECKING:
    from graph_rag.query.retriever import QueryRetriever

__all__ = ["add_agent_routes"]

_logger = get_logger(__name__)
_PREFIX = "/v1"
_STATUS = {
    "invalid_request": 400,
    "unauthorized": 401,
    "not_found": 404,
    "payload_too_large": 413,
    "unavailable": 503,
    "internal": 500,
}


class AgentError(Exception):
    """A failure with one of the contract's error codes."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


def _scope(tenant: str, namespace: str | None) -> Scope:
    """The scope a request names; a namespace that cannot be one is ``invalid_request``."""
    try:
        return Scope(tenant=tenant, namespace=namespace)
    except ValueError as error:
        raise AgentError(
            "invalid_request", f"namespace: {error}", {"field": "namespace"}
        ) from error


class RememberRequest(BaseModel):
    text: str = Field(min_length=1)
    namespace: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    force: bool = False


class RememberResponse(BaseModel):
    document_id: str
    source_id: str
    status: str


class RecallRequest(BaseModel):
    question: str = Field(min_length=1)
    synthesize: bool = False
    namespace: str | None = Field(default=None, min_length=1)


class ForgetRequest(BaseModel):
    source_id: str | None = Field(default=None, min_length=1)
    document_id: str | None = Field(default=None, min_length=1)
    namespace: str | None = Field(default=None, min_length=1)


class ForgetResponse(BaseModel):
    document_id: str
    source_id: str
    entities_removed: list[str]
    entities_kept: list[str]


def _body(error: AgentError) -> JSONResponse:
    content: dict[str, Any] = {"code": error.code, "message": error.message}
    if error.details:
        content["details"] = error.details
    return JSONResponse({"error": content}, status_code=_STATUS[error.code])


def _derived_source_id(text: str) -> str:
    """The id of a source nobody named: identical text lands on the same one."""
    return "text/" + hashlib.sha256(text.encode()).hexdigest()


def add_agent_routes(
    app: FastAPI,
    *,
    accept: Callable[[str, bytes, bool, Scope], dict[str, str]],
    retriever: QueryRetriever | None,
    eraser: Callable[[str, Scope], DeletedDocument | None] | None,
    bucket: str,
    max_text_bytes: int,
    tenant_of: Callable[[Request], str],
) -> None:
    """Serve ``/v1/remember``, ``/v1/recall`` and ``/v1/forget`` on ``app``.

    Args:
        app: The app to extend; its validation and unhandled-error responses are reshaped for
            ``/v1`` paths only.
        accept: Stores bytes under a source key and queues them (``/ingest``'s own logic).
        retriever: Backs ``recall``; ``None`` makes it answer ``unavailable``.
        eraser: Backs ``forget``; ``None`` makes it answer ``unavailable``.
        bucket: The bucket a ``document_id`` is derived in.
        max_text_bytes: The largest text ``remember`` takes; more is ``payload_too_large``.
        tenant_of: Names the tenant a request acts for, or raises :class:`AuthError`.
    """
    router = APIRouter(prefix=_PREFIX)

    @router.post("/remember", status_code=202)
    def remember(request: RememberRequest, tenant: str = Depends(tenant_of)) -> RememberResponse:
        """Store text under a source id and queue it for the pipeline."""
        data = request.text.encode()
        if len(data) > max_text_bytes:
            raise AgentError(
                "payload_too_large",
                f"The text is {len(data)} bytes; send at most {max_text_bytes}.",
                {"limit_bytes": max_text_bytes},
            )
        source_id = request.source_id or _derived_source_id(request.text)
        if source_id.startswith(RESERVED_KEY_PREFIX):
            raise AgentError(
                "invalid_request",
                f"source_id may not start with {RESERVED_KEY_PREFIX!r}.",
                {"field": "source_id"},
            )
        stored = accept(source_id, data, request.force, _scope(tenant, request.namespace))
        return RememberResponse(
            document_id=stored["document_id"], source_id=source_id, status=stored["status"]
        )

    @router.post("/recall")
    def recall(request: RecallRequest, tenant: str = Depends(tenant_of)) -> QueryResponse:
        """Answer a question and return the evidence behind the answer."""
        if retriever is None:
            raise AgentError("unavailable", "Recall is not configured on this service.")
        return retriever.retrieve(
            QueryRequest(question=request.question, synthesize=request.synthesize),
            _scope(tenant, request.namespace),
        )

    @router.post("/forget")
    def forget(request: ForgetRequest, tenant: str = Depends(tenant_of)) -> ForgetResponse:
        """Forget a source, named by its source id or its document id."""
        if (request.source_id is None) == (request.document_id is None):
            raise AgentError(
                "invalid_request", "Give exactly one of source_id and document_id.", None
            )
        if eraser is None:
            raise AgentError("unavailable", "Forget is not configured on this service.")
        scope = _scope(tenant, request.namespace)
        doc_id = request.document_id or document_id(bucket, request.source_id or "", scope)
        removed = eraser(doc_id, scope)
        if removed is None:
            raise AgentError("not_found", f"No source matches {request.source_id or doc_id}.")
        return ForgetResponse(
            document_id=removed.document_id,
            source_id=removed.object_key,
            entities_removed=removed.entities_removed,
            entities_kept=removed.entities_kept,
        )

    app.include_router(router)

    def on_v1(request: Request) -> bool:
        return request.url.path.startswith(_PREFIX + "/")

    @app.exception_handler(AgentError)
    def agent_error(_: Request, error: AgentError) -> Response:
        return _body(error)

    @app.exception_handler(AuthError)
    def unauthorized(request: Request, error: AuthError) -> Response:
        challenge = {"WWW-Authenticate": "Bearer"}
        if on_v1(request):
            response = _body(AgentError("unauthorized", f"{error} Send a valid API key."))
            response.headers.update(challenge)
            return response
        return JSONResponse({"detail": str(error)}, status_code=401, headers=challenge)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> Response:
        if not on_v1(request):
            return await request_validation_exception_handler(request, error)
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        return _body(
            AgentError(
                "invalid_request",
                f"{field or 'body'}: {first['msg']}.",
                {"field": field or "body"},
            )
        )

    @app.exception_handler(Exception)
    def unexpected(request: Request, error: Exception) -> Response:
        _logger.exception("unhandled error on %s", request.url.path)
        if on_v1(request):
            return _body(AgentError("internal", "Something went wrong on the server."))
        return PlainTextResponse("Internal Server Error", status_code=500)
