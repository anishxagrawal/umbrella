from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..config import load_settings
from ..generation.generator import build_generator
from ..pipeline.orchestrator import RCAOrchestrator, RCAResponse
from ..retrieval.reranker import build_reranker
from ..retrieval.retriever import build_retriever


logger = logging.getLogger(__name__)


class RCARequest(BaseModel):
    """Request schema for RCA queries."""

    query: str = Field(..., min_length=3, max_length=1000)


class ChunkResponse(BaseModel):
    """API representation of a retrieved chunk."""

    id: int
    text: str
    source: str
    page: int
    section: str
    section_title: str
    chunk_type: str
    spec_number: str
    series_id: str
    similarity: float
    rerank_score: float | None


class RCAApiResponse(BaseModel):
    """API response schema for RCA results."""

    query: str
    expanded_query: str
    answer: str
    chunks: list[ChunkResponse]
    retrieval_count: int
    reranked_count: int
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    """API response schema for the health endpoint."""

    status: str
    components: dict[str, bool]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down application components.

    Args:
        app: FastAPI application instance.

    Yields:
        None.

    Notes:
        Components are created once at startup and stored on app.state.
        The retriever is closed during shutdown when it exposes a close method.
    """
    settings = load_settings()
    retriever = build_retriever(settings)
    reranker = build_reranker(settings)
    generator = build_generator(settings)
    orchestrator = RCAOrchestrator(retriever, reranker, generator)

    # Warm up models at startup to avoid first-request latency.
    logger.info("Warming up embedding model...")
    retriever._embedder.embed_query("warmup")
    logger.info("Warming up reranker model...")
    reranker._get_model()
    logger.info("All models warmed up — ready to serve")

    app.state.settings = settings
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.generator = generator
    app.state.orchestrator = orchestrator

    try:
        yield
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                logger.exception("Failed to close retriever during shutdown")


app = FastAPI(title="Umbrella RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root path to the interactive API docs.

    Args:
        None.

    Returns:
        Redirect response pointing to /docs.

    Notes:
        This endpoint exists as a lightweight entry point for browser users.
    """
    return RedirectResponse(url="/docs")


@app.post("/rca", response_model=RCAApiResponse)
async def run_rca(request: RCARequest) -> RCAApiResponse:
    """Run the end-to-end RCA pipeline for a query.

    Args:
        request: Validated request body containing the query string.

    Returns:
        Structured RCA response suitable for API clients.

    Raises:
        HTTPException: If the orchestrator fails during pipeline execution.

    Notes:
        The orchestrator runs in a worker thread to avoid blocking the event loop.
    """
    orchestrator = _get_orchestrator()
    try:
        response = await run_in_threadpool(orchestrator.run, request.query)
    except RuntimeError as exc:
        logger.exception("RCA request failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected RCA request failure for query=%r", request.query)
        raise HTTPException(status_code=500, detail="RCA request failed.") from exc

    return _to_api_response(response)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return application and component health status.

    Args:
        None.

    Returns:
        Health response with component statuses.

    Notes:
        This endpoint never raises and reports False for missing components.
    """
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is None:
        return HealthResponse(status="degraded", components={
            "retriever": False,
            "reranker": False,
            "generator": False,
        })

    components = await run_in_threadpool(orchestrator.health)
    status = "ok" if all(components.values()) else "degraded"
    return HealthResponse(status=status, components=components)


def _get_orchestrator() -> RCAOrchestrator:
    """Fetch the orchestrator from application state.

    Args:
        None.

    Returns:
        The initialized RCA orchestrator.

    Raises:
        HTTPException: If startup has not populated app.state.orchestrator.

    Notes:
        This helper keeps the request handler focused on request processing.
    """
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Application not initialized.")
    return orchestrator


def _to_api_response(response: RCAResponse) -> RCAApiResponse:
    """Convert an orchestrator response into the API schema.

    Args:
        response: Structured orchestrator output.

    Returns:
        Pydantic response model for transport over HTTP.

    Notes:
        Dataclass chunks are converted field-for-field into Pydantic models.
    """
    return RCAApiResponse(
        query=response.query,
        expanded_query=response.expanded_query,
        answer=response.answer,
        chunks=[ChunkResponse(**asdict(chunk)) for chunk in response.chunks],
        retrieval_count=response.retrieval_count,
        reranked_count=response.reranked_count,
        model=response.model,
        latency_ms=response.latency_ms,
    )
