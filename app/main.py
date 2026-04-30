import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_config import setup_logging
from app.middleware.request_id import RequestIDMiddleware
from app.api.routes import root_router, v1_router
from app.utils.context import get_request_id

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fab SOP Knowledge Query API starting up (version 1.0.0)")
    # Pre-warm all heavy resources so the first real request has normal latency.
    try:
        from app.services.vector_store import _get_vector_store
        _get_vector_store()
        logger.info("Warm-up: embedding model + Chroma loaded")
    except Exception as e:
        logger.warning("Warm-up vector store failed (non-fatal): %s", e)
    try:
        from app.services.graph_store import _get_driver
        _get_driver()
        logger.info("Warm-up: Neo4j driver connected")
    except Exception as e:
        logger.warning("Warm-up Neo4j failed (non-fatal): %s", e)
    try:
        from app.services.llm_client import chat_completion
        chat_completion("ping", max_tokens=1)
        logger.info("Warm-up: LLM endpoint reachable")
    except Exception as e:
        logger.warning("Warm-up LLM failed (non-fatal): %s", e)
    yield
    logger.info("Fab SOP Knowledge Query API shutting down")


app = FastAPI(
    title="Fab SOP Knowledge Query API",
    description=(
        "Single-machine Enterprise MVP for querying a **wafer fab SOP document knowledge base** "
        "via hybrid Graph + Vector RAG.\n\n"
        "Designed for process, equipment, and integration engineers who need to look up "
        "SOP-recommended handling procedures for process anomalies, trace equipment state "
        "dependencies, or identify pre-check conditions across documents.\n\n"
        "**Guardrail stages** — injection detection · topic filter · evidence sufficiency · "
        "fact grounding — are applied at input, retrieval, and output to reduce the risk of "
        "off-topic responses and hallucinated SOP guidance.\n\n"
        "**Authentication:** Set `API_KEY` in the environment to enable `X-API-Key` header "
        "validation. Leave empty to disable auth (suitable for internal demos).\n\n"
        "**Internal use only.** No document-level access control. "
        "Do not use for automated fab operations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# RequestIDMiddleware must be added before exception handlers run so that
# every error response also carries the correlation ID.
app.add_middleware(RequestIDMiddleware)

# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with a uniform error envelope that includes the request ID."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "request_id": get_request_id(),
        },
    )


@app.exception_handler(Exception)
async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — prevents raw tracebacks reaching the client."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "request_id": get_request_id(),
        },
    )

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(root_router)            # GET /health  (liveness, no auth)
app.include_router(v1_router, prefix="/v1")  # GET /v1/health, POST /v1/ask
