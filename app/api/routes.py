"""
API route definitions.

Two routers are registered in main.py:

  root_router  — mounted at /
                 GET /health  → basic liveness (no auth, used by Docker healthcheck)

  v1_router    — mounted at /v1
                 GET /v1/health  → deep dependency probe
                 POST /v1/ask    → guardrailed RAG query
"""

import asyncio
import logging
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.auth import require_api_key
from app.config import settings
from app.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse, ServiceStatus
from app.services.pipeline import run_pipeline
from app.utils.context import get_request_id

logger = logging.getLogger(__name__)

# ── Liveness router (no auth, no version prefix) ─────────────────────────────

root_router = APIRouter(tags=["Operations"])


@root_router.get(
    "/health",
    summary="Liveness check",
    description=(
        "Returns `200 ok` when the process is running. "
        "Does **not** verify downstream connectivity (Neo4j, vLLM, Chroma). "
        "Used by Docker / load-balancer health checks."
    ),
)
async def health_liveness() -> dict[str, str]:
    return {"status": "ok"}


# ── v1 router (optional auth, /v1 prefix added by main.py) ───────────────────

v1_router = APIRouter(tags=["Knowledge Query"])


# ── Deep health ───────────────────────────────────────────────────────────────

def _probe_neo4j() -> ServiceStatus:
    """Synchronous Neo4j probe — run in a thread via asyncio.to_thread."""
    try:
        from app.services.graph_store import _get_driver
        t0 = time.perf_counter()
        driver = _get_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return ServiceStatus(
            status="ok",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as exc:
        return ServiceStatus(status="down", detail=str(exc)[:120])


async def _probe_vllm() -> ServiceStatus:
    # Strip /v1 suffix to get the vLLM server root
    base = settings.openai_api_base.removesuffix("/v1").removesuffix("/")
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base}/health")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        status = "ok" if resp.status_code == 200 else "degraded"
        return ServiceStatus(status=status, latency_ms=latency_ms)
    except Exception as exc:
        return ServiceStatus(status="down", detail=str(exc)[:120])


def _probe_chroma() -> ServiceStatus:
    p = Path(settings.chroma_dir)
    if p.exists() and p.is_dir():
        return ServiceStatus(status="ok")
    return ServiceStatus(status="down", detail=f"Directory not found: {settings.chroma_dir}")


@v1_router.get(
    "/health",
    summary="Deep health check",
    description=(
        "Probes Neo4j, vLLM, and Chroma in parallel. "
        "Returns `degraded` (HTTP 200) if any non-critical service is slow; "
        "`down` (HTTP 503) if a critical service is unreachable."
    ),
    tags=["Operations"],
    response_model=HealthResponse,
)
async def health_deep(_: None = Depends(require_api_key)) -> JSONResponse:
    req_id = get_request_id()

    neo4j_result, vllm_result, chroma_result = await asyncio.gather(
        asyncio.to_thread(_probe_neo4j),
        _probe_vllm(),
        asyncio.to_thread(_probe_chroma),
    )

    services = {
        "neo4j": neo4j_result,
        "vllm": vllm_result,
        "chroma": chroma_result,
    }

    statuses = {s.status for s in services.values()}
    if "down" in statuses:
        overall = "down"
        http_status = 503
    elif "degraded" in statuses:
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    logger.info(
        "Deep health | overall=%s neo4j=%s vllm=%s chroma=%s",
        overall,
        neo4j_result.status,
        vllm_result.status,
        chroma_result.status,
    )

    body = HealthResponse(
        status=overall,
        version="1.0.0",
        services=services,
        request_id=req_id,
    )
    return JSONResponse(
        status_code=http_status,
        content=body.model_dump(),
    )


# ── SOP knowledge query ───────────────────────────────────────────────────────

@v1_router.post(
    "/ask",
    summary="Query the SOP knowledge base",
    description=(
        "Submit a natural-language question about wafer fab SOP procedures, "
        "process anomaly handling, or equipment state dependencies.\n\n"
        "The request passes through four guardrail stages: "
        "injection detection → topic filter → evidence sufficiency → fact grounding.\n\n"
        "Set `enable_guards: false` to bypass all guardrails (debugging only). "
        "Set `debug: true` to include the raw LLM context and output in the response."
    ),
    response_description=(
        "Structured answer with guardrail trace, retrieved evidence triples, "
        "and `reasoning_type` indicating how the pipeline resolved the query."
    ),
)
async def ask(
    req: AskRequest,
    _: None = Depends(require_api_key),
) -> JSONResponse:
    req_id = get_request_id()
    logger.info("POST /v1/ask | question=%r", req.question[:80])
    result: AskResponse = run_pipeline(req)
    result.request_id = req_id
    return JSONResponse(
        content=result.model_dump(by_alias=True, exclude_none=True)
    )
