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
from app.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse, IngestRequest, IngestResponse, ServiceStatus
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
    result: AskResponse = await asyncio.to_thread(run_pipeline, req)
    result.request_id = req_id
    return JSONResponse(
        content=result.model_dump(by_alias=True, exclude_none=True)
    )


# ── SOP knowledge graph ingest ────────────────────────────────────────────────

def _run_ingest(req: IngestRequest) -> IngestResponse:
    """Merge nodes and edges into Neo4j, tagging each with source_file."""
    from app.services.graph_store import _get_driver

    driver = _get_driver()
    nodes_merged = 0
    edges_merged = 0
    edges_skipped = 0

    with driver.session() as session:
        for node in req.nodes:
            label = node.get("label", "Node")
            props = dict(node.get("properties", {}))
            props["source_file"] = req.source_file
            node_id = props.get("id", "")
            cypher = (
                f"MERGE (n:{label} {{id: $id}}) "
                "SET n += $props "
                "RETURN n.id AS id"
            )
            session.run(cypher, id=node_id, props=props)
            nodes_merged += 1

        for edge in req.edges:
            rel_type = edge.get("type", "RELATES_TO")
            from_label = edge.get("from_label", "")
            from_id = edge.get("from_id", "")
            to_label = edge.get("to_label", "")
            to_id = edge.get("to_id", "")
            props = dict(edge.get("properties", {}))
            props["source_file"] = req.source_file

            match_clause = (
                f"MATCH (a{':' + from_label if from_label else ''} {{id: $from_id}}) "
                f"MATCH (b{':' + to_label if to_label else ''} {{id: $to_id}}) "
            )
            cypher = (
                match_clause
                + f"MERGE (a)-[r:{rel_type}]->(b) "
                + "SET r += $props "
                + "RETURN type(r) AS rel"
            )
            result = session.run(cypher, from_id=from_id, to_id=to_id, props=props)
            if result.single():
                edges_merged += 1
            else:
                edges_skipped += 1

    logger.info(
        "Ingest complete | source=%s nodes=%d edges=%d skipped=%d",
        req.source_file, nodes_merged, edges_merged, edges_skipped,
    )
    return IngestResponse(
        status="ok",
        nodes_merged=nodes_merged,
        edges_merged=edges_merged,
        edges_skipped=edges_skipped,
    )


@v1_router.post(
    "/ingest",
    summary="Ingest SOP nodes and edges into the knowledge graph",
    description=(
        "Merge new SOP nodes and edges into Neo4j. "
        "Each node and edge is tagged with `source_file` for lifecycle tracking. "
        "Safe to re-run: existing nodes/edges are updated via MERGE, not duplicated."
    ),
    response_model=IngestResponse,
)
async def ingest(
    req: IngestRequest,
    _: None = Depends(require_api_key),
) -> IngestResponse:
    req_id = get_request_id()
    logger.info(
        "POST /v1/ingest | source=%s nodes=%d edges=%d req_id=%s",
        req.source_file, len(req.nodes), len(req.edges), req_id,
    )
    return await asyncio.to_thread(_run_ingest, req)
