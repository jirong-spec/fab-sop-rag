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
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import require_api_key
from app.config import APP_VERSION, settings
from app.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse, IngestRequest, IngestResponse, ServiceStatus
from app.services.pipeline import run_pipeline, run_pipeline_stream
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
    try:
        from app.services.vector_store import _get_vector_store
        t0 = time.perf_counter()
        db = _get_vector_store()
        count = len(db.get(include=[])["ids"])
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if count == 0:
            return ServiceStatus(
                status="degraded", latency_ms=latency_ms,
                detail="sop_docs collection is empty — run vector ingest first",
            )
        return ServiceStatus(status="ok", latency_ms=latency_ms)
    except Exception as exc:
        return ServiceStatus(status="down", detail=str(exc)[:120])


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
        version=APP_VERSION,
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


# ── SOP knowledge query (streaming) ──────────────────────────────────────────

@v1_router.post(
    "/ask/stream",
    summary="Query the SOP knowledge base (streaming)",
    description=(
        "Same guardrail pipeline as `/v1/ask`, but tokens stream via "
        "Server-Sent Events so the first word appears in <200ms.\n\n"
        "Event types:\n"
        "- `token` — one LLM output token (`{\"type\":\"token\",\"text\":\"...\"}`)\n"
        "- `done`  — final metadata after guard_grounding completes\n"
        "- `blocked` — a guardrail blocked the request\n\n"
        "guard_grounding runs after the last token; clients see the full answer "
        "~1150ms before the final `done` event arrives."
    ),
    response_class=StreamingResponse,
)
async def ask_stream(
    req: AskRequest,
    _: None = Depends(require_api_key),
) -> StreamingResponse:
    logger.info("POST /v1/ask/stream | question=%r", req.question[:80])
    req_id = get_request_id()

    async def _generate():
        import contextvars
        import threading
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        _SENTINEL = object()
        cancelled = threading.Event()
        ctx = contextvars.copy_context()  # carry request_id into the thread

        def _run_sync():
            try:
                for event in run_pipeline_stream(req, request_id=req_id):
                    if cancelled.is_set():
                        break
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    except (RuntimeError, asyncio.QueueFull):
                        break
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                except RuntimeError:
                    pass

        threading.Thread(target=lambda: ctx.run(_run_sync), daemon=True).start()

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            cancelled.set()

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── SOP knowledge graph ingest ────────────────────────────────────────────────

_ALLOWED_NODE_LABELS = frozenset({
    "SOPDocument", "SOPStep", "Anomaly", "Equipment", "Node",
})
_ALLOWED_REL_TYPES = frozenset({
    "TRIGGERS_SOP", "FIRST_STEP", "NEXT_STEP", "DEPENDS_ON",
    "REQUIRES_STATUS", "PRECONDITION", "DEFINED_IN",
    "INTERLOCK_WITH", "CROSS_DOC_DEPENDENCY", "RELATES_TO",
})
def _validate_identifier(value: str, allowed: frozenset[str], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"Disallowed {field}: {value!r}. Allowed: {sorted(allowed)}")
    return value


def _run_ingest(req: IngestRequest) -> IngestResponse:
    """Merge nodes and edges into Neo4j, tagging each with source_file."""
    from app.services.graph_store import _get_driver

    driver = _get_driver()
    nodes_merged = 0
    edges_merged = 0
    edges_skipped = 0

    try:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                for node in req.nodes:
                    raw_label = node.get("label", "Node")
                    label = _validate_identifier(raw_label, _ALLOWED_NODE_LABELS, "node label")
                    props = dict(node.get("properties", {}))
                    props["source_file"] = req.source_file
                    node_id = props.get("id", "")
                    if not node_id:
                        raise HTTPException(status_code=422, detail=f"Node missing 'id' property: {node}")
                    cypher = (
                        f"MERGE (n:{label} {{id: $id}}) "
                        "SET n += $props "
                        "RETURN n.id AS id"
                    )
                    tx.run(cypher, id=node_id, props=props)
                    nodes_merged += 1

                for edge in req.edges:
                    raw_rel = edge.get("type", "RELATES_TO")
                    rel_type = _validate_identifier(raw_rel, _ALLOWED_REL_TYPES, "relationship type")
                    raw_from = edge.get("from_label", "")
                    raw_to = edge.get("to_label", "")
                    from_label = _validate_identifier(raw_from, _ALLOWED_NODE_LABELS, "from_label") if raw_from else ""
                    to_label = _validate_identifier(raw_to, _ALLOWED_NODE_LABELS, "to_label") if raw_to else ""
                    from_id = edge.get("from_id", "")
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
                    result = tx.run(cypher, from_id=from_id, to_id=to_id, props=props)
                    if result.single():
                        edges_merged += 1
                    else:
                        edges_skipped += 1

                tx.commit()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Ingest failed | source=%s error=%s", req.source_file, exc)
        return IngestResponse(
            status="error",
            nodes_merged=nodes_merged,
            edges_merged=edges_merged,
            edges_skipped=edges_skipped,
            detail=str(exc)[:200],
        )

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
