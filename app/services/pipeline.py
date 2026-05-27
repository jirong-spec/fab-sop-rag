import json
import logging
import time
from collections.abc import Iterator

from app.schemas import AskRequest, AskResponse, DebugInfo, GuardrailResult
from app.services.guardrails import (
    guard_injection,
    guard_topic,
    guard_evidence,
    guard_grounding,
)
from app.services.retrieval_service import retrieve
from app.services.answer_service import generate_answer, generate_answer_stream, LLM_ERROR_ANSWER
from app.utils.text_utils import extract_source_docs

logger = logging.getLogger(__name__)


def run_pipeline(req: AskRequest) -> AskResponse:
    """
    Orchestrate the full guardrailed RAG pipeline.

    Flow:
        Input → [guard_injection] → [guard_topic]
             → Retrieval → [guard_evidence]
             → Generation → Output → [guard_grounding]
    """
    t0 = time.perf_counter()
    question = req.question
    guardrail_results: list[GuardrailResult] = []
    stage_latencies: dict[str, int] = {}

    logger.info(
        "Pipeline start | question=%r | guards=%s | hop=%d | top_k=%d",
        question[:80],
        req.enable_guards,
        req.max_hop,
        req.top_k,
    )

    # ── Input Guards ─────────────────────────────────────────────────────────
    if req.enable_guards:
        _ts = time.perf_counter()
        inj = guard_injection(question)
        stage_latencies["guard_injection"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(inj)
        logger.info("Guard[injection] pass=%s reason=%s", inj.passed, inj.reason)
        if not inj.passed:
            return _blocked(req, guardrail_results, "blocked_injection", inj.reason, t0)

        _ts = time.perf_counter()
        topic = guard_topic(question)
        stage_latencies["guard_topic"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(topic)
        logger.info("Guard[topic] pass=%s reason=%s", topic.passed, topic.reason)
        if not topic.passed:
            return _blocked(req, guardrail_results, "blocked_off_topic", topic.reason, t0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    _ts = time.perf_counter()
    entities, triples = retrieve(question, top_k=req.top_k, max_hop=req.max_hop)
    stage_latencies["retrieval"] = int((time.perf_counter() - _ts) * 1000)
    logger.info(
        "Retrieval done | entities=%s | triples=%d", entities, len(triples)
    )

    # ── Retrieval Guard ───────────────────────────────────────────────────────
    if req.enable_guards:
        ev = guard_evidence(triples)
        guardrail_results.append(ev)
        logger.info("Guard[evidence] pass=%s reason=%s", ev.passed, ev.reason)
        if not ev.passed:
            return _blocked(
                req, guardrail_results, "blocked_low_evidence", ev.reason, t0,
                entities=entities, triples=triples,
            )

    # ── Generation ────────────────────────────────────────────────────────────
    _ts = time.perf_counter()
    answer, model_triples = generate_answer(question, triples, entities=entities)
    stage_latencies["generation"] = int((time.perf_counter() - _ts) * 1000)
    logger.info("Answer generated | preview=%r", answer[:80])

    if answer == LLM_ERROR_ANSWER:
        return _blocked(req, guardrail_results, "llm_error", answer, t0,
                        entities=entities, triples=triples)

    # ── Output Guard ──────────────────────────────────────────────────────────
    reasoning_type = "graph_rag"
    confidence = 1.0

    if req.enable_guards:
        _ts = time.perf_counter()
        gr = guard_grounding(answer, model_triples)
        stage_latencies["guard_grounding"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(gr)
        logger.info("Guard[grounding] pass=%s reason=%s", gr.passed, gr.reason)
        if not gr.passed:
            reasoning_type = "answered_with_warning"
            confidence = 0.5

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Pipeline done | status=answered | reasoning=%s | latency_ms=%d stages=%s",
        reasoning_type,
        latency_ms,
        stage_latencies,
    )

    debug = _make_debug(req, triples, answer, stage_latencies) if req.debug else None

    return AskResponse(
        question=question,
        status="answered",
        answer=answer,
        entities=entities,
        evidence_triples=triples,
        model_triples=model_triples,
        source_docs=extract_source_docs(triples),
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=confidence,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

def run_pipeline_stream(req: AskRequest, request_id: str = "-") -> Iterator[str]:
    """
    Streaming variant of run_pipeline.

    Yields Server-Sent Events (text/event-stream format):
      {"type": "token",   "text": "..."}          — one per LLM output token
      {"type": "done",    "answer": "...", ...}    — final metadata after grounding
      {"type": "blocked", "status": "...", ...}    — if any guard blocks

    Flow:
      Guards 1-3 + Retrieval run synchronously (fast, ~100ms total).
      Generation tokens stream immediately to the client (~1900ms).
      guard_grounding runs after the last token (~1150ms, but client already
      has the full answer by then).
    """
    t0 = time.perf_counter()
    question = req.question
    guardrail_results: list[GuardrailResult] = []

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    # ── Input Guards ───────────────────────────────────────────────────────
    if req.enable_guards:
        inj = guard_injection(question)
        guardrail_results.append(inj)
        if not inj.passed:
            yield _sse({"type": "blocked", "status": "blocked",
                        "reasoning_type": "blocked_injection", "reason": inj.reason})
            return

        topic = guard_topic(question)
        guardrail_results.append(topic)
        if not topic.passed:
            yield _sse({"type": "blocked", "status": "blocked",
                        "reasoning_type": "blocked_off_topic", "reason": topic.reason})
            return

    # ── Retrieval ──────────────────────────────────────────────────────────
    entities, triples = retrieve(question, top_k=req.top_k, max_hop=req.max_hop)

    if req.enable_guards:
        ev = guard_evidence(triples)
        guardrail_results.append(ev)
        if not ev.passed:
            yield _sse({"type": "blocked", "status": "blocked",
                        "reasoning_type": "blocked_low_evidence", "reason": ev.reason,
                        "entities": entities})
            return

    # ── Streaming Generation ───────────────────────────────────────────────
    token_iter, model_triples = generate_answer_stream(question, triples, entities=entities)
    full_answer = ""
    try:
        for token in token_iter:
            full_answer += token
            yield _sse({"type": "token", "text": token})
    except Exception as exc:
        logger.error("Stream generation failed mid-stream: %s", exc)
        yield _sse({"type": "error", "status": "error", "reason": "LLM 串流中斷，請重試"})
        return

    if full_answer == LLM_ERROR_ANSWER:
        yield _sse({
            "type": "done", "status": "error", "answer": full_answer,
            "reasoning_type": "llm_error", "confidence": 0.0,
            "entities": entities, "evidence_triples": triples,
            "model_triples": model_triples, "source_docs": [],
            "guardrail_results": [r.model_dump() for r in guardrail_results],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        })
        return

    # ── Output Guard (after last token) ───────────────────────────────────
    reasoning_type = "graph_rag"
    confidence = 1.0
    if req.enable_guards:
        gr = guard_grounding(full_answer, model_triples)
        guardrail_results.append(gr)
        if not gr.passed:
            reasoning_type = "answered_with_warning"
            confidence = 0.5

    latency_ms = int((time.perf_counter() - t0) * 1000)

    yield _sse({
        "type": "done",
        "status": "answered",
        "answer": full_answer,
        "reasoning_type": reasoning_type,
        "confidence": confidence,
        "entities": entities,
        "evidence_triples": triples,
        "model_triples": model_triples,
        "source_docs": extract_source_docs(triples),
        "guardrail_results": [r.model_dump(by_alias=True) for r in guardrail_results],
        "latency_ms": latency_ms,
        "request_id": request_id,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blocked(
    req: AskRequest,
    guardrail_results: list[GuardrailResult],
    reasoning_type: str,
    reason: str,
    t0: float,
    entities: list[str] | None = None,
    triples: list[str] | None = None,
) -> AskResponse:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Pipeline blocked | reasoning=%s | latency_ms=%d", reasoning_type, latency_ms
    )
    entities = entities or []
    triples = triples or []
    debug = _make_debug(req, triples, "", {}) if req.debug else None
    return AskResponse(
        question=req.question,
        status="blocked",
        answer=reason,
        entities=entities,
        evidence_triples=triples,
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=0.0,
        debug=debug,
    )


def _make_debug(
    req: AskRequest,
    triples: list[str],
    llm_output: str,
    stage_latencies: dict[str, int] | None = None,
) -> DebugInfo:
    return DebugInfo(
        context="\n".join(triples),
        llm_raw_output=llm_output,
        retrieval_count=len(triples),
        stage_latencies=stage_latencies or {},
    )
