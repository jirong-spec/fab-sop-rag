import logging
import time

from app.schemas import AskRequest, AskResponse, DebugInfo, GuardrailResult
from app.services.guardrails import (
    guard_topic,
    guard_grounding,
)
from app.services.retrieval_service import retrieve
from app.services.answer_service import generate_answer
from app.utils.text_utils import extract_source_docs

logger = logging.getLogger(__name__)


def run_pipeline(req: AskRequest) -> AskResponse:
    """
    Orchestrate the full guardrailed RAG pipeline.

    Flow:
        Input → [guard_topic] → Retrieval → Generation → Output → [guard_grounding]
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

    # ── Input Guard ──────────────────────────────────────────────────────────
    if req.enable_guards:
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

    # ── Generation ────────────────────────────────────────────────────────────
    _ts = time.perf_counter()
    answer, model_triples = generate_answer(question, triples, entities=entities)
    stage_latencies["generation"] = int((time.perf_counter() - _ts) * 1000)
    logger.info("Answer generated | preview=%r", answer[:80])

    # ── Output Guard ──────────────────────────────────────────────────────────
    reasoning_type = "graph_rag"
    confidence = 1.0

    if req.enable_guards:
        _ts = time.perf_counter()
        gr = guard_grounding(answer, triples)
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
