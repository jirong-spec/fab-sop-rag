import logging
import time

from app.schemas import AskRequest, AskResponse, DebugInfo, GuardrailResult
from app.services.guardrails import (
    guard_injection,
    guard_topic,
    guard_evidence,
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
        Input → [guard_injection] → [guard_topic]
             → Retrieval → [guard_evidence]
             → Generation
             → Output → [guard_grounding]
    """
    t0 = time.perf_counter()
    question = req.question
    guardrail_results: list[GuardrailResult] = []

    logger.info(
        "Pipeline start | question=%r | guards=%s | hop=%d | top_k=%d",
        question[:80],
        req.enable_guards,
        req.max_hop,
        req.top_k,
    )

    # ── Input Guards ─────────────────────────────────────────────────────────
    if req.enable_guards:
        inj = guard_injection(question)
        guardrail_results.append(inj)
        logger.info("Guard[injection] pass=%s reason=%s", inj.passed, inj.reason)
        if not inj.passed:
            return _blocked(req, guardrail_results, "blocked_injection", inj.reason, t0)

        topic = guard_topic(question)
        guardrail_results.append(topic)
        logger.info("Guard[topic] pass=%s reason=%s", topic.passed, topic.reason)
        if not topic.passed:
            return _blocked(req, guardrail_results, "blocked_off_topic", topic.reason, t0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    entities, triples = retrieve(question, top_k=req.top_k, max_hop=req.max_hop)
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
                req,
                guardrail_results,
                "blocked_low_evidence",
                ev.reason,
                t0,
                entities=entities,
                triples=triples,
            )

    # ── Generation ────────────────────────────────────────────────────────────
    answer, model_triples = generate_answer(question, triples, entities=entities)
    logger.info("Answer generated | preview=%r", answer[:80])

    # ── Output Guard ──────────────────────────────────────────────────────────
    reasoning_type = "graph_rag"
    confidence = 1.0

    if req.enable_guards:
        gr = guard_grounding(answer, triples)
        guardrail_results.append(gr)
        logger.info("Guard[grounding] pass=%s reason=%s", gr.passed, gr.reason)
        if not gr.passed:
            reasoning_type = "answered_with_warning"
            confidence = 0.5

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Pipeline done | status=answered | reasoning=%s | latency_ms=%d",
        reasoning_type,
        latency_ms,
    )

    debug = _make_debug(req, triples, answer) if req.debug else None

    return AskResponse(
        question=question,
        status="answered",
        answer=answer,
        entities=entities,
        candidate_entities=entities,
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
    debug = _make_debug(req, triples, "") if req.debug else None
    return AskResponse(
        question=req.question,
        status="blocked",
        answer=reason,
        entities=entities,
        candidate_entities=entities,
        evidence_triples=triples,
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=0.0,
        debug=debug,
    )


def _make_debug(req: AskRequest, triples: list[str], llm_output: str) -> DebugInfo:
    return DebugInfo(
        context="\n".join(triples),
        llm_raw_output=llm_output,
        retrieval_count=len(triples),
    )
