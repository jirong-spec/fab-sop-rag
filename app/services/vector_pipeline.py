"""
Traditional (vector-only) RAG pipeline — latency baseline.

Flow: question → Qdrant similarity search → LLM generation
Same 4 guardrails as the Graph RAG pipeline so the comparison is apples-to-apples.
"""

import logging
import time

from app.schemas import AskRequest, AskResponse, GuardrailResult
from app.services.answer_service import LLM_ERROR_ANSWER
from app.services.guardrails import guard_injection, guard_topic, guard_grounding
from app.services.llm_client import chat_completion
from app.services.vector_store import similarity_search

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
你是一位晶圓廠 SOP 查詢助理。請根據下方 SOP 文件內容回答工程師的問題。
只能使用提供的文件內容作答，不得推測或補充文件中未記載的資訊。
若文件中找不到相關資訊，請回答：「查詢結果：此問題不在目前 SOP 文件涵蓋範圍。」

【SOP 文件片段】
{context}

【工程師問題】
{question}

【查詢結果】"""


def _guard_evidence(chunks: list[str]) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(
            stage="retrieval", name="evidence_sufficiency", passed=False,
            reason="未檢索到任何 SOP 文件片段，拒絕生成以避免幻覺",
        )
    return GuardrailResult(
        stage="retrieval", name="evidence_sufficiency", passed=True,
        reason=f"檢索到 {len(chunks)} 個 SOP 文件片段，證據充足",
    )


def run_vector_pipeline(req: AskRequest) -> tuple[AskResponse, dict[str, int]]:
    """
    Returns (AskResponse, stage_latencies).
    stage_latencies keys: guard_injection, guard_topic, retrieval, generation, guard_grounding
    """
    t0 = time.perf_counter()
    question = req.question
    guardrail_results: list[GuardrailResult] = []
    stage_latencies: dict[str, int] = {}

    # ── Input Guards ──────────────────────────────────────────────────────────
    if req.enable_guards:
        _ts = time.perf_counter()
        inj = guard_injection(question)
        stage_latencies["guard_injection"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(inj)
        if not inj.passed:
            return _blocked(req, guardrail_results, "blocked_injection", inj.reason), stage_latencies

        _ts = time.perf_counter()
        topic = guard_topic(question)
        stage_latencies["guard_topic"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(topic)
        if not topic.passed:
            return _blocked(req, guardrail_results, "blocked_off_topic", topic.reason), stage_latencies

    # ── Retrieval (Qdrant only) ───────────────────────────────────────────────
    _ts = time.perf_counter()
    chunks = similarity_search(question, k=6)
    stage_latencies["retrieval"] = int((time.perf_counter() - _ts) * 1000)

    # ── Evidence Guard ────────────────────────────────────────────────────────
    if req.enable_guards:
        ev = _guard_evidence(chunks)
        guardrail_results.append(ev)
        if not ev.passed:
            return _blocked(req, guardrail_results, "blocked_low_evidence", ev.reason), stage_latencies

    # ── Generation ────────────────────────────────────────────────────────────
    context = "\n\n---\n\n".join(chunks)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    _ts = time.perf_counter()
    try:
        answer = chat_completion(prompt, temperature=0.0, max_tokens=512)
    except Exception as exc:
        logger.error("Vector RAG generation failed: %s", exc)
        answer = LLM_ERROR_ANSWER
    stage_latencies["generation"] = int((time.perf_counter() - _ts) * 1000)

    if answer == LLM_ERROR_ANSWER:
        return _blocked(req, guardrail_results, "llm_error", answer), stage_latencies

    # ── Output Guard ──────────────────────────────────────────────────────────
    reasoning_type = "vector_rag"
    confidence = 1.0
    if req.enable_guards:
        _ts = time.perf_counter()
        gr = guard_grounding(answer, chunks)
        stage_latencies["guard_grounding"] = int((time.perf_counter() - _ts) * 1000)
        guardrail_results.append(gr)
        if not gr.passed:
            reasoning_type = "answered_with_warning"
            confidence = 0.5

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("Vector pipeline done | latency_ms=%d stages=%s", latency_ms, stage_latencies)

    return AskResponse(
        question=question,
        status="answered",
        answer=answer,
        evidence_triples=chunks,
        model_triples=[],
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=confidence,
    ), stage_latencies


def _blocked(req, guardrail_results, reasoning_type, reason) -> AskResponse:
    return AskResponse(
        question=req.question,
        status="blocked",
        answer=reason,
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=0.0,
    )
