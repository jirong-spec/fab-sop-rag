"""
Baseline vector-only RAG pipeline for comparison with Graph RAG.

This pipeline uses only Chroma vector similarity search to retrieve
text chunks, then generates an answer directly from those chunks.
It deliberately omits graph expansion to serve as a fair baseline
showing what a standard RAG system achieves on the same queries.

Comparison axis
---------------
  baseline_rag    : vector retrieval → text chunks → LLM answer
  graph_rag       : vector retrieval → entity extraction → graph
                    expansion → structured triples → LLM answer

Multi-hop queries (step sequences, cross-doc dependencies) are
expected to show the largest performance gap because those
relationships only become explicit after graph traversal.
"""

import logging
import time

from app.schemas import AskRequest, AskResponse, DebugInfo, GuardrailResult
from app.services.guardrails import guard_injection, guard_topic
from app.services.vector_store import similarity_search
from app.services.llm_client import chat_completion
from app.utils.text_utils import extract_source_docs

logger = logging.getLogger(__name__)

_NO_INFO_ANSWER = "【查詢結果】此問題的答案不在目前的 SOP 文件庫中，無法回答。"
_LLM_ERROR_ANSWER = "（LLM 服務暫時無法使用，請稍後再試）"

_BASELINE_PROMPT = """\
你是一位晶圓廠 SOP 文件查詢助理。
請根據下方檢索到的文件片段，以繁體中文回答工程師的問題。
僅根據文件內容作答，不得補充文件外的資訊或推測。
若文件片段中找不到相關資訊，請回答：「查詢結果：無足夠資訊回答此問題。」

【文件片段】
{context}

【工程師問題】
{question}

【查詢結果】"""


def _generate_baseline_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return _NO_INFO_ANSWER
    context = "\n\n---\n\n".join(chunks[:5])
    prompt = _BASELINE_PROMPT.format(context=context, question=question)
    try:
        return chat_completion(prompt, temperature=0.0, max_tokens=512)
    except Exception as exc:
        logger.error("Baseline answer generation failed: %s", exc)
        return _LLM_ERROR_ANSWER


def run_baseline_pipeline(req: AskRequest) -> AskResponse:
    """
    Baseline vector-only RAG pipeline (no knowledge-graph expansion).

    Flow:
        Input → [guard_injection] → [guard_topic]
             → Vector Retrieval → [evidence check]
             → Generation (text chunks as context)
    """
    t0 = time.perf_counter()
    question = req.question
    guardrail_results: list[GuardrailResult] = []

    logger.info(
        "Baseline pipeline start | question=%r | guards=%s | top_k=%d",
        question[:80],
        req.enable_guards,
        req.top_k,
    )

    # ── Input Guards ──────────────────────────────────────────────────────────
    if req.enable_guards:
        inj = guard_injection(question)
        guardrail_results.append(inj)
        if not inj.passed:
            return _blocked(req, guardrail_results, "blocked_injection", inj.reason, t0)

        topic = guard_topic(question)
        guardrail_results.append(topic)
        if not topic.passed:
            return _blocked(req, guardrail_results, "blocked_off_topic", topic.reason, t0)

    # ── Vector retrieval only (no graph) ─────────────────────────────────────
    chunks = similarity_search(question, k=req.top_k)
    logger.info("Baseline retrieval | chunks=%d", len(chunks))

    # ── Evidence check ───────────────────────────────────────────────────────
    if req.enable_guards:
        if len(chunks) == 0:
            ev = GuardrailResult(
                stage="retrieval",
                name="evidence_sufficiency",
                passed=False,
                reason="未檢索到任何文件片段，拒絕生成以避免幻覺",
            )
            guardrail_results.append(ev)
            return _blocked(req, guardrail_results, "blocked_low_evidence", ev.reason, t0)
        ev = GuardrailResult(
            stage="retrieval",
            name="evidence_sufficiency",
            passed=True,
            reason=f"檢索到 {len(chunks)} 個文件片段，證據充足",
        )
        guardrail_results.append(ev)

    # ── Generation ────────────────────────────────────────────────────────────
    answer = _generate_baseline_answer(question, chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Baseline pipeline done | latency_ms=%d | preview=%r",
        latency_ms,
        answer[:80],
    )

    debug = DebugInfo(
        context="\n---\n".join(chunks),
        llm_raw_output=answer,
        retrieval_count=len(chunks),
    ) if req.debug else None

    return AskResponse(
        question=question,
        status="answered",
        answer=answer,
        entities=[],
        candidate_entities=[],
        evidence_triples=chunks,
        source_docs=extract_source_docs(chunks),
        guardrail_results=guardrail_results,
        reasoning_type="baseline_rag",
        confidence=1.0,
        debug=debug,
    )


def _blocked(
    req: AskRequest,
    guardrail_results: list[GuardrailResult],
    reasoning_type: str,
    reason: str,
    t0: float,
) -> AskResponse:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("Baseline pipeline blocked | reasoning=%s | latency_ms=%d", reasoning_type, latency_ms)
    return AskResponse(
        question=req.question,
        status="blocked",
        answer=reason,
        entities=[],
        candidate_entities=[],
        evidence_triples=[],
        guardrail_results=guardrail_results,
        reasoning_type=reasoning_type,
        confidence=0.0,
    )
