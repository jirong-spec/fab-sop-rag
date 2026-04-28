"""
Four guardrail functions for the wafer fab SOP knowledge query pipeline.

Industrial context
------------------
In a fab SOP system, guardrail failures have concrete engineering consequences:

  guard_injection   — prevents adversarial inputs from hijacking the LLM's
                      role as a strict SOP lookup assistant.

  guard_topic       — ensures questions stay within the SOP knowledge domain;
                      off-topic answers waste compute and may confuse engineers
                      who expect domain-specific responses.

  guard_evidence    — blocks answers when the graph has no supporting triples.
                      Answering "what does SOP step 3 say" with zero retrieved
                      evidence would produce pure hallucination — unacceptable
                      in a system engineers rely on during fault isolation.

  guard_grounding   — verifies the final answer against retrieved triples.
                      A hallucinated SOP step (wrong sequence, wrong equipment
                      pre-check) could lead an engineer to take an incorrect
                      action during a time-critical troubleshooting procedure.
                      This guard prefers "answered_with_warning" over silent
                      hallucination pass-through.

Guardrail results are always returned (never raise), so the pipeline can
collect a full trace even when a stage blocks.
"""

import re
import logging

from app.schemas import GuardrailResult
from app.services.judge_service import judge_topic_relevance, judge_grounding

# SOP entity code pattern: SOP_Etch_001, CheckVacuumPump, TurboVacuumPump, etc.
# If the question explicitly references a known fab SOP entity, it is unambiguously
# in-domain — bypass the LLM judge to avoid false negatives from small models.
_SOP_ENTITY_RE = re.compile(r"[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")

logger = logging.getLogger(__name__)

# ── Injection detection patterns ──────────────────────────────────────────────
# Matched case-insensitively.  Coverage:
#   English  — instruction override, system-prompt references, jailbreak keywords,
#              role-play / persona hijack
#   Chinese  — instruction override, role-play / persona hijack (Traditional & Simplified)
INJECTION_PATTERNS: list[str] = [
    # English – instruction override
    r"ignore\s+(all\s+)?(previous|above|prior)\s*(instructions?|prompts?)?",
    r"forget\s+(all\s+)?(previous|above|prior|your|my)\s*(instructions?|prompts?)?",
    r"disregard\s+(all\s+)?(previous|above|prior)\s*(instructions?|prompts?)?",
    r"override\s+(previous|all|prior)\s*(instructions?|prompts?|settings?)",
    # English – system / meta-prompt references
    r"system\s*prompt",
    r"your\s+(original\s+)?instructions?",
    r"initial\s+prompt",
    # English – jailbreak keywords
    r"\bjailbreak\b",
    r"\bDAN\s*mode\b",
    r"do\s+anything\s+now",
    # English – role-play / persona hijack
    r"role[\-\s]?play",
    r"pretend\s+(you\s+(are|were)|to\s+be)",
    r"act\s+as\s+(if\s+you\s+(are|were)|a\s+)",
    r"you\s+are\s+now\s+a",
    r"simulate\s+(being|a)\s+",
    # Chinese – instruction override
    r"忽略(之前|上面|前面|所有|一切)(的指示|指示|指令)?",
    r"忘記(之前|上面|前面|所有|一切)(的指示|指示|指令)?",
    r"不要(理會|遵守|遵從)(之前|上面|前面|所有|一切)",
    r"新(的)?指令",
    # Chinese – role-play / persona hijack
    r"你現在是",
    r"扮演",
    r"假裝(你是|自己是)",
    r"以.{1,10}的身份",
]


def guard_injection(question: str) -> GuardrailResult:
    """
    Input guard #1 — rule-based prompt injection detection.

    Scans the question against 20 patterns covering English and Chinese
    instruction-override and persona-hijack exploits.  If any pattern
    matches, the request is blocked before reaching the LLM.
    """
    for pat in INJECTION_PATTERNS:
        if re.search(pat, question, re.IGNORECASE):
            logger.warning(
                "Injection pattern matched: pattern=%r question=%r", pat, question[:80]
            )
            return GuardrailResult(
                stage="input",
                name="injection_detection",
                passed=False,
                reason="偵測到可能的提示注入（prompt injection）",
            )
    return GuardrailResult(
        stage="input",
        name="injection_detection",
        passed=True,
        reason="未偵測到注入模式",
    )


def guard_topic(question: str) -> GuardrailResult:
    """
    Input guard #2 — LLM-as-judge topic relevance filter with rule-based pre-check.

    Rule-based pre-check: if the question contains an explicit SOP entity code
    (e.g. SOP_Etch_001, CheckVacuumPump, TurboVacuumPump) it is unambiguously
    in-domain — pass immediately without calling the LLM judge.  This prevents
    false negatives from small models that may misparse valid SOP queries.

    For questions without explicit entity codes, the LLM judge checks whether the
    question falls within the seven categories of the wafer fab SOP knowledge domain.
    """
    if _SOP_ENTITY_RE.search(question):
        logger.info("Topic pre-check: SOP entity code detected, passing without LLM judge")
        return GuardrailResult(
            stage="input",
            name="topic_filter",
            passed=True,
            reason="問題包含明確的 SOP 實體代碼，屬於知識庫範疇",
        )

    result = judge_topic_relevance(question)
    passed = bool(result.get("relevant", False))
    reason = str(result.get("reason", ""))
    return GuardrailResult(
        stage="input",
        name="topic_filter",
        passed=passed,
        reason=reason,
    )


def guard_evidence(triples: list[str], min_count: int = 1) -> GuardrailResult:
    """
    Retrieval guard — block if retrieved SOP graph evidence is insufficient.

    Answering with zero triples would produce pure hallucination: the LLM
    has no SOP content to ground on and will generate a plausible-sounding
    but fabricated procedure.  This guard prevents that path entirely.

    min_count=1 is conservative; raise it (e.g. to 3) if the knowledge graph
    is dense enough that a single triple provides insufficient context for
    multi-step SOP procedures.
    """
    n = len(triples)
    if n < min_count:
        return GuardrailResult(
            stage="retrieval",
            name="evidence_sufficiency",
            passed=False,
            reason=(
                f"僅檢索到 {n} 筆 SOP 三元組，最少需要 {min_count} 筆，"
                "證據不足，拒絕生成以避免幻覺"
            ),
        )
    return GuardrailResult(
        stage="retrieval",
        name="evidence_sufficiency",
        passed=True,
        reason=f"檢索到 {n} 筆 SOP 三元組，證據充足",
    )


def guard_grounding(answer: str, triples: list[str]) -> GuardrailResult:
    """
    Output guard #4 — LLM-as-judge factual grounding check.

    Verifies that every SOP step, equipment condition, and pre-check
    requirement in the answer has a corresponding graph triple as evidence.

    When grounding fails, the pipeline returns `answered_with_warning`
    (confidence 0.5) rather than silently passing the answer.  Engineers
    should treat such answers as provisional and verify against the source
    SOP document before acting.
    """
    result = judge_grounding(answer, triples)
    passed = bool(result.get("grounded", False))
    reason = str(result.get("reason", ""))
    return GuardrailResult(
        stage="output",
        name="fact_grounding",
        passed=passed,
        reason=reason,
    )
