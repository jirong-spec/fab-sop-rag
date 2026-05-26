"""
LLM-as-judge services for topic relevance and fact grounding.

Both judges are called with temperature=0 for deterministic verdicts.
Both have explicit fallback policies (configurable) for when the LLM
returns unparseable output or times out.

Industrial context
------------------
In a wafer fab SOP system, incorrect guardrail verdicts have asymmetric costs:
  - A false-negative topic pass lets irrelevant questions consume compute.
  - A false-negative grounding pass lets a hallucinated SOP step reach an
    engineer during fault isolation — the higher-risk failure mode.

For this reason, grounding_fallback_policy defaults to "strict" (conservative),
while topic_fallback_policy defaults to "lenient" (suitable for demos).
"""

import logging

from app.config import settings
from app.services.llm_client import chat_completion
from app.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


# ── Topic relevance ───────────────────────────────────────────────────────────

_TOPIC_PROMPT = """\
你是一位晶圓廠 SOP 文件庫的主題過濾助理。
你的任務是判斷工程師的問題是否屬於「晶圓廠 SOP 知識庫」的查詢範疇。

本知識庫的查詢範疇涵蓋以下七類：
1. 製程異常處置
   （壓力異常、溫度超標、真空度不足、氣體流量偏移、粒子污染事件等）
2. SOP 操作步驟與流程
   （異常排查程序、前置確認步驟、操作順序、interlocking 條件觸發）
3. 設備與機台狀態條件
   （pump 狀態、腔體 interlock、感測器讀值閾值、RF 功率條件）
4. 步驟依賴關係
   （PRECONDITION、NEXT_STEP、DEPENDS_ON、REQUIRES_STATUS 等圖譜關係）
5. 製程條件與 recipe 參數
   （溫度、壓力、氣體種類與流量、偏壓、腔體清潔週期）
6. 良率異常與排查流程
   （缺陷模式分析、製程能力異常 Cpk、晶圓抽驗結果對照 SOP）
7. 跨文件設備依賴關係
   （某 SOP 步驟所引用的機台狀態定義於另一份 SOP 的情況）

不屬於本知識庫範疇的例子：
- 一般程式設計、數學、物理化學原理（非 SOP 內容）
- 市場行情、財務或人事管理
- 廠外標準（客戶規格、IP 法規等）

問題：{question}

請判斷此問題是否屬於上述晶圓廠 SOP 知識庫範疇。
請僅回傳 JSON，不要有其他文字：
{{"relevant": true, "reason": "簡短說明（20字以內）"}}
或
{{"relevant": false, "reason": "簡短說明（20字以內）"}}"""


def judge_topic_relevance(question: str) -> dict[str, bool | str]:
    """
    Judge whether the question belongs to the wafer fab SOP knowledge domain.

    Returns {"relevant": bool, "reason": str}.

    Fallback policy (TOPIC_FALLBACK_POLICY env var):
      "lenient" — allow with warning (default; suitable for demos and PoC)
      "strict"  — block; appropriate when the knowledge base boundary must
                  be enforced even at the cost of some false rejections
    """
    prompt = _TOPIC_PROMPT.format(question=question)
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=128)
        data = extract_json(raw)
        if data is not None:
            return {
                "relevant": bool(data.get("relevant", False)),
                "reason": str(data.get("reason", "")),
            }
        logger.warning("judge_topic_relevance: JSON parse failed, raw=%r", raw[:120])
    except Exception as exc:
        logger.warning("judge_topic_relevance: LLM call failed: %s", exc)

    if settings.topic_fallback_policy == "strict":
        logger.warning("Topic judge fallback → STRICT block")
        return {"relevant": False, "reason": "（主題判斷失敗，基於嚴格政策拒絕）"}
    logger.warning("Topic judge fallback → LENIENT allow with warning")
    return {"relevant": True, "reason": "（主題判斷失敗，寬鬆模式下放行，建議檢查 LLM 服務）"}


# ── Evidence relevance ───────────────────────────────────────────────────────

_EVIDENCE_PROMPT = """\
你是一位 SOP 查詢系統的證據評估助理。
判斷下方 SOP 圖譜三元組是否包含足以回答工程師問題的相關證據。

【工程師問題】
{question}

【檢索到的 SOP 圖譜三元組】
{triples}

判斷標準：
1. 三元組中是否有任何一條與問題直接相關（如問步驟順序 → 是否有 FIRST_STEP / NEXT_STEP；問設備狀態 → 是否有 REQUIRES_STATUS / PRECONDITION；問異常處置 → 是否有 TRIGGERS_SOP）
2. 若所有三元組均與問題無關，或僅為噪音（來自不相關的 SOP 或設備），則視為證據不足
3. 只要有至少一條三元組能直接支撐問題的部分答案，即視為證據充足

請僅回傳 JSON，不要有其他文字：
{{"sufficient": true, "reason": "簡短說明（20字以內）"}}
或
{{"sufficient": false, "reason": "說明為何圖譜中無相關證據（20字以內）"}}"""


def judge_evidence_relevance(question: str, triples: list[str]) -> dict[str, bool | str]:
    """
    Judge whether the retrieved triples contain evidence relevant to the question.

    Pre-filters to top-10 by embedding similarity before sending to LLM,
    so the model judges a focused set rather than 40+ raw triples.

    Returns {"sufficient": bool, "reason": str}.

    Fallback policy: lenient — pass with warning when LLM is unavailable,
    to avoid blocking valid queries due to LLM service issues.
    """
    # Pre-filter: rerank and keep top-10 so LLM judges the most relevant triples
    try:
        from app.services.answer_service import _score_triples
        scored = _score_triples(question, triples)
        top_triples = [t for _, t in scored[:10]]
    except Exception:
        top_triples = triples[:10]

    context = "\n".join(top_triples)
    prompt = _EVIDENCE_PROMPT.format(question=question, triples=context)
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=128)
        data = extract_json(raw)
        if data is not None:
            return {
                "sufficient": bool(data.get("sufficient", True)),
                "reason": str(data.get("reason", "")),
            }
        logger.warning("judge_evidence_relevance: JSON parse failed, raw=%r", raw[:120])
    except Exception as exc:
        logger.warning("judge_evidence_relevance: LLM call failed: %s", exc)

    logger.warning("Evidence judge fallback → LENIENT pass")
    return {"sufficient": True, "reason": "（證據評估失敗，寬鬆模式下放行）"}


# ── Fact grounding ────────────────────────────────────────────────────────────

_GROUNDING_PROMPT = """\
你是一位嚴格的 SOP 事實查核助理。
你的任務是判斷 LLM 生成的查詢結果是否完全基於下方 SOP 圖譜證據。

SOP 圖譜證據：
{context}

LLM 生成的查詢結果：
{answer}

查核原則：
1. 答案中引述的 SOP 步驟（FIRST_STEP、NEXT_STEP 等）是否有圖譜依據
2. 答案中提到的設備狀態要求（REQUIRES_STATUS、DEPENDS_ON）是否有圖譜依據
3. 答案中的前置條件（PRECONDITION、INTERLOCK_WITH）是否有圖譜依據
4. 若答案包含任何超出圖譜的推論、一般製程常識補充或猜測，視為未接地（grounded: false）
5. 拒答句型（答案中含「查詢結果：此問題不在目前 SOP 圖譜涵蓋範圍」或「無足夠資訊」等）視為 grounded: true（無幻覺內容）

工業場景說明：晶圓廠 SOP 中一個錯誤的步驟順序或設備狀態要求，
可能導致工程師在排障過程中採取錯誤行動。本查核採用保守策略。

請僅回傳 JSON，不要有其他文字：
{{"grounded": true, "reason": "簡短說明（20字以內）"}}
或
{{"grounded": false, "reason": "指出哪些陳述缺乏 SOP 圖譜依據（30字以內）"}}"""


def judge_grounding(answer: str, triples: list[str]) -> dict[str, bool | str]:
    """
    Verify that every claim in `answer` is supported by the retrieved SOP triples.

    Returns {"grounded": bool, "reason": str}.

    Fallback policy (GROUNDING_FALLBACK_POLICY env var):
      "strict"  — treat as ungrounded (default; conservative for industrial use)
      "lenient" — treat as grounded with warning

    The strict default reflects the asymmetric cost of a false-negative grounding
    pass in a fab SOP context: a hallucinated step in a fault-handling procedure
    could mislead an engineer during a time-critical troubleshooting sequence.
    """
    context = "\n".join(triples) if triples else "（無圖譜證據）"
    prompt = _GROUNDING_PROMPT.format(context=context, answer=answer)
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=256)
        data = extract_json(raw)
        if data is not None:
            return {
                "grounded": bool(data.get("grounded", False)),
                "reason": str(data.get("reason", "")),
            }
        logger.warning("judge_grounding: JSON parse failed, raw=%r", raw[:120])
    except Exception as exc:
        logger.warning("judge_grounding: LLM call failed: %s", exc)

    if settings.grounding_fallback_policy == "lenient":
        logger.warning("Grounding judge fallback → LENIENT allow")
        return {"grounded": True, "reason": "（事實查核失敗，寬鬆模式下放行）"}
    logger.warning("Grounding judge fallback → STRICT ungrounded")
    return {"grounded": False, "reason": "（事實查核失敗，基於保守策略標記為未接地）"}
