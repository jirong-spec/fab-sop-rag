"""
SOP-grounded answer generation.

The LLM is constrained to act as a fab SOP lookup assistant — not a general
process engineer. It may only use the supplied graph triples as evidence and
must produce a structured answer (steps, conditions, dependencies) when the
triples support it, or a safe refusal when they do not.

Design principles
-----------------
temperature=0     Deterministic output; reproducibility matters in industrial systems.
max_tokens=512    Enough for a multi-step SOP procedure; prevents verbose hallucination.
Explicit fallback  The refusal phrase is fixed so that guard_grounding can recognise it
                  as a grounded response (no hallucinated content to flag).
"""

import logging

from app.services.llm_client import chat_completion

logger = logging.getLogger(__name__)


def _score_triples(
    question: str, triples: list[str], entities: list[str] | None = None
) -> list[tuple[int, str]]:
    """
    Rank triples by cosine similarity (bi-encoder) to the question.
    Scores are shown as percentages so the LLM can weigh relevance.
    """
    from app.services.vector_store import _get_embeddings
    emb = _get_embeddings()
    q_vec = emb.embed_query(question)
    t_vecs = emb.embed_documents(triples)

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm if norm else 0.0

    scored = sorted(
        [(cosine(q_vec, tv), t) for tv, t in zip(t_vecs, triples)],
        reverse=True,
    )
    return [(round(score * 100), triple) for score, triple in scored]

_NO_INFO_ANSWER = "【查詢結果】此問題的答案不在目前的 SOP 知識圖譜中，無法回答。"
_LLM_ERROR_ANSWER = "（LLM 服務暫時無法使用，請稍後再試）"

_PROMPT_TEMPLATE = """\
你是一位嚴謹的晶圓廠 SOP 知識查詢助理，專門協助製程、設備與整合工程師查詢 SOP 文件。
你只能根據下方「SOP 知識圖譜關係」回答問題，嚴禁推測或編造任何圖譜中未記載的資訊。

【圖譜關係說明】
圖譜中的邊（關係）含義如下，請依題意選擇正確的邊回答：
- TRIGGERS_SOP      ：某異常（Anomaly）觸發應執行的 SOP 文件
- FIRST_STEP        ：SOP 文件的第一個步驟
- NEXT_STEP         ：步驟的下一個步驟（用於呈現步驟順序）
- DEPENDS_ON        ：步驟執行前必須完成的前一個步驟
- REQUIRES_STATUS   ：步驟執行時某設備必須處於的狀態（邊上有 required_status 屬性）
- PRECONDITION      ：整份 SOP 執行前的設備狀態前置條件
- DEFINED_IN        ：步驟所屬的 SOP 文件
- INTERLOCK_WITH    ：設備間的聯鎖關係（含觸發條件與動作）
- CROSS_DOC_DEPENDENCY：SOP 文件間的跨文件依賴（含 reason 屬性說明原因）

【查詢原則】
1. 僅使用圖譜中明確記載的關係作為答案依據，不得補充一般製程常識
2. 若問題詢問「異常應執行哪份 SOP」，請使用 TRIGGERS_SOP 關係回答
3. 若問題詢問步驟順序，請從 FIRST_STEP 出發，沿 NEXT_STEP 鏈依序列出所有步驟，並明確列出每個步驟的節點 ID
4. 若問題詢問設備狀態要求，請使用 REQUIRES_STATUS 或 PRECONDITION 邊的 required_status 屬性回答，並逐一列出每台設備 ID 及其對應狀態值
5. 若問題詢問「哪份文件定義了某設備狀態」，請直接引用 CROSS_DOC_DEPENDENCY 邊的 reason 屬性內容回答，例如：「依據圖譜，SOP_Pump_002 定義了 TurboVacuumPump 的狀態。」
6. 若問題詢問 Interlock 條件，請明確引用圖譜中的 interlock_id、觸發條件（trigger）及執行動作（action），並點名相關設備節點 ID
7. 若問題詢問步驟的前置依賴，請列出所有透過 DEPENDS_ON 連結的前置步驟節點 ID
8. 只有在圖譜中完全找不到任何相關關係時，才回答：「查詢結果：此問題不在目前 SOP 圖譜涵蓋範圍。」否則請根據圖譜回答。
9. 使用繁體中文，回答請簡潔、結構化（可用條列式說明步驟）；引用圖譜中的節點 ID 時直接使用原始英文 ID，不要翻譯

【SOP 知識圖譜關係】（已依與問題的相關性由高到低排列，括號內數字為相關度百分比，優先使用相關度高的關係作答）
{context}

【工程師問題】
{question}

【查詢結果】"""


def generate_answer(
    question: str, triples: list[str], entities: list[str] | None = None
) -> tuple[str, list[str]]:
    """
    Generate a grounded SOP answer from graph triples.

    Returns (answer, model_triples) where model_triples is the subset actually
    passed to the LLM (after rerank + cap). This lets callers distinguish what
    the model saw from the full evidence_triples returned by graph traversal.

    Returns the fixed no-info phrase when triples is empty — this is treated
    as a grounded response by guard_grounding (no hallucinated content).
    """
    if not triples:
        return _NO_INFO_ANSWER, []

    # Rank all triples by relevance, then take the minimum subset that keeps
    # every triple scoring ≥ 50% of the top score (dynamic cap).
    # Floor at 5 triples, hard ceiling at 100 to bound LLM context size.
    scored_all = _score_triples(question, triples, entities=entities)
    if scored_all:
        threshold = max(scored_all[0][0] * 0.50, 20)
        scored = [item for item in scored_all if item[0] >= threshold]
        scored = scored[:100] if len(scored) > 100 else scored
        if len(scored) < 5:
            scored = scored_all[:5]
    else:
        scored = scored_all
    logger.debug("Dynamic cap: %d/%d triples (threshold=%.0f%%)", len(scored), len(triples), threshold if scored_all else 0)
    model_triples = [triple for _, triple in scored]
    context = "\n".join(f"[{pct}%] {triple}" for pct, triple in scored)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        return chat_completion(prompt, temperature=0.0, max_tokens=512), model_triples
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        # Return empty model_triples: the LLM never received them, so callers
        # should not score them as evidence the model actually saw.
        return _LLM_ERROR_ANSWER, []
