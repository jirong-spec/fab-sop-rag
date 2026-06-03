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
import re
from collections.abc import Iterator

from app.config import settings
from app.services.llm_client import chat_completion, chat_completion_stream

logger = logging.getLogger(__name__)

# Generation completion budget; also reserved when trimming the prompt to fit
# the model context window (see _fit_context_to_budget).
GEN_MAX_TOKENS = 512


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free, conservative token estimate for mixed CJK/ASCII.
    CJK chars ≈ 1 token each; other chars ≈ 3 chars/token. Overestimates slightly
    (safe — we'd rather trim one extra triple than overflow the context window)."""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk + (len(text) - cjk + 2) // 3


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return dot / norm if norm else 0.0


def _score_triples(question: str, triples: list[str]) -> list[tuple[int, str]]:
    """
    Rank triples by cosine similarity (bi-encoder) to the question.
    Scores are shown as percentages so the LLM can weigh relevance.
    """
    from app.services.vector_store import _get_reranker_embeddings

    emb = _get_reranker_embeddings()
    q_vec = emb.embed_query(question)
    t_vecs = emb.embed_documents(triples)
    scored = sorted(
        [(_cosine(q_vec, tv), t) for tv, t in zip(t_vecs, triples, strict=False)],
        reverse=True,
    )
    return [(round(score * 100), triple) for score, triple in scored]


_NO_INFO_ANSWER = "此問題的答案不在目前的 SOP 知識圖譜中，無法回答。"
_LLM_ERROR_ANSWER = "（LLM 服務暫時無法使用，請稍後再試）"
LLM_ERROR_ANSWER = _LLM_ERROR_ANSWER  # exported sentinel for pipeline error detection

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
3. 若問題詢問步驟順序，請從該 SOP 的 FIRST_STEP 出發，嚴格沿 NEXT_STEP 鏈依序列出步驟，只列出 DEFINED_IN 被問 SOP 的步驟節點 ID；其他 SOP 的步驟即使出現在圖譜關係中也一律忽略
4. 若問題詢問設備狀態要求，請使用 REQUIRES_STATUS 或 PRECONDITION 邊的 required_status 屬性回答，並逐一列出每台設備 ID 及其對應狀態值
5. 若問題詢問「哪份文件定義了某設備狀態」，請直接引用 CROSS_DOC_DEPENDENCY 邊的 reason 屬性內容回答，例如：「依據圖譜，SOP_Pump_002 定義了 TurboVacuumPump 的狀態。」
6. 若問題詢問 Interlock 條件，每條 INTERLOCK_WITH 邊必須依照下列格式回答：「來源設備 → 目標設備（interlock_id: XXX，觸發條件: XXX，動作: XXX）」，來源與目標設備節點 ID 均不得省略
7. 若問題詢問步驟的前置依賴，請列出所有透過 DEPENDS_ON 連結的前置步驟節點 ID
8. 只有在圖譜中完全找不到任何相關關係時，才回答：「查詢結果：此問題不在目前 SOP 圖譜涵蓋範圍。」否則請根據圖譜回答。
9. 使用繁體中文，回答請簡潔、結構化（可用條列式說明步驟）；引用圖譜中的節點 ID 時直接使用原始英文 ID，不要翻譯

【SOP 知識圖譜關係】（已依與問題的相關性由高到低排列，括號內數字為相關度百分比，優先使用相關度高的關係作答）
{context}

【工程師問題】
{question}

在輸出答案前，請先在心中逐一核對每條圖譜關係是否與問題相關，確認所有相關屬性（如 required_status、trigger、action、reason、interlock_id）都已納入最終答案；若涉及 INTERLOCK_WITH，請額外確認已同時列出來源與目標設備節點 ID，再輸出【查詢結果】。

【查詢結果】"""


def _prepare_generation(question: str, triples: list[str]) -> tuple[str, list[str]]:
    """
    Rerank triples, apply dynamic cap + SOP filter, build the LLM prompt.
    Returns (prompt, model_triples).  Pure CPU/embedding — no LLM call.
    """
    scored_all = _score_triples(question, triples)
    threshold = 0
    if scored_all:
        threshold = max(int(scored_all[0][0] * 0.50), 20)
        scored = [item for item in scored_all if item[0] >= threshold]
        scored = scored[:100] if len(scored) > 100 else scored
        if len(scored) < 5:
            scored = scored_all[:5]
    else:
        scored = scored_all
    logger.debug("Dynamic cap: %d/%d triples (threshold=%.0f%%)", len(scored), len(triples), threshold)
    model_triples = [triple for _, triple in scored]

    sop_ids = re.findall(r"SOP_\w+", question)
    if sop_ids:
        foreign_steps: set[str] = set()
        for t in model_triples:
            m_step = re.search(r"^\((\w+)[\[\)]", t)
            m_sop = re.search(r"-\[:DEFINED_IN[^\]]*\]->\((\w+)", t)
            if m_step and m_sop and m_sop.group(1).startswith("SOP_") and m_sop.group(1) not in sop_ids:
                foreign_steps.add(m_step.group(1))

        foreign_step_pat = (
            re.compile(r"[\(\)](" + "|".join(re.escape(s) for s in foreign_steps) + r")[\[\)\-]")
            if foreign_steps
            else None
        )

        def _is_foreign(triple: str) -> bool:
            m = re.search(r"^\((\w+)[\[\)]", triple)
            if m and m.group(1).startswith("SOP_") and m.group(1) not in sop_ids:
                return True
            return bool(foreign_step_pat and foreign_step_pat.search(triple))

        filtered = [t for t in model_triples if not _is_foreign(t)]
        if len(filtered) >= 3:
            logger.debug("SOP filter: %d→%d triples (sop_ids=%s)", len(model_triples), len(filtered), sop_ids)
            model_triples = filtered
            scored = [(pct, t) for pct, t in scored if t in set(filtered)]

        if re.search(r"步驟|順序|流程", question):
            existing = set(model_triples)
            score_lookup = {t: s for s, t in scored_all}
            supplements = [t for t in triples if "NEXT_STEP" in t and t not in existing and not _is_foreign(t)]
            if supplements:
                logger.debug("NEXT_STEP supplement: +%d triples", len(supplements))
                for t in supplements:
                    model_triples.append(t)
                    scored.append((score_lookup.get(t, 0), t))

    # ── Token-budget trim ────────────────────────────────────────────────────
    # Cap by triple COUNT (above) is not enough: on a dense graph 100 triples can
    # blow the model context window (HTTP 400). Keep the highest-ranked triples
    # that fit within (context_window − completion − chat-template margin).
    scored = _fit_context_to_budget(question, scored)
    model_triples = [triple for _, triple in scored]

    context = "\n".join(f"[{pct}%] {triple}" for pct, triple in scored)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    return prompt, model_triples


def _fit_context_to_budget(question: str, scored: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Greedily keep top-ranked triples whose serialized lines fit the context budget."""
    overhead = _estimate_tokens(_PROMPT_TEMPLATE.format(context="", question=question))
    budget = settings.llm_max_model_len - GEN_MAX_TOKENS - 256  # 256 ≈ chat template + safety
    used = overhead
    kept: list[tuple[int, str]] = []
    for pct, triple in scored:
        cost = _estimate_tokens(f"[{pct}%] {triple}") + 1
        if used + cost > budget and kept:
            logger.warning(
                "Context budget reached: kept %d/%d triples (~%d tokens, budget %d)",
                len(kept),
                len(scored),
                used,
                budget,
            )
            break
        used += cost
        kept.append((pct, triple))
    return kept


def generate_answer(question: str, triples: list[str]) -> tuple[str, list[str]]:
    """
    Generate a grounded SOP answer from graph triples (synchronous).
    Returns (answer, model_triples).
    """
    if not triples:
        return _NO_INFO_ANSWER, []
    try:
        prompt, model_triples = _prepare_generation(question, triples)
        return chat_completion(prompt, temperature=0.0, max_tokens=GEN_MAX_TOKENS), model_triples
    except Exception as exc:
        logger.error("Answer generation failed: %s", exc)
        return _LLM_ERROR_ANSWER, []


def generate_answer_stream(question: str, triples: list[str]) -> tuple[Iterator[str], list[str]]:
    """
    Prepare generation and return (token_iterator, model_triples).

    model_triples is returned immediately (before streaming starts) so the
    caller can start guard_grounding as soon as the stream finishes.

    Usage:
        token_iter, model_triples = generate_answer_stream(question, triples)
        for token in token_iter:
            ...stream to client...
    """
    if not triples:

        def _empty():
            yield _NO_INFO_ANSWER

        return _empty(), []

    try:
        prompt, model_triples = _prepare_generation(question, triples)
        return chat_completion_stream(prompt, temperature=0.0, max_tokens=GEN_MAX_TOKENS), model_triples
    except Exception as exc:
        logger.error("Answer stream preparation failed: %s", exc)

        def _error():
            yield _LLM_ERROR_ANSWER

        return _error(), []
