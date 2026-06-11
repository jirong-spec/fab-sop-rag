"""
Fab SOP RAG — Streamlit Demo (Graph RAG vs Vector RAG, side by side)
Run: streamlit run demo_app.py
Requires: pip install streamlit requests
"""

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st

GRAPH_URL = "http://localhost:8000/v1/ask"
VECTOR_URL = "http://localhost:8000/v1/ask/vector"

GUARD_LABELS = {
    "injection_detection": "注入偵測",
    "topic_filter": "主題過濾",
    "evidence_sufficiency": "證據充分性",
    "fact_grounding": "事實接地",
}

SAFETY_CATS = {
    "off_topic_blocked",
    "injection_blocked",
    "refusal_unknown_entity",
    "refusal_unknown_step",
}


def _load_queries() -> list[dict]:
    """Load every eval query (dev + test) so the sidebar can offer the full bank."""
    out: list[dict] = []
    qdir = Path(__file__).parent / "data" / "sample_queries"
    for f in sorted(qdir.glob("fab_queries_*.json")):
        try:
            out.extend(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return out


ALL_QUERIES = _load_queries()
# question -> its gold triples ([from, rel, to], answerable questions only)
GOLD_BY_Q = {q["question"]: (q.get("gold_triples") or []) for q in ALL_QUERIES}

st.set_page_config(
    page_title="晶圓廠 SOP 知識查詢",
    page_icon="🏭",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 Fab SOP RAG")
    st.caption("Graph RAG vs Vector RAG")
    st.divider()

    st.subheader("題庫（點擊直接查詢）")
    if ALL_QUERIES:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for q in ALL_QUERIES:
            by_cat[q.get("category", "其他")].append(q)
        ans_cats = [c for c in by_cat if c not in SAFETY_CATS]
        saf_cats = [c for c in by_cat if c in SAFETY_CATS]
        for cat in ans_cats + saf_cats:
            tag = "🚫" if cat in SAFETY_CATS else "✅"
            with st.expander(f"{tag} {cat}（{len(by_cat[cat])}）"):
                for q in by_cat[cat]:
                    if st.button(q["question"], key="q_" + q.get("id", q["question"]), use_container_width=True):
                        st.session_state["pending_question"] = q["question"]
    else:
        st.caption("（找不到 data/sample_queries，題庫為空）")

    st.divider()
    st.subheader("系統資訊")
    st.markdown("""
- **圖譜**：Neo4j（29 節點 / 48 邊）
- **向量庫**：Qdrant（sop_docs collection）
- **LLM**：Qwen2.5-7B-AWQ-int4（vLLM）
- **Embedding**：multilingual-e5-small（reranker: MiniLM-L12-v2）
- **Guardrails**：4 階段（注入偵測 / 主題過濾 / 證據充分性 / 事實接地）
""")

    st.divider()
    st.subheader("API 設定")
    api_key = st.text_input("API Key（選填）", type="password", placeholder="留空則無認證")

    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("晶圓廠 SOP 知識查詢系統")
st.caption("同一問題並排對照：Graph RAG（圖譜 triples）vs Vector RAG（文件 chunks）")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _fetch(url: str, question: str, headers: dict):
    """Run one pipeline. Never calls st.* so it is safe inside a worker thread."""
    try:
        t0 = time.time()
        r = requests.post(url, json={"question": question}, headers=headers, timeout=120)
        ms = round((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}, ms
        return r.json(), ms
    except requests.exceptions.ConnectionError:
        return {"_error": "無法連線 API（localhost:8000）— 請確認 docker compose up 已執行。"}, None
    except Exception as e:  # noqa: BLE001
        return {"_error": f"錯誤：{e}"}, None


def render_compact(data: dict | None, elapsed_ms: int | None) -> None:
    """Render one pipeline's result. Flat (no nested columns) so it fits inside a column."""
    if not data:
        return
    if data.get("_error"):
        st.error(data["_error"])
        return

    status = data.get("status", "")
    if status == "blocked":
        st.error("🚫 已攔截（Guardrail）")
    elif data.get("reasoning_type") == "answered_with_warning":
        st.warning("⚠ 已回答（grounding 有疑慮）")
    else:
        st.success("✅ 已回答")
    if elapsed_ms:
        st.caption(f"⏱ {elapsed_ms} ms")

    st.markdown(data.get("answer", ""))

    # What actually reached the LLM: graph = structured triples; vector = raw chunks.
    triples = data.get("model_triples") or []
    chunks = data.get("evidence_triples") or []
    if triples:
        with st.expander(f"📊 送入 LLM 的圖譜關係（{len(triples)} 條 triples）"):
            for t in triples:
                st.code(t, language="")
    elif chunks:
        with st.expander(f"📄 送入 LLM 的文件片段（{len(chunks)} 段 chunks）"):
            for c in chunks:
                st.code(c, language="")

    # Guardrail trace — which of the 4 guards passed, and where (if any) it was blocked.
    guards = data.get("guardrail_results") or []
    if guards:
        blocked = next((g for g in guards if not g.get("pass")), None)
        summary = (
            f"❌ 擋於 {GUARD_LABELS.get(blocked['name'], blocked['name'])}"
            if blocked
            else "✅ 全通過"
        )
        with st.expander(f"🛡 Guardrails — {summary}"):
            for g in guards:
                icon = "✅" if g.get("pass") else "❌"
                name = GUARD_LABELS.get(g.get("name"), g.get("name", ""))
                st.markdown(f"{icon} **{name}** — {g.get('reason', '')}")


def render_pair(graph, graph_ms, vector, vector_ms) -> None:
    col_g, col_v = st.columns(2)
    with col_g:
        st.markdown("#### 🟢 Graph RAG")
        render_compact(graph, graph_ms)
    with col_v:
        st.markdown("#### ⚪ Vector RAG（baseline）")
        render_compact(vector, vector_ms)


def render_gold_comparison(question: str, graph: dict | None, vector: dict | None) -> None:
    """For questions with gold triples, show which pipeline actually retrieved each one.

    Graph hit  = the gold [from, rel, to] appears as a retrieved triple.
    Vector hit = both gold entities appear in the retrieved chunks (text only — vector has
                 no notion of the *relation*, which is exactly the point).
    """
    gold = GOLD_BY_Q.get(question) or []
    if not gold:
        return
    g_triples = (graph or {}).get("evidence_triples") or []
    v_text = "\n".join((vector or {}).get("evidence_triples") or [])

    rows, g_hits, v_hits = [], 0, 0
    for tri in gold:
        if len(tri) != 3:
            continue
        f, r, t = tri
        g_hit = any(f in s and r in s and t in s for s in g_triples)
        v_hit = bool(f in v_text and t in v_text)
        g_hits += g_hit
        v_hits += v_hit
        rows.append(
            {
                "Gold triple（標準答案邊）": f"({f})-[{r}]->({t})",
                "🟢 Graph 撈到": "✅" if g_hit else "❌",
                "⚪ Vector 文字含實體": "✅" if v_hit else "❌",
            }
        )
    if not rows:
        return
    st.markdown("##### 🔬 Gold triple 檢索對照")
    st.table(rows)
    st.caption(
        f"Graph 撈到 {g_hits}/{len(rows)} 條 gold 邊（結構化關係）　|　"
        f"Vector {v_hits}/{len(rows)} 條（文字裡有實體，但沒有『關係』結構）"
    )


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_pair(msg.get("graph"), msg.get("graph_ms"), msg.get("vector"), msg.get("vector_ms"))
            render_gold_comparison(msg.get("question", ""), msg.get("graph"), msg.get("vector"))
        else:
            st.write(msg["content"])

# Handle sidebar question-bank clicks
pending = st.session_state.pop("pending_question", None)

# Chat input
prompt = st.chat_input("輸入問題，或從左側題庫點一題") or pending

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    headers = {"X-API-Key": api_key} if api_key else {}
    with st.chat_message("assistant"):
        with st.spinner("查詢中（Graph + Vector 並行）…"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_g = ex.submit(_fetch, GRAPH_URL, prompt, headers)
                fut_v = ex.submit(_fetch, VECTOR_URL, prompt, headers)
                graph, graph_ms = fut_g.result()
                vector, vector_ms = fut_v.result()
        render_pair(graph, graph_ms, vector, vector_ms)
        render_gold_comparison(prompt, graph, vector)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "question": prompt,
            "graph": graph,
            "graph_ms": graph_ms,
            "vector": vector,
            "vector_ms": vector_ms,
        }
    )
