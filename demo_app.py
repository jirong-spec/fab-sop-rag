"""
Fab SOP RAG — Streamlit Demo
Run: streamlit run demo_app.py
Requires: pip install streamlit requests
"""

import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/v1/ask"

EXAMPLE_QUESTIONS = [
    "蝕刻站發生壓力異常時，應該執行哪份 SOP？",
    "SOP_Etch_001 的步驟順序為何？",
    "執行 SOP_Etch_001 前，TurboVacuumPump 需要是什麼狀態？",
    "EtchStation 的壓力 Interlock 在什麼條件下觸發？",
    "SOP_Etch_001 中 TurboVacuumPump 的狀態定義在哪份文件？",
    "Edwards nXDS 泵浦啟動的步驟順序是什麼？",
    "PumpOverheat 異常應執行哪個 SOP？",
]

st.set_page_config(
    page_title="晶圓廠 SOP 知識查詢",
    page_icon="🏭",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 Fab SOP RAG")
    st.caption("Graph RAG × Knowledge Graph")
    st.divider()

    st.subheader("範例問題")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["pending_question"] = q

    st.divider()
    st.subheader("系統資訊")
    st.markdown("""
- **圖譜**：Neo4j（57 節點 / 92 邊）
- **LLM**：Qwen2.5-7B-AWQ-int4（vLLM）
- **Embedding**：paraphrase-multilingual-MiniLM-L12-v2
- **Guardrails**：4 階段（注入偵測 / 主題過濾 / 證據充分性 / 事實接地）
""")

    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("晶圓廠 SOP 知識查詢系統")
st.caption("輸入問題查詢蝕刻站 SOP、設備狀態、Interlock 條件、跨文件依賴")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_response(data: dict, elapsed_ms: int | None = None) -> None:
    status = data.get("status", "")
    answer = data.get("answer", "")

    # Status badge + latency
    badge_col, lat_col = st.columns([3, 1])
    with badge_col:
        if status == "blocked":
            st.error("🚫 已攔截（Topic Guard）")
        elif status == "answered":
            st.success("✅ 已回答")
        else:
            st.warning(f"⚠ {status}")
    with lat_col:
        if elapsed_ms:
            st.metric("延遲", f"{elapsed_ms} ms")

    # Answer
    st.markdown(answer)

    # Source docs
    source_docs = data.get("source_docs", [])
    if source_docs:
        st.caption("📄 來源文件：" + "　".join(f"`{d}`" for d in source_docs))

    # Evidence details (collapsible)
    model_triples = data.get("model_triples", [])
    guardrail_results = data.get("guardrail_results", [])

    col1, col2 = st.columns(2)
    with col1:
        if model_triples:
            with st.expander(f"📊 送入 LLM 的圖譜關係（{len(model_triples)} 條）"):
                for t in model_triples:
                    st.code(t, language="")
    with col2:
        if guardrail_results:
            with st.expander("🛡 Guardrail 結果"):
                for g in guardrail_results:
                    icon = "✅" if g.get("pass") else "❌"
                    st.markdown(f"{icon} **{g['name']}** — {g.get('reason', '')}")


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(msg["data"], msg.get("elapsed_ms"))
        else:
            st.write(msg["content"])

# Handle sidebar example button clicks
pending = st.session_state.pop("pending_question", None)

# Chat input
prompt = st.chat_input("輸入問題，例如：蝕刻站壓力異常應執行哪份 SOP？") or pending

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("查詢知識圖譜中..."):
            try:
                t0 = time.time()
                resp = requests.post(API_URL, json={"question": prompt}, timeout=120)
                elapsed_ms = round((time.time() - t0) * 1000)
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("無法連線至 API（localhost:8000），請確認 docker compose up 已執行。")
                st.stop()
            except Exception as e:
                st.error(f"API 錯誤：{e}")
                st.stop()

        render_response(data, elapsed_ms)

    st.session_state.messages.append({
        "role": "assistant",
        "data": data,
        "elapsed_ms": elapsed_ms,
    })
