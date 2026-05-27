"""
Fab SOP RAG — Streamlit Demo
Run: streamlit run demo_app.py
Requires: pip install streamlit requests
"""

import json
import time

import requests
import streamlit as st

STREAM_URL = "http://localhost:8000/v1/ask/stream"

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

    st.divider()
    st.subheader("API 設定")
    api_key = st.text_input("API Key（選填）", type="password", placeholder="留空則無認證")

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

    badge_col, lat_col = st.columns([3, 1])
    with badge_col:
        if status == "blocked":
            st.error("🚫 已攔截（Guardrail）")
        elif data.get("reasoning_type") == "answered_with_warning":
            st.warning("⚠ 已回答（事實接地性有疑慮，confidence 0.5）")
        else:
            st.success("✅ 已回答")
    with lat_col:
        if elapsed_ms:
            st.metric("延遲", f"{elapsed_ms} ms")

    st.markdown(answer)

    source_docs = data.get("source_docs", [])
    if source_docs:
        st.caption("📄 來源文件：" + "　".join(f"`{d}`" for d in source_docs))

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
        token_placeholder = st.empty()
        accumulated = ""
        final_data = None
        elapsed_ms = None

        try:
            t0 = time.time()
            headers = {"X-API-Key": api_key} if api_key else {}
            with requests.post(
                STREAM_URL,
                json={"question": prompt},
                headers=headers,
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    st.error(f"API 錯誤 {resp.status_code}: {resp.text[:300]}")
                    st.stop()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])

                    if event["type"] == "token":
                        accumulated += event["text"]
                        # ▌ 游標效果讓使用者知道還在輸出
                        token_placeholder.markdown(accumulated + "▌")

                    elif event["type"] == "done":
                        elapsed_ms = round((time.time() - t0) * 1000)
                        final_data = event

                    elif event["type"] == "blocked":
                        elapsed_ms = round((time.time() - t0) * 1000)
                        final_data = {
                            "status": "blocked",
                            "answer": event.get("reason", ""),
                            "reasoning_type": event.get("reasoning_type", "blocked"),
                            "guardrail_results": event.get("guardrail_results", []),
                            "source_docs": [],
                            "model_triples": [],
                        }

                    elif event["type"] == "error":
                        elapsed_ms = round((time.time() - t0) * 1000)
                        final_data = {
                            "status": "blocked",
                            "answer": event.get("reason", "LLM 串流中斷，請重試"),
                            "reasoning_type": "error",
                            "guardrail_results": [],
                            "source_docs": [],
                            "model_triples": [],
                        }

        except requests.exceptions.ConnectionError:
            st.error("無法連線至 API（localhost:8000），請確認 docker compose up 已執行。")
            st.stop()
        except Exception as e:
            st.error(f"API 錯誤：{e}")
            st.stop()

        # 清除串流文字，改用完整 render（含 guardrail 展開區塊）
        token_placeholder.empty()
        if final_data:
            render_response(final_data, elapsed_ms)

    if final_data:
        st.session_state.messages.append({
            "role": "assistant",
            "data": final_data,
            "elapsed_ms": elapsed_ms,
        })
