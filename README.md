# Fab SOP Knowledge Query API

**Guardrailed Hybrid Graph RAG — Single-machine Enterprise MVP**

用自然語言詢問晶圓廠 SOP 文件，系統自動從知識圖譜（Neo4j）和向量庫（Chroma）撈出 SOP 步驟、設備依賴、前置條件，再由 LLM 合成答案，並經過四道 guardrail 過濾。

![image](https://github.com/jirong-spec/fab-sop-rag/blob/main/visualisation.png)

---

## 一、評估結果

### 最終成績（Qwen2.5-7B-Instruct-AWQ-int4，2026-05-13）

| 指標 | Graph RAG | Baseline RAG | 差距 |
|------|-----------|--------------|------|
| **Retrieval 命中率** | **100%** (24/24) | 58.3% (14/24) | **+41.7 pp** |
| **Answer 命中率** | **100%** (24/24) | 54.2% (13/24) | **+45.8 pp** |
| **多跳查詢 Answer** | **100%** (8/8) | 25.0% (2/8) | **+75.0 pp** |
| **正確攔截非 SOP 問題** | 2/2 | 2/2 | — |
| **平均端對端延遲** | 4067 ms | 1348 ms | +2719 ms |

```
ID   │ 類別                        │ R / A        │  延遲
─────┼─────────────────────────────┼──────────────┼────────
q01  │ anomaly_handling            │ R✅ A✅ 2/2  │  4881ms
q02  │ sop_step_sequence  [↑HOP]   │ R✅ A✅ 4/4  │  4349ms
q03  │ equipment_precondition      │ R✅ A✅ 4/4  │  5012ms
q04  │ step_dependency             │ R✅ A✅ 3/3  │  3102ms
q05  │ cross_doc_dependency [↑HOP] │ R✅ A✅ 2/2  │  5404ms
q06  │ interlock_condition         │ R✅ A✅ 3/3  │  6104ms
q07  │ vent_procedure      [↑HOP]  │ R✅ A✅ 2/2  │  4855ms
q08  │ off_topic_blocked           │ ✅ blocked   │   585ms
q09  │ off_topic_blocked           │ ✅ blocked   │   664ms
q10  │ pump_check_sequence         │ R✅ A✅ 4/4  │  5716ms
─────┼─────────────────────────────┼──────────────┼────────
     │ TOTALS                      │ R 100% A 100%│ avg 4067ms
```

- **R（Retrieval）**：預期關鍵字出現在模型實際收到的 triples（`model_triples`）中
- **A（Answer）**：預期關鍵字出現在模型 `answer` 欄位中
- `[↑HOP]` 需要多跳推理（step 鏈、跨文件依賴、DEPENDS_ON 鏈）

### Citation Traceability

每個回應包含 `source_docs` 欄位，自動從 `evidence_triples` 提取引用的 SOP 文件 ID：

```json
{
  "answer": "依據圖譜，SOP_Etch_001 步驟 CheckVacuumPump 需要 TurboVacuumPump 狀態為 RUNNING...",
  "source_docs": ["SOP_Etch_001", "SOP_Pump_002"],
  "evidence_triples": [
    "(SOP_Etch_001)-[:PRECONDITION {required_status: 'RUNNING'}]->(TurboVacuumPump)",
    "(SOP_Etch_001)-[:CROSS_DOC_DEPENDENCY]->(SOP_Pump_002)"
  ]
}
```

---

## 二、技術優化歷程

### Retrieval：Bi-encoder Reranking + Edge Enrichment + 動態 Cap

Graph traversal 對 q06 回傳 ~48 條 triples，但關鍵的 INTERLOCK_WITH triple 排在第 47 位。三項優化讓 R 從 71% → 100%：

| 方法 | cap | R | A | avg 延遲 |
|------|-----|---|---|---------|
| 無 reranking | 25 | ~71% | 71% | 2683 ms |
| Bi-encoder cosine | 50 | 100% | 83% | 3012 ms |
| + Edge enrichment | 50 | 100% | 83% | 3067 ms |
| + 動態 cap | 動態 | 100% | 88% | 2944 ms |
| + Implicit CoT（7B） | 動態 | **100%** | **100%** | **4067 ms** ✅ |

**Edge Enrichment**：`DEPENDS_ON`、`NEXT_STEP` 等邊序列化後只有 CamelCase ID，中文問題的 embedding 向量相似度趨近零。為邊加 `description` 屬性後，DEPENDS_ON triple 從第 47 位升至第 13 位。

**動態 Cap**（`app/services/answer_service.py`）：
```python
threshold = max(top_score * 0.50, 20)
scored = [t for t in scored_all if t[0] >= threshold][:100]
```

### Generation：Implicit Chain-of-Thought

R=100% 後，q05/q06 仍 A⚠——LLM 沒有完整提取 triple 屬性值（`reason`、`interlock_id`、`trigger`、`action`）。測試兩種 CoT 方法：

| 方法 | Answer | 延遲 |
|------|--------|------|
| 無 CoT | 91.7% (22/24) | 3667 ms |
| 方法 B：顯式 scratchpad（max_tokens 800） | 95.8% (23/24) | 6217 ms |
| **方法 A：Implicit CoT（max_tokens 512）** | **100% (24/24)** | **4067 ms** |

方法 A 勝出：scratchpad 佔用 ~300 tokens 壓縮答案空間，導致 q06 仍截斷。Implicit CoT 讓模型在內部推理，512 tokens 全用於輸出。

---

## 三、系統架構

**資料流：**

```
問題
 │
 ▼
[Guard 1] 注入偵測（regex）
 │
 ▼
[Guard 2] 主題過濾（LLM-as-judge）
 │
 ▼
 ├─ 實體抽取 → Neo4j Cypher 圖譜遍歷
 └─ 語意向量 → Chroma 相似度搜尋
         │
         ▼ bi-encoder reranking + 動態 cap
[Guard 3] 證據充足性（triple 數量）
         │
         ▼
       LLM 生成答案（implicit CoT）
         │
         ▼
[Guard 4] 事實接地性（LLM-as-judge）
         │
         ▼
       JSON 回應
```

**服務組成（Docker Compose）：**

| 服務 | 技術 | 功能 |
|------|------|------|
| `api` | FastAPI + Python 3.12 | RAG pipeline + 4 道 guardrail |
| `neo4j` | Neo4j 5 | SOP 知識圖譜（29 節點、48 條邊） |
| `vllm` | vLLM + Qwen2.5-7B-Instruct-AWQ-int4 | 本地 LLM 推論（OpenAI 相容） |

---

## 四、快速開始

### 需求

| 需求 | 規格 |
|------|------|
| Docker Engine | 24.x |
| Docker Compose | v2.20+ |
| NVIDIA GPU | VRAM ≥ 8 GB |
| NVIDIA Container Toolkit | latest |

### 啟動

```bash
# 1. 設定環境變數
cp .env.example .env

# 2. 啟動服務（第一次約 5–15 分鐘，vLLM 載入模型）
docker compose up --build -d

# 3. 植入範例資料
docker compose run --rm api python scripts/ingest_all.py

# 4. 確認
curl http://localhost:8000/health
# → {"status":"ok"}
```

### 執行評測

```bash
docker compose exec -T api python scripts/eval_compare.py
```

---

## 五、API 使用

### POST /v1/ask

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "SOP_Etch_001 的步驟順序為何？"}'
```

**Request 欄位：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `question` | string | 必填 | 自然語言問題（1–1000 字） |
| `enable_guards` | bool | `true` | 是否啟用四道 guardrail |
| `debug` | bool | `false` | 回應中包含 LLM context 與 stage latency |
| `max_hop` | int（1–4） | `2` | 圖譜遍歷深度 |
| `top_k` | int（1–20） | `4` | 向量庫取回數量 |

**Response 範例：**

```json
{
  "status": "answered",
  "answer": "SOP_Etch_001 步驟順序：CheckVacuumPump → VerifyGasFlow → InspectChamberLeak → RestoreProcessCondition",
  "entities": ["SOP_Etch_001"],
  "evidence_triples": ["(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)", "..."],
  "model_triples": ["(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)", "..."],
  "source_docs": ["SOP_Etch_001"],
  "reasoning_type": "graph_rag",
  "confidence": 1.0,
  "guardrail_results": [...]
}
```

**status / reasoning_type：**

| 值 | 意義 |
|----|------|
| `answered` | 正常回答，所有 guardrail 通過 |
| `answered_with_warning` | 回答了，但事實接地性有疑慮（confidence 0.5） |
| `blocked` | 被 guardrail 擋下 |
| `blocked_injection` | Guard 1 擋住 |
| `blocked_off_topic` | Guard 2 擋住 |
| `blocked_low_evidence` | Guard 3 擋住 |

### 其他端點

| 端點 | 說明 |
|------|------|
| `GET /health` | Liveness check |
| `GET /v1/health` | Deep health（Neo4j + vLLM + Chroma） |
| `POST /v1/ingest` | 動態新增節點和邊（含 source_file 版本追蹤） |
| `GET /docs` | Swagger UI |

---

## 六、Guardrail 四道關卡

| # | 名稱 | 機制 | 擋什麼 |
|---|------|------|--------|
| 1 | injection_detection | Regex（20 個 pattern） | Prompt injection / jailbreak |
| 2 | topic_filter | LLM-as-judge | 非 SOP 相關問題 |
| 3 | evidence_sufficiency | Triple 數量 | 圖譜找不到任何 triple |
| 4 | fact_grounding | LLM-as-judge | 答案含圖譜外的推論或猜測 |

---

## 七、資料說明

```
data/
├── sop_docs/                     # 原始 SOP Markdown（向量庫用）
│   ├── etch_pressure_anomaly.md  # SOP_Etch_001
│   ├── vacuum_pump_check.md      # SOP_Pump_002
│   └── chamber_vent_procedure.md # SOP_Vent_003
├── graph_seed/
│   ├── nodes.json                # 29 個節點
│   └── edges.json                # 48 條邊
└── sample_queries/
    └── fab_queries.json          # 10 道測試題（含預期關鍵字）
```

**知識圖譜 Schema：**

```
Anomaly ──[TRIGGERS_SOP]──────────▶ SOPDocument
SOPDocument ──[FIRST_STEP]─────────▶ SOPStep
SOPStep ──[NEXT_STEP]──────────────▶ SOPStep
SOPStep ──[DEPENDS_ON]─────────────▶ SOPStep
SOPStep ──[REQUIRES_STATUS]────────▶ Equipment
SOPDocument ──[PRECONDITION]───────▶ Equipment
Equipment ──[INTERLOCK_WITH]───────▶ Equipment
SOPDocument ──[CROSS_DOC_DEPENDENCY]▶ SOPDocument
```

### 新增 SOP 文件

```bash
# LLM 自動抽取（需要 vLLM）
cp my_sop.md data/sop_docs/
docker compose run --rm api python scripts/extract_graph_from_sop.py

# Review → merge → ingest
docker compose run --rm api python scripts/extract_graph_from_sop.py --merge
docker compose run --rm api python scripts/ingest_all.py
```

---

## 八、Neo4j Browser

開啟 `http://localhost:7474`，連線設定：

| 欄位 | 值 |
|------|---|
| Connect URL | `bolt://localhost:7687` |
| Username | `neo4j` |
| Password | `password123` |

**常用 Cypher：**

```cypher
-- 看整張圖
MATCH p = (n)-[r]->(m) RETURN p

-- SOP_Etch_001 步驟順序
MATCH p = (d:SOPDocument {id:"SOP_Etch_001"})-[:FIRST_STEP|NEXT_STEP*]->(s)
RETURN p

-- 異常觸發哪份 SOP
MATCH p = (a:Anomaly)-[:TRIGGERS_SOP]->(d:SOPDocument)
RETURN p

-- 確認資料筆數
MATCH (n) RETURN count(n)        // 29
MATCH ()-[r]->() RETURN count(r) // 48
```

---

## 九、常見問題

**vLLM 一直沒啟動？**
```bash
docker compose logs vllm | tail -30
```
- `CUDA out of memory` → GPU VRAM 不足
- `model not found` → 確認模型路徑

**`/v1/ask` 回傳 500？**
```bash
docker compose logs api | tail -30
```
- Neo4j 還沒 ready → 等 `docker compose ps` 顯示 `healthy`
- vLLM 還沒 ready → 等模型載入完成（約 3–10 分鐘）

**沒有 GPU 怎麼辦？**

在 `docker-compose.yml` 把 `vllm` 服務 comment 掉。Guard 2/4 會 fallback，答案生成回傳錯誤訊息。加 `"enable_guards": false` 可跳過 LLM 依賴的 guardrail，只跑圖譜 + 向量檢索。

**停止與重置：**
```bash
docker compose down          # 停止（資料保留）
docker compose down -v       # 完全清除（含 volumes）
docker compose up --build -d # 重新啟動
```
