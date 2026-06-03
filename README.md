# Fab SOP Knowledge Query API

> **Guardrailed Hybrid Graph RAG — 單機企業級 MVP**
> 用自然語言查詢晶圓廠 SOP 文件：系統從知識圖譜（Neo4j）與向量庫（Qdrant）撈出 SOP 步驟、設備依賴與前置條件，由本地 LLM（vLLM）合成答案，全程經過四道 guardrail 過濾。

![visualisation](https://github.com/jirong-spec/fab-sop-rag/blob/main/visualisation.png)

---

## 亮點

- **結構化圖譜 RAG**：步驟順序、`DEPENDS_ON` 鏈、跨文件依賴、設備聯鎖等「多跳」問題，Graph RAG **100%**，傳統 Vector RAG 僅 **37.5%**。
- **嚴謹評估，不是自我感覺良好**：held-out dev/test 切分、retrieval **recall@k**、**LLM-as-judge**、拒答/離題/注入**負例**、3 次重跑報變異——不是 keyword 子字串自我安慰。
- **靠合成大圖壓測抓到並修掉 2 個 scaling bug**：context 溢出、低相似度邊被 rerank 砍掉（詳見 [評估](#評估)）。
- **四道 guardrail**：注入偵測 → 主題過濾 → 證據充足性 → 事實接地性，降低離題與幻覺。
- **單機 Docker Compose**：FastAPI + Neo4j + Qdrant + vLLM，一個 `docker compose up` 起整套；含 SSE 串流與 Streamlit demo。

---

## 目錄

- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [API 使用](#api-使用)
- [Guardrail 四道關卡](#guardrail-四道關卡)
- [評估](#評估)
- [工程決策與優化歷程](#工程決策與優化歷程)
- [資料與知識圖譜](#資料與知識圖譜)
- [運維與常見問題](#運維與常見問題)

---

## 系統架構

**資料流：**

```
問題
 │
 ▼ [Guard 1] 注入偵測（regex，中英文）
 ▼ [Guard 2] 主題過濾（LLM-as-judge）
 │
 ▼ 實體抽取  ├─ 問題本身的 CamelCase / SOP_ID token
 │           └─ Qdrant 相似度搜尋（輔助擴充實體候選詞）
 ▼ Neo4j 圖譜遍歷（hop 1–4，distinct-edge）→ evidence triples（全量）
 ▼ bi-encoder rerank + 動態 cap + token 預算裁切 → model triples（送入 LLM）
 │
 ▼ [Guard 3] 證據充足性（triple 數量）
 ▼ LLM 生成答案（implicit CoT）
 ▼ [Guard 4] 事實接地性（LLM-as-judge）
 │
 ▼ JSON 回應（含 source_docs 引用追蹤）
```

**服務組成（Docker Compose）：**

| 服務 | 技術 | 功能 |
|------|------|------|
| `api` | FastAPI + Python 3.12 | RAG pipeline + 4 道 guardrail |
| `neo4j` | Neo4j 5 | SOP 知識圖譜（29 節點 / 48 邊） |
| `qdrant` | Qdrant v1.18 | SOP 文件向量庫（語意檢索 / 實體擴充） |
| `vllm` | vLLM + Qwen2.5-7B-Instruct-AWQ-int4 | 本地 LLM 推論（OpenAI 相容） |

---

## 快速開始

### 需求

| 需求 | 規格 |
|------|------|
| Docker Engine | 24.x |
| Docker Compose | v2.20+ |
| NVIDIA GPU | VRAM ≥ 8 GB |
| NVIDIA Container Toolkit | latest |

> 沒有 GPU？見 [常見問題](#沒有-gpu-怎麼辦)。

### 啟動

```bash
# 1. 設定環境變數（將 LLM_MODEL_DIR 指向本機模型目錄）
cp .env.example .env

# 2. 啟動服務（第一次約 5–15 分鐘，vLLM 載入模型）
docker compose up --build -d

# 3. 植入範例資料（圖譜 + 向量庫）
docker compose run --rm api python scripts/ingest_all.py

# 4. 確認
curl http://localhost:8000/health      # → {"status":"ok"}
```

### Demo UI（Streamlit）

```bash
pip install streamlit requests
streamlit run demo_app.py               # → http://localhost:8501
```

含側邊欄範例題、token-by-token 串流、可展開檢視送入 LLM 的 `model_triples` 與 4 道 guardrail 結果。
（前提：`docker compose up -d` 已執行、API 在 `localhost:8000`。）

### 執行評測

```bash
# 嚴謹評估（held-out + LLM-judge + 負例，39 題）
docker compose exec -T api python scripts/eval_rigorous.py --runs 3

# Graph vs Vector 基線比較（keyword，10 題）
docker compose exec -T api python scripts/eval_compare.py
```

---

## API 使用

### `POST /v1/ask`

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

**Response 範例（含 citation traceability）：** `source_docs` 自動從 `evidence_triples` 提取引用的 SOP 文件 ID，讓答案可追溯。

```json
{
  "status": "answered",
  "answer": "SOP_Etch_001 步驟順序：CheckVacuumPump → VerifyGasFlow → InspectChamberLeak → RestoreProcessCondition",
  "entities": ["SOP_Etch_001"],
  "evidence_triples": ["(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)", "..."],
  "model_triples": ["(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)", "..."],
  "source_docs": ["SOP_Etch_001", "SOP_Pump_002"],
  "reasoning_type": "graph_rag",
  "confidence": 1.0,
  "guardrail_results": [...]
}
```

**`status` / `reasoning_type`：**

| 值 | 意義 |
|----|------|
| `answered` | 正常回答，所有 guardrail 通過 |
| `answered_with_warning` | 回答了，但事實接地性有疑慮（confidence 0.5） |
| `blocked_injection` / `blocked_off_topic` / `blocked_low_evidence` | 分別被 Guard 1 / 2 / 3 擋下 |

### `POST /v1/ask/stream`

相同的 guardrail pipeline，但 LLM 輸出以 **Server-Sent Events** 串流回傳（第一個 token < 200ms）。事件 `type`：`token`（每個 token）、`done`（最終結果與 metadata）、`blocked`（被擋下）。

### 其他端點

| 端點 | 說明 |
|------|------|
| `GET /health` | Liveness check |
| `GET /v1/health` | Deep health（Neo4j + vLLM + Qdrant） |
| `POST /v1/ingest` | 動態新增節點/邊（含 `source_file` 版本追蹤） |
| `GET /docs` | Swagger UI |

---

## Guardrail 四道關卡

| # | 名稱 | 機制 | 擋什麼 |
|---|------|------|--------|
| 1 | `injection_detection` | Regex（中英文多模式） | Prompt injection / jailbreak |
| 2 | `topic_filter` | LLM-as-judge（+ SOP 實體碼快速路徑） | 非 SOP 相關問題 |
| 3 | `evidence_sufficiency` | Triple 數量 | 圖譜找不到任何 triple |
| 4 | `fact_grounding` | LLM-as-judge | 答案含圖譜外的推論或猜測 |

LLM-judge 的 fallback 政策可由環境變數調整（`TOPIC_FALLBACK_POLICY` / `GROUNDING_FALLBACK_POLICY`）；grounding 預設 `strict`（保守），topic 預設 `lenient`。

---

## 評估

評估分三層，由嚴到鬆——**先看嚴謹評估**，它才是可信的泛化指標。

### 1. 嚴謹評估（held-out + LLM-judge + 負例）

`scripts/eval_rigorous.py`（39 題於 `data/sample_queries/fab_queries_v2.json`），補上四項方法論修掉舊評估的硬傷：

- **held-out 切分**：test 的 24/27 條 gold 邊「從未被調參看過」，避免「拿考古題考自己」的過擬合假象。
- **retrieval recall@k**：以每題的 `gold_triples`（`[from, rel, to]`）量測檢索是否撈到正確的邊。
- **LLM-as-judge**：答案對「圖譜推導的標準答案」評 correct/partial/wrong，比 keyword 子字串嚴格。
- **負例 + 變異**：拒答（查無實體/步驟）、離題、注入各自計分；每題跑 3 次報 mean±std。

| 指標 | DEV（調過參） | **TEST（held-out）** |
|------|--------------|---------------------|
| Answer keyword-match | 100% | **100%** |
| Retrieval recall@k（model triples） | 100% | **96.5%** |
| Answer correctness（LLM-judge） | 100% | **100%** |
| 負例：拒答 / 離題 / 注入 | — | **100% / 100% / 100%** |

3 次重跑全 **±0.0**（temp=0 可重現）。recall@k 96.5%（非 100%）來自多跳依賴鏈：低相似度的 `DEPENDS_ON` 邊偶爾被 cap 砍掉，但 evidence recall 仍 100%、judge 仍 100%（模型由其餘 context 重建出鏈）。

> **限制**：3 SOP / 48 邊的圖太小，2-hop 遍歷幾乎撈回整張圖，retrieval 沒被真正壓測——這正是下方 scale 壓測的動機。LLM-judge 與生成用同一個 vLLM（self-grading bias），keyword 與 recall 為 model-independent 的交叉驗證。

### 2. Graph RAG vs Vector RAG（keyword 基線，10 題）

把同一套 pipeline 換成純向量檢索做對照（keyword 子字串比對，與調參同批題，**會高估**，僅作基線參考）：

| 指標 | Graph RAG | Vector RAG |
|------|-----------|------------|
| Answer 命中率 | **100%** (24/24) | 62.5% (15/24) |
| **多跳查詢** | **100%** (8/8) | 37.5% (3/8) |
| 正確攔截非 SOP 問題 | 2/2 | 2/2 |
| 平均端對端延遲 | ~3.9 s | ~3.7 s |

差距集中在需要圖結構推理的問題（步驟順序、`DEPENDS_ON` 鏈、跨文件依賴）——Vector RAG 在這些題上完全失敗。

### 3. Scale 壓力測試（10-SOP 合成圖）

小圖上連 held-out 都接近滿分，代表 retrieval 沒被壓到。`scripts/gen_synthetic_sops.py` 確定性生成 7 份合成 SOP（→ **10 SOP / 93 節點 / 168 邊**），刻意讓 `RFPowerSupply`、`N2PurgeSystem`、`TurboVacuumPump` 被多份 SOP 共用成 **hub**，製造「一堆長得很像、狀態卻不同的競爭邊」。用 `fab_queries_scale.json`（15 題 hub 題）評測，**暴露兩個小圖藏住的 bug**並修掉：

| 指標（10-SOP held-out test） | 修前 | **修後** |
|------|------|----------|
| Answer keyword-match | 84.3% | **96.2%** |
| Retrieval recall@k（model） | 88.5% | **96.2%** |
| Answer correctness（LLM-judge） | 84.6% | **98.1%** |
| `step_requires_status` recall | 50% | **100%** |

兩個 bug 與修法見[工程決策](#工程決策與優化歷程)的「Scaling」一節。重現方式（不污染 production seed）：

```bash
python scripts/gen_synthetic_sops.py
docker compose run --rm api python scripts/ingest_all.py
docker compose run --rm api python scripts/eval_rigorous.py --queries data/sample_queries/fab_queries_scale.json
git checkout data/graph_seed data/sop_docs   # 還原 3-SOP demo
```

---

## 工程決策與優化歷程

### Retrieval：Bi-encoder Reranking + Edge Enrichment + 動態 Cap

Graph traversal 對 q06 回傳 ~48 條 triples，但關鍵的 `INTERLOCK_WITH` triple 排在第 47 位。三項優化讓 R 從 71% → 100%（dev set，keyword）：

| 方法 | cap | R | A | avg 延遲 |
|------|-----|---|---|---------|
| 無 reranking | 25 | ~71% | 71% | 2683 ms |
| Bi-encoder cosine | 50 | 100% | 83% | 3012 ms |
| + Edge enrichment | 50 | 100% | 83% | 3067 ms |
| + 動態 cap | 動態 | 100% | 88% | 2944 ms |
| + Implicit CoT（7B） | 動態 | **100%** | **100%** | **4067 ms** |

- **Edge Enrichment**：`DEPENDS_ON`、`NEXT_STEP` 等邊序列化後只有 CamelCase ID，中文問題的 embedding 相似度趨近零。為邊加 `description` 屬性後，DEPENDS_ON triple 從第 47 位升至第 13 位。
- **動態 Cap**（`app/services/answer_service.py`）：`threshold = max(top_score × 0.50, 20)`，保留高分 triple 再截斷。

### Generation：Implicit Chain-of-Thought

R=100% 後，q05/q06 仍 A⚠——LLM 沒有完整提取 triple 屬性值（`reason`、`interlock_id`、`trigger`、`action`）。比較兩種 CoT：顯式 scratchpad（max_tokens 800）佔用 ~300 tokens 壓縮答案空間導致 q06 仍截斷（95.8%）；**Implicit CoT**（max_tokens 512，要求模型在內部推理）512 tokens 全用於輸出，達 **100%（24/24）**。

### Graph Traversal：distinct-edge（取代 path + LIMIT 200）

原本 `MATCH p=(n)-[*1..2]-(m) RETURN p LIMIT 200`，但 `LIMIT 200` 限制的是**路徑數**——2-hop 路徑爆炸時可能在撈到關鍵邊前就截斷。改為收集 hop 內的 **distinct relationships**（`UNWIND r ... WITH DISTINCT rel`，LIMIT 500）。三模式實測：

| 模式 | R | A | 多跳 | 備註 |
|------|---|---|------|------|
| undirected（原狀） | 100% | 100% | 8/8 | 命中 LIMIT 200 截斷 |
| directed（純 outgoing） | 92% | 88% | 6/8 | recall 退步，否決 |
| **distinct** | **100%** | **100%** | **8/8** | 無截斷 ✅ |

有向遍歷會退步：實體抽取無法保證 seed 落在邊的來源端，純 outgoing 會漏掉入邊（`REQUIRES_STATUS`、`TRIGGERS_SOP`）。可用 `GRAPH_TRAVERSAL_MODE` 切換（預設 `distinct`）。

### 向量庫：Chroma → Qdrant

由 in-process Chroma（`persist_directory`）改為獨立 Qdrant server。讀寫統一走 LangChain `QdrantVectorStore`（payload schema 一致）；ingest 用 `force_recreate=True`，刪除來源 `.md` 後不留孤兒向量。embedding 模型不變（`paraphrase-multilingual-MiniLM-L12-v2`，384 維 Cosine），檢索品質不變：Graph RAG 維持 100%，Vector baseline 54%→62.5%（HNSW 排序與 vLLM 批次的細微差異）。

### Scaling：token 預算裁切 + 邊 gloss enrichment

10-SOP 壓測（見[評估](#3-scale-壓力測試10-sop-合成圖)）暴露兩個小圖藏住的 bug：

1. **context 溢出**：dense 子圖回傳 ~100 條 triple，prompt 撐爆 4096 context → HTTP 400、整題生成失敗。
   → **token-aware 裁切**（`answer_service`）：依 `llm_max_model_len − 完成 − margin` 預算保留高分 triple，prompt 絕不溢出。
2. **低相似度邊被砍**：`REQUIRES_STATUS / PRECONDITION / INTERLOCK` 邊只有 CamelCase ID，對中文問題相似度低 → 被 rerank cap 砍掉。
   → **邊 gloss enrichment**（`graph_store`）：為缺 `description` 的邊型合成中文 gloss（比照 NEXT_STEP/DEPENDS_ON），10-SOP `step_requires_status` recall 50%→100%。這兩個修正在 3-SOP 上也讓 LLM-judge 90.9%→100%，無回歸。

---

## 資料與知識圖譜

```
data/
├── sop_docs/                     # 原始 SOP Markdown（向量庫用）
│   ├── etch_pressure_anomaly.md  # SOP_Etch_001
│   ├── vacuum_pump_check.md      # SOP_Pump_002
│   └── chamber_vent_procedure.md # SOP_Vent_003
├── graph_seed/
│   ├── nodes.json                # 29 節點
│   └── edges.json                # 48 邊
└── sample_queries/
    ├── fab_queries.json          # 10 題（Graph vs Vector 基線，eval_compare）
    ├── fab_queries_v2.json       # 39 題（嚴謹評估，含 dev/test、gold_triples、負例）
    └── fab_queries_scale.json    # 15 題（10-SOP scale 壓測 fixture）
```

> 所有資料皆為**合成範例資料**，僅供教學/測試，不代表任何真實機台程序。

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
SOPStep ──[DEFINED_IN]─────────────▶ SOPDocument
```

**新增 SOP 文件**（LLM 自動從 Markdown 抽取節點/邊）：

```bash
cp my_sop.md data/sop_docs/
docker compose run --rm api python scripts/extract_graph_from_sop.py --merge  # review 後 merge 進 seed
docker compose run --rm api python scripts/ingest_all.py
```

---

## 運維與常見問題

### Neo4j Browser

開啟 `http://localhost:7474`，以 `bolt://localhost:7687` / `neo4j` / `password123` 連線。常用 Cypher：

```cypher
MATCH p = (n)-[r]->(m) RETURN p                                            // 看整張圖
MATCH p = (d:SOPDocument {id:"SOP_Etch_001"})-[:FIRST_STEP|NEXT_STEP*]->(s) RETURN p  // 步驟順序
MATCH (n) RETURN count(n)         // 29
MATCH ()-[r]->() RETURN count(r)  // 48
```

### vLLM 一直沒啟動？

```bash
docker compose logs vllm | tail -30
```
`CUDA out of memory` → GPU VRAM 不足；`model not found` / `HFValidationError` → 確認 `.env` 的 `LLM_MODEL_DIR` 指向正確的本機模型目錄。

### `/v1/ask` 回傳 500？

`docker compose logs api | tail -30`。多半是 Neo4j 或 vLLM 還沒 `healthy`（模型載入約 3–10 分鐘，`docker compose ps` 確認）。

### 沒有 GPU 怎麼辦？

在 `docker-compose.yml` 把 `vllm` 服務 comment 掉：Guard 2/4 會 fallback、答案生成回傳錯誤訊息。加 `"enable_guards": false` 可跳過 LLM 依賴的 guardrail，只跑圖譜 + 向量檢索。

### 停止與重置

```bash
docker compose down            # 停止（資料保留）
docker compose down -v         # 完全清除（含 volumes）
docker compose up --build -d   # 重新啟動
```
