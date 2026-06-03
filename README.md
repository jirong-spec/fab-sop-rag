# Fab SOP Knowledge Query API

**Guardrailed Hybrid Graph RAG — Single-machine Enterprise MVP**

用自然語言詢問晶圓廠 SOP 文件，系統自動從知識圖譜（Neo4j）和向量庫（Qdrant）撈出 SOP 步驟、設備依賴、前置條件，再由 LLM 合成答案，並經過四道 guardrail 過濾。

![image](https://github.com/jirong-spec/fab-sop-rag/blob/main/visualisation.png)

---

## 一、評估結果

### Graph RAG vs Vector RAG（keyword 比對 · 10 題 · Qwen2.5-7B-Instruct-AWQ-int4）

> 此表為 **Graph vs Vector 基線比較**，採 keyword 子字串比對、且調參與評測為同一批題（dev=test），
> 會高估泛化能力。更嚴謹的 held-out + LLM-judge 評估見下一節。

| 指標 | Graph RAG | Vector RAG | 差距 |
|------|-----------|------------|------|
| **Retrieval 命中率** | **100%** (24/24) | — | — |
| **Answer 命中率** | **100%** (24/24) | 54.2% (13/24) | **+45.8 pp** |
| **多跳查詢 Answer** | **100%** (8/8) | 25.0% (2/8) | **+75.0 pp** |
| **正確攔截非 SOP 問題** | 2/2 | 2/2 | — |
| **平均端對端延遲** | 3946 ms | 3695 ms | +251 ms |

```
ID   │ 類別                        │ R / A        │  Graph 延遲  │ Vector 延遲
─────┼─────────────────────────────┼──────────────┼─────────────┼────────────
q01  │ anomaly_handling            │ R✅ A✅ 2/2  │  4623 ms    │  3311 ms ✅
q02  │ sop_step_sequence  [↑HOP]   │ R✅ A✅ 4/4  │  4515 ms    │  4385 ms ❌
q03  │ equipment_precondition      │ R✅ A✅ 4/4  │  5432 ms    │  4399 ms ✅
q04  │ step_dependency             │ R✅ A✅ 3/3  │  4238 ms    │  4285 ms ✅
q05  │ cross_doc_dependency [↑HOP] │ R✅ A✅ 2/2  │  3293 ms    │  5150 ms ✅
q06  │ interlock_condition         │ R✅ A✅ 3/3  │  6668 ms    │  4144 ms ⚠
q07  │ vent_procedure      [↑HOP]  │ R✅ A✅ 2/2  │  4716 ms    │  4982 ms ❌
q08  │ off_topic_blocked           │ ✅ blocked   │   845 ms    │   845 ms ✅
q09  │ off_topic_blocked           │ ✅ blocked   │   877 ms    │   923 ms ✅
q10  │ pump_check_sequence         │ R✅ A✅ 4/4  │  4257 ms    │  4526 ms ⚠
─────┼─────────────────────────────┼──────────────┼─────────────┼────────────
     │ TOTALS                      │ R 100% A 100%│ avg 3946 ms │ avg 3695 ms
```

Vector RAG 欄位：✅ = A 全對、⚠ = 部分對、❌ = 全錯

- **R（Retrieval）**：預期關鍵字出現在模型實際收到的 triples（`model_triples`）中
- **A（Answer）**：預期關鍵字出現在模型 `answer` 欄位中
- `[↑HOP]` 需要多跳推理（step 鏈、跨文件依賴、DEPENDS_ON 鏈）

### 嚴謹評估：held-out test + LLM-judge + 負例（2026-06-02）

上面的 10 題表用 keyword 子字串比對、且 dev=test，會高估能力。為此另建一套嚴謹 harness
（`scripts/eval_rigorous.py`，27 題於 `data/sample_queries/fab_queries_v2.json`），補上四項方法論：

- **held-out 切分**：test 的 15/18 條 gold 邊「從未被調參看過」，test 分數才是泛化指標。
- **retrieval recall@k**：以每題的 `gold_triples`（`[from, rel, to]`）量測檢索是否撈到正確的邊。
- **LLM-as-judge**：每個答案對「圖譜推導出的標準答案」評 correct/partial/wrong，比 keyword 嚴格。
- **負例 + 變異**：拒答（查無實體/步驟）、離題、注入各自計分；每題跑 3 次報 mean±std。

| 指標 | DEV（調過參） | **TEST（held-out）** |
|------|--------------|---------------------|
| Answer keyword-match | 100% | **100%** |
| Retrieval recall@k（model triples） | 100% | **100%** |
| Answer correctness（LLM-judge） | 100% | **100%** |

負例（held-out）：拒答 **100%** · 離題攔截 **100%** · 注入攔截 **100%**。3 次重跑全 **±0.0**（temp=0 可重現）。

**但「太完美」本身就是警訊**：3 SOP / 48 邊的圖太小，2-hop 遍歷幾乎撈回整張圖，
retrieval 根本沒被壓力測試（evidence recall 必然 100%）。要看真實水準，需要更大、有干擾的圖 ⤵

### Scale 壓力測試：10-SOP 合成圖（找出並修掉 2 個 scaling bug）

`scripts/gen_synthetic_sops.py` 確定性生成 7 份合成 SOP（→ **10 SOP / 93 節點 / 168 邊**），刻意讓
`RFPowerSupply`、`N2PurgeSystem`、`TurboVacuumPump` 被多份 SOP 共用成 **hub**，製造「一堆長得很像、
狀態卻不同的競爭邊」——這才是 retrieval 真正要解的 disambiguation。用 `fab_queries_scale.json`（15 題
hub 題）評測，**暴露兩個小圖藏住的 bug**：

1. dense 子圖回傳 ~100 條 triple，prompt 撐爆 4096 context → HTTP 400、整題生成失敗。
2. `REQUIRES_STATUS / PRECONDITION / INTERLOCK` 邊只有 CamelCase ID，對中文問題相似度低 → 被 rerank cap 砍掉。

修法：**(#1)** prompt 依 token 預算裁切（`answer_service`，絕不溢出 context）；
**(#2)** 替缺 `description` 的邊型合成中文 gloss（`graph_store`，比照 NEXT_STEP/DEPENDS_ON 的 enrichment）。
修前 / 修後（10-SOP held-out test）：

| 指標（10-SOP TEST） | 修前 | **修後** |
|------|------|----------|
| Answer keyword-match | 84.3% | **96.2%** |
| Retrieval recall@k（model） | 88.5% | **96.2%** |
| Answer correctness（LLM-judge） | 84.6% | **98.1%** |
| `step_requires_status` recall | 50% | **100%** |

修後 `recall_model == recall_evidence`（96.2%）→ rerank/cap 不再丟掉已撈到的 gold；殘留 ~4% 來自
entity 抽取 / 2-hop 覆蓋（1 題 interlock 未觸及），屬較難的 NER 問題。這兩個修正在 3-SOP 上也讓
LLM-judge 90.9% → 100%（中文 gloss 讓答案更完整），**無回歸**。

**誠實限制：** LLM-judge 用與生成相同的 vLLM，有 self-grading bias；keyword 與 recall 為
model-independent 的交叉驗證。

```bash
# 基準（3-SOP）
docker compose exec -T api python scripts/eval_rigorous.py --runs 3

# Scale 壓測（10-SOP fixture；不污染 seed，跑完用 git checkout 還原）
python scripts/gen_synthetic_sops.py
docker compose run --rm api python scripts/ingest_all.py
docker compose run --rm api python scripts/eval_rigorous.py --queries data/sample_queries/fab_queries_scale.json
git checkout data/graph_seed data/sop_docs
```

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

### Graph Traversal：distinct-edge（取代 path + LIMIT 200）

原本圖遍歷用 `MATCH p=(n)-[*1..2]-(m) RETURN p LIMIT 200`，但 `LIMIT 200` 限制的是**路徑數**——2-hop 在連通圖會路徑爆炸，命中上限後可能漏掉關鍵邊。改為收集 hop 內的 **distinct relationships**（`UNWIND r ... WITH DISTINCT rel`，LIMIT 500），小圖根本碰不到上限。

三模式實測（10 題，graph-only）：

| 模式 | R | A | 多跳 (q02/q05/q07) | 備註 |
|------|---|---|------|------|
| undirected（原狀） | 100% | 100% | 8/8 | 命中 LIMIT 200 截斷 |
| directed（純 outgoing） | 92% | 88% | 6/8 | recall 退步，否決 |
| **distinct** | **100%** | **100%** | **8/8** | 無截斷 ✅ |

有向遍歷會退步，因實體抽取無法保證 seed 落在邊的來源端，純 outgoing 會漏掉入邊（如 `REQUIRES_STATUS`、`TRIGGERS_SOP`）。distinct 與 undirected 準確率打平、且消除截斷風險。可用 `GRAPH_TRAVERSAL_MODE` 切換（預設 `distinct`，見 `app/services/graph_store.py`）。

### Generation：Implicit Chain-of-Thought

R=100% 後，q05/q06 仍 A⚠——LLM 沒有完整提取 triple 屬性值（`reason`、`interlock_id`、`trigger`、`action`）。測試兩種 CoT 方法：

| 方法 | Answer | 延遲 |
|------|--------|------|
| 無 CoT | 91.7% (22/24) | 3667 ms |
| 方法 B：顯式 scratchpad（max_tokens 800） | 95.8% (23/24) | 6217 ms |
| **方法 A：Implicit CoT（max_tokens 512）** | **100% (24/24)** | **4067 ms** |

方法 A 勝出：scratchpad 佔用 ~300 tokens 壓縮答案空間，導致 q06 仍截斷。Implicit CoT 讓模型在內部推理，512 tokens 全用於輸出。

### 向量庫：Chroma → Qdrant

向量庫由 in-process 的 Chroma（`persist_directory`）改為獨立的 Qdrant server。讀寫統一走 LangChain `QdrantVectorStore`，payload schema 一致（`page_content` / `metadata`）；ingest 用 `force_recreate=True`，刪除來源 `.md` 後不會留下孤兒向量（plain upsert 會殘留）。

embedding 模型不變（`paraphrase-multilingual-MiniLM-L12-v2`，384 維、Cosine），因此檢索品質不變：Graph RAG 維持 R/A 100%，Vector RAG baseline 54%→62%（Qdrant HNSW 排序與 vLLM 批次的細微差異）。

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
實體抽取
 ├─ 問題本身的 CamelCase / SOP_ID token
 └─ Qdrant 相似度搜尋（輔助擴充實體候選詞）
         │
         ▼ 實體列表
Neo4j Cypher 圖譜遍歷（hop=1–4）
         │
         ▼ evidence triples（全量）
bi-encoder reranking + 動態 cap → model triples（送入 LLM）
         │
         ▼
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
| `qdrant` | Qdrant v1.18 | SOP 文件向量庫（語意檢索 / 實體擴充） |
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
# 編輯 .env，將 LLM_MODEL_DIR 設為本機模型目錄（預設 /opt/models）
# LLM_MODEL_DIR=/home/your-user/models

# 2. 啟動服務（第一次約 5–15 分鐘，vLLM 載入模型）
docker compose up --build -d

# 3. 植入範例資料
docker compose run --rm api python scripts/ingest_all.py

# 4. 確認
curl http://localhost:8000/health
# → {"status":"ok"}
```

### 啟動 Demo UI（Streamlit）

```bash
pip install streamlit requests
streamlit run demo_app.py
```

開啟 `http://localhost:8501`，功能包含：
- 側邊欄快速範例題一鍵帶入
- 串流輸出（token-by-token，含游標效果）
- 展開檢視送入 LLM 的圖譜關係（model_triples）
- 展開檢視 4 道 guardrail 結果
- API Key 輸入（若有設定 `API_KEY`）

> **前提**：`docker compose up -d` 已執行，API 在 `localhost:8000`。

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

### POST /v1/ask/stream

與 `/v1/ask` 相同的 guardrail pipeline，但 LLM 輸出以 **Server-Sent Events** 串流傳回，第一個 token 在 200ms 內到達。

```bash
curl -N -X POST http://localhost:8000/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "SOP_Etch_001 的步驟順序為何？"}'
```

**事件格式：**

| type | 說明 |
|------|------|
| `token` | `{"type":"token","text":"..."}` — 每個 LLM 輸出 token |
| `done` | 最終結果（含 entities、evidence_triples、guardrail_results） |
| `blocked` | 被 guardrail 擋下（含 reason） |

### 其他端點

| 端點 | 說明 |
|------|------|
| `GET /health` | Liveness check |
| `GET /v1/health` | Deep health（Neo4j + vLLM + Qdrant） |
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
├── sop_docs/                        # 原始 SOP Markdown（向量庫用）
│   ├── etch_pressure_anomaly.md     # SOP_Etch_001
│   ├── vacuum_pump_check.md         # SOP_Pump_002
│   └── chamber_vent_procedure.md    # SOP_Vent_003
├── graph_seed/
│   ├── nodes.json                   # 29 個節點
│   └── edges.json                   # 48 條邊
└── sample_queries/
    └── fab_queries.json             # 10 道測試題（含預期關鍵字）
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
// 看整張圖
MATCH p = (n)-[r]->(m) RETURN p

// SOP_Etch_001 步驟順序
MATCH p = (d:SOPDocument {id:"SOP_Etch_001"})-[:FIRST_STEP|NEXT_STEP*]->(s)
RETURN p

// 異常觸發哪份 SOP
MATCH p = (a:Anomaly)-[:TRIGGERS_SOP]->(d:SOPDocument)
RETURN p

// 確認資料筆數
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
- `model not found` / `HFValidationError` → 確認 `.env` 中 `LLM_MODEL_DIR` 指向正確的本機模型目錄，例如：
  ```
  LLM_MODEL_DIR=/home/your-user/models
  ```

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
