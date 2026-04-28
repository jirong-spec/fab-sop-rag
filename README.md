# Fab SOP Knowledge Query API

**Guardrailed Hybrid Graph RAG — Single-machine Enterprise MVP**

用自然語言詢問晶圓廠 SOP 文件，系統自動從知識圖譜（Neo4j）和向量庫（Chroma）
撈出 SOP 步驟、設備依賴關係、前置條件，再由 LLM 合成答案，並經過四道 guardrail 過濾。

---

## 一、這是什麼

| 服務 | 技術 | 功能 |
|------|------|------|
| `api` | FastAPI | RAG pipeline + 4 道 guardrail |
| `neo4j` | Neo4j 5 | SOP 知識圖譜（節點 + 關係） |
| `vllm` | vLLM + Qwen2.5-3B | 本地 LLM 推論（OpenAI 相容） |

三個服務用 Docker Compose 跑在同一台機器上，無需外部雲端服務。

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
         ▼ 合併 evidence triples
[Guard 3] 證據充足性（triple 數量）
         │
         ▼
       LLM 生成答案
         │
         ▼
[Guard 4] 事實接地性（LLM-as-judge）
         │
         ▼
       JSON 回應
```

---

## 二、事前準備

| 需求 | 最低版本 | 備註 |
|------|---------|------|
| Docker Engine | 24.x | |
| Docker Compose | v2.20 | Docker Desktop 內建 |
| NVIDIA GPU | VRAM ≥ 8 GB | 僅 vLLM 需要 |
| NVIDIA Container Toolkit | latest | 讓 Docker 存取 GPU |

**安裝 NVIDIA Container Toolkit（Ubuntu）：**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**確認 GPU 可被 Docker 看到：**

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

---

## 三、啟動步驟

### 步驟 1：設定環境變數

```bash
cd fab-sop-rag
cp .env.example .env
```

`.env` 預設值即可直接使用。需要改的情況：

| 情境 | 要改的變數 |
|------|-----------|
| 換不同的 LLM | `LLM_HF_MODEL`、`LLM_MODEL` |
| 需要 HuggingFace 登入的 model | `HUGGING_FACE_HUB_TOKEN` |
| 改 Neo4j 密碼 | `NEO4J_PASSWORD` |

### 步驟 2：啟動服務

```bash
docker compose up --build -d
```

第一次啟動會下載 LLM model（Qwen2.5-3B，約 6 GB），需要等 5–15 分鐘。

**觀察啟動狀態：**

```bash
docker compose ps                  # 看各服務狀態
docker compose logs -f vllm        # 觀察 model 載入進度
docker compose logs -f api         # 觀察 API 啟動
```

vLLM 啟動完成的標誌（logs 出現）：
```
INFO:     Application startup complete.
```

### 步驟 3：植入範例資料

服務啟動後，Neo4j 和 Chroma 都是空的，需要先寫入 SOP 資料：

```bash
docker compose run --rm api python scripts/ingest_all.py
```

這會執行兩個步驟：
1. **Graph seed**：把 `data/graph_seed/nodes.json` 和 `edges.json` 寫入 Neo4j（29 節點、48 條邊）
2. **Vector seed**：把 `data/sop_docs/` 下的 3 份 SOP Markdown 切塊嵌入，寫入 Chroma

輸出範例：
```
============================================================
  Graph seed  (Neo4j)
============================================================
2026-04-28 [INFO] Merging 29 nodes ...
2026-04-28 [INFO] Merging 48 edges ...
2026-04-28 [INFO] Graph seed complete.

[OK] Graph seed  (Neo4j) completed successfully.

============================================================
  Vector seed (Chroma)
============================================================
2026-04-28 [INFO] Found 3 SOP documents ...
2026-04-28 [INFO] Vector ingest complete: 3 files, 42 chunks
[OK] Vector seed (Chroma) completed successfully.
```

### 步驟 4：確認服務正常

```bash
# Liveness check
curl http://localhost:8000/health
# → {"status":"ok"}

# Deep health（檢查 Neo4j + Chroma + vLLM 連線）
curl http://localhost:8000/v1/health
```

`/v1/health` 回應範例：
```json
{
  "status": "ok",
  "version": "1.0.0",
  "services": {
    "neo4j":  {"status": "ok", "latency_ms": 12},
    "chroma": {"status": "ok", "latency_ms": 3},
    "vllm":   {"status": "ok", "latency_ms": 45}
  }
}
```

`status: "degraded"` 表示某個服務連不上，但 API 本身還活著。

---

## 四、呼叫 API

### 端點

```
POST /v1/ask
Content-Type: application/json
```

### Request 欄位

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `question` | string | 必填 | 自然語言問題 |
| `enable_guards` | bool | `true` | 是否啟用四道 guardrail |
| `debug` | bool | `false` | 回應中包含 LLM context |
| `max_hop` | int（1–4） | `2` | 圖譜遍歷深度 |
| `top_k` | int（1–20） | `4` | 向量庫取回數量 |

### 範例查詢

**1. 查詢 SOP 步驟順序**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "SOP_Etch_001 的第一個步驟是什麼？之後的步驟順序為何？"}'
```

**2. 查詢設備前置條件**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "執行 SOP_Etch_001 之前，TurboVacuumPump 需要是什麼狀態？"}'
```

**3. 查詢異常觸發流程**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "蝕刻站發生 PressureAnomaly 時，應執行哪份 SOP？"}'
```

**4. 查詢跨文件設備依賴**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "SOP_Etch_001 裡 TurboVacuumPump 的狀態定義是在哪份文件中說明的？"}'
```

**5. 啟用 debug 模式（看 LLM 實際收到的 context）**

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "CheckVacuumPump 步驟需要什麼設備狀態？", "debug": true}'
```

---

## 五、讀懂回應

```json
{
  "question": "SOP_Etch_001 的第一個步驟是什麼？",
  "status": "answered",
  "answer": "依據 SOP 知識圖譜，SOP_Etch_001 的第一步驟為 CheckVacuumPump（確認真空泵浦運作狀態）。後續步驟依序為 VerifyGasFlow → InspectChamberLeak → RestoreProcessCondition。",
  "entities": ["SOP_Etch_001", "CheckVacuumPump"],
  "evidence_triples": [
    "(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)",
    "(CheckVacuumPump)-[:NEXT_STEP]->(VerifyGasFlow)",
    "(VerifyGasFlow)-[:NEXT_STEP]->(InspectChamberLeak)",
    "(InspectChamberLeak)-[:NEXT_STEP]->(RestoreProcessCondition)"
  ],
  "guardrail_results": [
    {"stage": "input",     "name": "injection_detection", "pass": true,  "reason": "未偵測到注入模式"},
    {"stage": "input",     "name": "topic_filter",        "pass": true,  "reason": "屬於 SOP 操作步驟範疇"},
    {"stage": "retrieval", "name": "evidence_sufficiency","pass": true,  "reason": "檢索到 4 筆 SOP 三元組，證據充足"},
    {"stage": "output",    "name": "fact_grounding",      "pass": true,  "reason": "答案完全基於圖譜"}
  ],
  "reasoning_type": "graph_rag",
  "confidence": 1.0
}
```

### `status` 的意義

| status | 意義 |
|--------|------|
| `answered` | 正常回答，所有 guardrail 通過 |
| `answered_with_warning` | 回答了，但事實接地性有疑慮（confidence 0.5） |
| `blocked` | 被 guardrail 擋下，`answer` 說明原因 |

### `reasoning_type` 的意義

| reasoning_type | 哪道 guardrail 擋的 |
|----------------|-------------------|
| `graph_rag` | 正常通過 |
| `answered_with_warning` | Guard 4（事實接地）警告 |
| `blocked_injection` | Guard 1（注入偵測）擋住 |
| `blocked_off_topic` | Guard 2（主題過濾）擋住 |
| `blocked_low_evidence` | Guard 3（證據不足）擋住 |

### `confidence` 的意義

| 值 | 意義 |
|----|------|
| `1.0` | 全部 guardrail 通過 |
| `0.5` | 事實接地性警告 |
| `0.0` | 被擋住，未生成答案 |

---

## 六、Guardrail 四道關卡

| # | 階段 | 名稱 | 機制 | 擋什麼 |
|---|------|------|------|--------|
| 1 | 輸入 | injection_detection | Regex（20 個 pattern） | Prompt injection / jailbreak |
| 2 | 輸入 | topic_filter | LLM-as-judge | 非 SOP 相關問題 |
| 3 | 檢索 | evidence_sufficiency | Triple 數量 | 圖譜找不到任何 triple |
| 4 | 輸出 | fact_grounding | LLM-as-judge | 答案含圖譜外的推論或猜測 |

**Guard 2 的 SOP 知識庫範疇（主題過濾依據）：**

1. 製程異常處置（壓力異常、溫度超標、真空度不足…）
2. SOP 操作步驟與流程
3. 設備與機台狀態條件
4. 步驟依賴關係（PRECONDITION、NEXT_STEP…）
5. 製程條件與 recipe 參數
6. 良率異常與排查流程
7. 跨文件設備依賴關係

範疇外的問題（股價、程式設計、數學等）會被 Guard 2 攔截並回傳 `status: blocked`。

---

## 七、資料說明

```
data/
├── sop_docs/                     # 原始 SOP Markdown（植入向量庫用）
│   ├── etch_pressure_anomaly.md  # SOP_Etch_001 蝕刻站壓力異常處置
│   ├── vacuum_pump_check.md      # SOP_Pump_002 真空泵浦狀態檢查
│   └── chamber_vent_procedure.md # SOP_Vent_003 腔體洩壓程序
├── graph_seed/
│   ├── nodes.json                # 29 個節點（SOPDocument/SOPStep/Equipment/Anomaly/ProcessCondition）
│   └── edges.json                # 48 條邊（FIRST_STEP/NEXT_STEP/REQUIRES_STATUS/DEPENDS_ON/…）
└── sample_queries/
    └── fab_queries.json          # 10 個測試問題（含預期行為標記）
```

> 全部為**教學範例資料**，不代表任何真實製造商的操作規範。

**新增自己的 SOP 文件：**

1. 將 Markdown 檔放入 `data/sop_docs/`
2. 在 `data/graph_seed/nodes.json` 新增對應節點
3. 在 `data/graph_seed/edges.json` 新增對應關係
4. 重新執行 `docker compose run --rm api python scripts/ingest_all.py`（idempotent，不會重複寫入）

---

## 八、服務端點總覽

| 端點 | 說明 |
|------|------|
| `http://localhost:8000/health` | Liveness check |
| `http://localhost:8000/v1/health` | Deep health（含各服務狀態） |
| `http://localhost:8000/v1/ask` | 主要查詢 API |
| `http://localhost:8000/docs` | Swagger UI（互動測試介面） |
| `http://localhost:7474` | Neo4j Browser（圖譜視覺化） |

### Neo4j Browser 連線方式

**步驟 1** — 開啟 `http://localhost:7474`

**步驟 2** — 連線對話框填入：

| 欄位 | 值 |
|------|---|
| Connect URL | `bolt://localhost:7687` |
| Username | `neo4j` |
| Password | `password123` |

> **注意：Connect URL 必須用 `bolt://`，不能用 `neo4j://`。**
> `neo4j://` 是 cluster routing 協定，單機實例沒有 router 會連線失敗。

**如果瀏覽器開在遠端機器（非 Linux 本機）：**

先在本機建 SSH port forward：
```bash
ssh -L 7474:localhost:7474 -L 7687:localhost:7687 your-user@linux-server
```
然後在本機瀏覽器開 `http://localhost:7474`，Connect URL 同樣填 `bolt://localhost:7687`。

**不使用 Browser，直接在 container 內查詢：**
```bash
docker exec -it fab-sop-rag-neo4j-1 \
  cypher-shell -u neo4j -p password123 \
  "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
```

---

### Cypher 查詢範例

**看整張圖（29 節點、48 條邊）：**
```cypher
MATCH p = (n)-[r]->(m) RETURN p
```

**SOP_Etch_001 完整步驟順序：**
```cypher
MATCH p = (d:SOPDocument {id:"SOP_Etch_001"})-[:FIRST_STEP|NEXT_STEP*]->(s:SOPStep)
RETURN p
```

**所有步驟的設備狀態要求：**
```cypher
MATCH p = (s:SOPStep)-[:REQUIRES_STATUS|PRECONDITION]->(e:Equipment)
RETURN p
```

**異常觸發哪份 SOP：**
```cypher
MATCH p = (a:Anomaly)-[:TRIGGERS_SOP]->(d:SOPDocument)
RETURN p
```

**跨文件依賴關係：**
```cypher
MATCH p = (a:SOPDocument)-[:CROSS_DOC_DEPENDENCY]->(b:SOPDocument)
RETURN p
```

**確認資料筆數：**
```cypher
MATCH (n) RETURN count(n)       // 應為 29
MATCH ()-[r]->() RETURN count(r) // 應為 48
```

---

## 九、停止與重置

```bash
# 停止服務（資料保留）
docker compose down

# 重新啟動（不重建 image）
docker compose up -d

# 完全清除所有資料（volumes 一起刪）
docker compose down -v

# 清除後重新來過
docker compose up --build -d
docker compose run --rm api python scripts/ingest_all.py
```

---

## 十、常見問題

**Q：vLLM 一直沒啟動？**

```bash
docker compose logs vllm | tail -30
```

- 出現 `CUDA out of memory` → GPU VRAM 不足，換更小的 model
- 出現 `model not found` → 確認 `LLM_HF_MODEL` 是正確的 HuggingFace model ID
- 一直卡在下載 → 網路問題，設定 `HUGGING_FACE_HUB_TOKEN` 或改用已下載的本地 model

**Q：`/v1/ask` 回傳 500？**

```bash
docker compose logs api | tail -30
```

- Neo4j 還沒 ready → 等 `docker compose ps` 顯示 neo4j `healthy`
- vLLM 還沒 ready → Guard 2 / 4 會 fallback，答案生成會失敗；等 vLLM ready 後重試

**Q：答案都說「無足夠資訊」？**

確認資料有成功寫入：

```bash
# 確認 Neo4j 有資料
curl http://localhost:8000/v1/health

# 或在 Neo4j Browser 跑
MATCH (n) RETURN count(n)   # 應為 29
```

如果是 0，代表 `ingest_all.py` 沒跑成功，重跑：

```bash
docker compose run --rm api python scripts/ingest_all.py
```

**Q：想在沒有 GPU 的環境跑？**

在 `docker-compose.yml` 把 `vllm` 服務整個 comment 掉，同時移除 `api.depends_on` 裡的 `vllm` 項目。
API 仍可啟動，但：
- Guard 2、Guard 4 會 fallback（`topic_fallback_policy=lenient` 放行、`grounding_fallback_policy=strict` 標為未接地）
- 答案生成會回傳 `（LLM 服務暫時無法使用）`

可加 `"enable_guards": false` 跳過所有 LLM 依賴的 guardrail，只跑圖譜 + 向量檢索。

---

## 十一、架構說明

```
fab-sop-rag/
├── app/
│   ├── api/routes.py          # /health, /v1/health, /v1/ask
│   ├── services/
│   │   ├── pipeline.py        # 四道 guardrail 的主控流程
│   │   ├── retrieval_service.py  # 圖譜 + 向量混合檢索
│   │   ├── guardrails.py      # guard_injection / guard_topic / guard_evidence / guard_grounding
│   │   ├── judge_service.py   # LLM-as-judge（主題過濾 + 事實接地）
│   │   └── answer_service.py  # LLM 答案生成
│   ├── middleware/request_id.py  # X-Request-ID 關聯 ID
│   ├── config.py              # pydantic-settings（.env 驅動）
│   └── schemas.py             # Pydantic 資料模型
├── scripts/
│   ├── ingest_graph.py        # 寫入 Neo4j
│   ├── ingest_vector.py       # 寫入 Chroma
│   └── ingest_all.py          # 一次跑完
├── data/
│   ├── sop_docs/              # 原始 SOP Markdown
│   ├── graph_seed/            # 節點 + 邊 JSON
│   └── sample_queries/        # 測試問題集
├── docker-compose.yml
├── Dockerfile
└── .env.example
```
