# Fab SOP Knowledge Query API

**Guardrailed Hybrid Graph RAG — Single-machine Enterprise MVP**

用自然語言詢問晶圓廠 SOP 文件，系統自動從知識圖譜（Neo4j）和向量庫（Chroma）
撈出 SOP 步驟、設備依賴關係、前置條件，再由 LLM 合成答案，並經過四道 guardrail 過濾。

![image](https://github.com/jirong-spec/fab-sop-rag/blob/main/visualisation.png)
---

## 一、這是什麼

| 服務 | 技術 | 功能 |
|------|------|------|
| `api` | FastAPI | RAG pipeline + 4 道 guardrail |
| `neo4j` | Neo4j 5 | SOP 知識圖譜（節點 + 關係） |
| `vllm` | vLLM + Qwen2.5-7B-Instruct-AWQ-int4 | 本地 LLM 推論（OpenAI 相容） |

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

## 二、Baseline RAG vs Graph RAG 評估結果

### 評估方法

使用 `data/sample_queries/fab_queries.json` 的 10 道測試題，同時跑兩條 pipeline：

| Pipeline | 檢索機制 | `reasoning_type` |
|----------|---------|-----------------|
| **Graph RAG**（本系統） | 向量 → 實體抽取 → Neo4j 圖譜遍歷 → triples | `graph_rag` |
| **Baseline RAG**（對照組） | 向量 → Chroma 文本片段（無圖譜遍歷） | `baseline_rag` |

兩條 pipeline 使用**相同的四道 guardrail**，評測標準：

- **R（Retrieval）**：預期關鍵字出現在 **`evidence_triples`** 中的比例 → 圖譜/向量有沒有**抓到**正確資料
- **A（Answer）**：預期關鍵字出現在模型 **`answer` 欄位**中的比例 → 模型有沒有**回答出**來（不計 `evidence_triples`，避免虛報）
- **正確攔截率**：非 SOP 問題是否被 Guard 2 正確 `blocked`
- **端對端延遲**：整條 pipeline 耗時

R✅ A❌ 代表圖譜有撈到資料但模型沒有引用 → generation 瓶頸；R❌ A❌ 代表資料根本沒撈到 → retrieval 瓶頸。

### 逐題比較（Live 實測，2026-05-12，Qwen2.5-7B-Instruct-AWQ-int4，bi-encoder reranking + 動態 cap）

```
ID   │ 類別                        │ Graph RAG (R=檢索 A=回答)  │ Baseline RAG               │  Graph  │  Base
─────┼─────────────────────────────┼────────────────────────────┼────────────────────────────┼─────────┼────────
q01  │ anomaly_handling            │ R✅ A✅ 2/2               │ R✅ A✅ 2/2               │  4506ms │  1599ms
q02  │ sop_step_sequence  [↑HOP]   │ R✅ A✅ 4/4               │ R❌ A❌ 0/4               │  4297ms │  1218ms
q03  │ equipment_precondition      │ R✅ A✅ 4/4               │ R✅ A✅ 4/4               │  4028ms │  2246ms
q04  │ step_dependency             │ R✅ A✅ 3/3               │ R✅ A✅ 3/3               │  3060ms │  1452ms
q05  │ cross_doc_dependency [↑HOP] │ R✅ A⚠  1/2               │ R✅ A✅ 2/2               │  3489ms │  1030ms
q06  │ interlock_condition         │ R✅ A⚠  2/3               │ R✅ A⚠  1/3               │  4861ms │  2102ms
q07  │ vent_procedure      [↑HOP]  │ R✅ A✅ 2/2               │ R❌ A❌ 0/2               │  4766ms │  1255ms
q08  │ off_topic_blocked           │ ✅ blocked                │ ✅ blocked                │   586ms │   585ms
q09  │ off_topic_blocked           │ ✅ blocked                │ ✅ blocked                │   615ms │   664ms
q10  │ pump_check_sequence         │ R✅ A✅ 4/4               │ R❌ A⚠  1/4               │  4711ms │  1353ms
─────┼─────────────────────────────┼────────────────────────────┼────────────────────────────┼─────────┼────────
     │ TOTALS                      │ R:24/24(100%) A:22/24(92%) │ R:14/24(58%) A:13/24(54%)  │ avg 3491ms │ avg 1350ms
```

`[↑HOP]` 標記的題目需要多跳推理（step 鏈、跨文件依賴、DEPENDS_ON 鏈）。

### 量化成果

| 指標 | Graph RAG | Baseline RAG | 差距 |
|------|-----------|--------------|------|
| **Retrieval 命中率**（有抓到） | **100%** (24/24) | 58.3% (14/24) | **+41.7 pp** |
| **Answer 命中率**（有回答對） | **91.7%** (22/24) | 54.2% (13/24) | **+37.5 pp** |
| **多跳查詢 Answer 命中率** | **87.5%** (7/8) | 25.0% (2/8) | **+62.5 pp** |
| **正確攔截非 SOP 問題** | 2/2 | 2/2 | — |
| **平均端對端延遲** | 3491 ms | 1350 ms | +2141 ms |

> 以上數字為 Live 實測（Neo4j + vLLM **Qwen2.5-7B-Instruct-AWQ-int4** + Chroma 全服務啟動，2026-05-12），Graph RAG 採用 bi-encoder cosine reranking + 動態 cap（top_score × 0.5 閾值）。Answer 命中率以 **answer 欄位**計算，R 命中率以 **model_triples**（模型實際收到的 triples）計算。評測關鍵字為語意實體（移除 schema 術語如 FIRST_STEP、PRECONDITION 等）。

**結論：**

- Graph RAG Retrieval 達到 **100%**：bi-encoder reranking + 動態 cap 確保所有關鍵 triples 都進入模型 context（INTERLOCK_WITH triple 原本排在第 47 位，reranking 後升至第 1 位）。Baseline 只有 58%，步驟鏈（q02、q10）和依賴鏈（q07）完全抓不到。
- **7B Answer 91.7%**：較 3B（83.3%）提升 +8.4 pp；q04、q07 從 partial → 全對，顯示 7B 的指令跟隨能力明顯優於 3B。
- **q06 仍為最難題**（R✅ A⚠ 2/3）：即使 7B 也只答出 IL-E001 和 trigger，漏掉 RF，屬複雜屬性提取的 generation 上限。
- 兩條 pipeline 的 guardrail 行為完全一致，topic guard 與 injection guard 均正確攔截非 SOP 問題。

### Citation Traceability

每個 Graph RAG 回應包含 `source_docs` 欄位，自動從 `evidence_triples` 提取引用的 SOP 文件 ID：

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

### 執行評測

```bash
# 離線 mock 模式（不需要任何服務）
python3 scripts/eval_compare.py --mock

# 儲存 JSON 結果
python3 scripts/eval_compare.py --mock --output data/eval_results/latest.json

# Live 模式（需要 Neo4j + vLLM + Chroma 全部啟動後）
docker compose run --rm api python scripts/eval_compare.py
```

Live 結果存放於 `data/eval_results/live_baseline_vs_graph.json`。

---

## 三、LLM 模型效能比較

### 評估方式

使用相同的 10 道測試題跑 Graph RAG pipeline，**準確率只看模型實際回答的 `answer` 欄位**是否包含預期關鍵字，不計入 `evidence_triples`（避免虛報：關鍵字出現在圖譜證據但模型沒真正回答到）。

測試環境：RTX 3060 12GB，vLLM v0.6.3，`gpu_memory_utilization=0.8`，`max_model_len=4096`。

### 結果

| 模型 | 準確率（answer only） | avg 延遲 | vs FP16 延遲 |
|------|----------------------|---------|-------------|
| Qwen2.5-3B-Instruct（FP16） | **20/24 (83.3%)** | 3038 ms | — |
| Qwen2.5-3B-Instruct-GPTQ-int8 | **20/24 (83.3%)** | 2051 ms | -32% |
| Qwen2.5-3B-Instruct-GPTQ-int4 | 19/24 (79.2%) | 1725 ms | -43% |
| Qwen2.5-3B-Instruct-AWQ-int4 | 19/24 (79.2%) | 2312 ms | -24% |
| Qwen2.5-7B-Instruct-AWQ-int4 | **22/24 (91.7%)** | 3961 ms | +30% |

### 分析

- **GPTQ-int8 是最佳平衡點**：準確率與 FP16 完全相同（83.3%），推論速度快 32%，是延遲敏感場景的首選。
- **GPTQ-int4** 速度最快（-43%），準確率小降 4 pp（79.2%），VRAM 需求最低。
- **AWQ-int4（3B）**：準確率與 GPTQ-int4 相同（79.2%），但速度比 GPTQ-int4 慢（vLLM v0.6.3 強制 `--dtype float16`，在 RTX 3060 無速度優勢）。
- **7B AWQ-int4 準確率最高**（91.7%，+8.4 pp vs 3B FP16），延遲 3961 ms（+30%）。若 VRAM 允許，7B 是精度優先的最佳選擇。

> 評測採用 bi-encoder reranking + 動態 cap（top_score × 0.5 閾值），準確率以 **model_triples**（模型實際收到的 triples）的 answer 欄位計算（2026-05-12）。

---

## 四、Triple Reranking 策略比較

### 背景

Graph traversal 對 q06（interlock 查詢）會回傳 ~48 條 triples，但關鍵的 INTERLOCK_WITH triple 排在第 47 位。若直接截取前 N 條，模型永遠看不到它。Reranking 可將最相關的 triple 排到前面，並搭配動態截取上限確保關鍵 triple 進入 context。

### R / A 指標說明

- **R（Retrieval）**：預期關鍵字出現在模型實際收到的 triples（`model_triples`）中的比例
- **A（Answer）**：預期關鍵字出現在模型回答（`answer`）中的比例

### 演進過程（Qwen2.5-3B，10 道題，2026-05-12）

| 方法 | cap | R | A | avg 延遲 |
|------|-----|---|---|---------|
| 無 reranking（原始） | 25 | ~71%\* | 71% | 2683 ms |
| Bi-encoder cosine | 25 | 92% | 71% | 2506 ms |
| BM25 + Entity boost | 25 | 92% | 63% | 2718 ms |
| Cross-encoder（BGE） | 50 | 100% | 79% | 22618 ms 🔴 |
| BM25 + Entity boost | 50 | 100% | 67% | 2554 ms |
| Bi-encoder cosine | 50 | 100% | 83% | 3012 ms |
| Bi-encoder + Edge enrichment | 50 | 100% | 83% | 3067 ms |
| **Bi-encoder + Edge enrichment** | **動態** | **100%** | **88%** | **2944 ms** ✅ |

\* 原始 R 是對全部 triples 計算（含模型未看到的），實際模型可見 R 約 71%。以上為 3B 結果；7B AWQ-int4 + 動態 cap 達到 **91.7%**，延遲 3491 ms。

### Edge Enrichment：為空屬性邊補語意描述

Reranking 依賴 embedding 相似度，但 `DEPENDS_ON`、`NEXT_STEP` 等邊序列化後只含兩個 CamelCase ID，無任何語意文字：

```
(VerifyGasFlow)-[:DEPENDS_ON]->(CheckVacuumPump)
```

中文問題（「前置依賴」）的 embedding 向量與純英文 ID 距離遠，導致這類 triple 被排到第 47 位以後，超出任何合理的截取上限。

**解法**：在 `data/graph_seed/edges.json` 為這兩類邊加 `description` 屬性：

```json
{
  "type": "DEPENDS_ON",
  "from_id": "VerifyGasFlow",
  "to_id": "CheckVacuumPump",
  "properties": {
    "description": "VerifyGasFlow 執行前必須先完成前置依賴步驟 CheckVacuumPump"
  }
}
```

效果：q04 的 DEPENDS_ON triple 從第 **47** 位升至第 **13** 位，進入模型 context 範圍。

### 動態 Cap：以分數閾值取代固定截取數

固定 cap=50 是在不知道最少需要幾條的情況下預留 buffer，但會帶入不相關的雜訊 triples。

**動態 cap 邏輯**（`app/services/answer_service.py`）：

```python
# 保留分數 ≥ 最高分 50% 的所有 triples，最多 100 條
threshold = max(top_score * 0.50, 20)
scored = [t for t in scored_all if t[0] >= threshold][:100]
```

每題的實際 cap 依分數分布自動決定：

| 題目 | 總 triples | 動態 cap | 說明 |
|------|-----------|---------|------|
| q01 anomaly | 48 | 26 | 問題焦點明確，只需少量 triples |
| q05 cross_doc | 32 | 21 | 分數壓縮（最高只 49%），threshold 低但仍足夠 |
| q06 interlock | 41 | 29 | INTERLOCK_WITH 排 #1-2，其他補充 context |
| q07 vent_procedure | 41 | 35 | DEPENDS_ON #25 壓線，動態 cap 確保納入 |
| q10 pump_sequence | 46 | 41 | 多個 SOP 交叉，需要廣泛 context |

與固定 cap=50 相比，動態 cap 減少雜訊：context 更乾淨 → 模型注意力更集中 → Answer 從 83% → **88%**（3B），延遲 -123 ms。7B 準確率持平 91.7%，延遲少 470 ms（-12%）。

### 結論

- **最終組合：Bi-encoder + Edge enrichment + 動態 cap**：R 100%、A 91.7%（7B）、延遲 3491 ms
- **cap 的影響大於排序演算法**：從固定 25 → 動態讓 Answer 從 71% → 88%（3B）
- **Edge enrichment 解決跨語言 embedding 盲區**：空屬性邊加中文 description，DEPENDS_ON 排名從 #47 → #13
- **BM25 的問題**：中文問題 + 英文 triple，跨語言場景下 BM25 keyword 匹配不如 embedding 語意相似度（q05 退到 A❌）
- **Cross-encoder（BGE）** 準確率不錯（79%）但延遲 22 秒，不適合 online serving
- 剩餘失敗（q06 A⚠ 2/3）為 generation 上限：模型能看到 INTERLOCK_WITH triple，但無法可靠提取全部屬性值

---

## 五、Docker Compose 服務架構

### 服務拓撲

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose (單機)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  api  (fab-sop-rag-api)  python:3.12-slim               │   │
│  │  port  : 8000 → 8000                                    │   │
│  │  cmd   : uvicorn app.main:app --workers 1               │   │
│  │  volumes: chroma_data:/data/chroma                      │   │
│  │           hf_cache:/root/.cache/huggingface             │   │
│  │  health : GET /health → 200                             │   │
│  └──────┬──────────────────────┬───────────────────────────┘   │
│         │ bolt://neo4j:7687    │ http://vllm:8000/v1           │
│         ▼                      ▼                               │
│  ┌──────────────┐   ┌──────────────────────────────────────┐   │
│  │  neo4j       │   │  vllm  (vllm/vllm-openai:v0.6.3)    │   │
│  │  neo4j:5     │   │  port  : 8299 → 8000                 │   │
│  │  ports:      │   │  model : /llm/Qwen2.5-7B-AWQ-int4    │   │
│  │  7474 → 7474 │   │  GPU   : device_ids ["0"]            │   │
│  │  7687 → 7687 │   │  shm   : 16 GB                       │   │
│  │  heap: 512m~ │   │  ctx   : max-model-len 4096          │   │
│  │  1g          │   │  util  : gpu-memory-utilization 0.8  │   │
│  │  health: wget│   │  health: GET /v1/models              │   │
│  └──────────────┘   └──────────────────────────────────────┘   │
│                                                                 │
│  Named volumes                                                  │
│  ┌──────────────┬─────────────────────────────────────────┐    │
│  │ neo4j_data   │ Neo4j 圖資料庫持久化                    │    │
│  │ neo4j_logs   │ Neo4j 日誌                              │    │
│  │ chroma_data  │ Chroma 向量庫（embedding 索引）         │    │
│  │ hf_cache     │ HuggingFace model cache（embedding 模型）│    │
│  └──────────────┴─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 服務明細

| 服務 | Image | 對外 Port | 功能 | 健康檢查 |
|------|-------|-----------|------|---------|
| `api` | `python:3.12-slim`（自建） | `8000` | FastAPI + RAG pipeline | `GET /health` |
| `neo4j` | `neo4j:5` | `7474`（Browser）、`7687`（Bolt） | SOP 知識圖譜 | `wget localhost:7474` |
| `vllm` | `vllm/vllm-openai:v0.6.3.post1` | `8299` | Qwen2.5-7B-AWQ-int4 本地推論 | `GET /v1/models` |

### 啟動依賴順序

```
neo4j (service_healthy)  ──┐
                            ├──▶ api (service_started)
vllm  (service_started)  ──┘
```

- `api` 等 `neo4j` **healthy** 後才啟動（確保 Bolt 連線可用）
- `api` 等 `vllm` **started**（非 healthy），避免 vLLM 模型載入期間（3–10 min）卡住 api 啟動；前幾筆 LLM 呼叫會失敗直到 vLLM ready

### Volume 掛載說明

| Volume | 掛載位置 | 說明 |
|--------|---------|------|
| `neo4j_data` | `/data`（neo4j container） | 圖資料庫節點、邊、索引持久化 |
| `neo4j_logs` | `/logs`（neo4j container） | Neo4j server 日誌 |
| `chroma_data` | `/data/chroma`（api container） | Chroma 向量索引，ingest 後即持久化 |
| `hf_cache` | `/root/.cache/huggingface`（api container） | sentence-transformers embedding 模型快取 |
| `/home/jimmy/models` | `/llm`（vllm container，read-only） | 本機預先下載的 LLM 權重（Qwen2.5-7B-Instruct-AWQ-int4） |

### vLLM 推論參數

| 參數 | 值 | 說明 |
|------|----|------|
| `--model` | `/llm/Qwen2.5-7B-Instruct-AWQ-int4` | 從本機掛載路徑載入，離線不需 HuggingFace |
| `--max-model-len` | `4096` | context window 上限 |
| `--gpu-memory-utilization` | `0.8` | 保留 20% GPU VRAM 給 OS / CUDA overhead |
| `--tensor-parallel-size` | `1` | 單卡推論 |
| `--kv-cache-dtype` | `auto` | 自動選擇 KV cache 精度 |
| `shm_size` | `16 GB` | `ipc: host` + 大 shm 避免 PyTorch 張量共享問題 |

### api container 建構（Dockerfile）

```
python:3.12-slim
    │
    ├── apt install: gcc g++ curl
    ├── pip install -r requirements.txt   ← layer cache（先裝依賴）
    ├── COPY app/  scripts/  data/
    └── CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
              （single worker — 瓶頸是 vLLM GPU，多 worker 不增加吞吐量
                但會讓每個 worker 各自載 embedding model，浪費 RAM；
                並發靠 asyncio.to_thread 在 thread pool 處理即可）
```

---

## 六、事前準備

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

## 七、啟動步驟

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

## 八、呼叫 API

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

## 九、讀懂回應

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
  "source_docs": ["SOP_Etch_001"],
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

| reasoning_type | 說明 |
|----------------|------|
| `graph_rag` | 正常通過，答案來自圖譜遍歷 |
| `baseline_rag` | 正常通過，答案來自向量文本片段（eval 對照組） |
| `answered_with_warning` | Guard 4（事實接地）警告，confidence 降為 0.5 |
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

## 十、Guardrail 四道關卡

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

## 十一、資料說明

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

### 知識圖譜 Schema

**節點標籤（Node Labels）**

| 標籤 | 數量 | 關鍵屬性 |
|------|------|---------|
| `SOPDocument` | 3 | `id`, `title`, `version`, `equipment` |
| `SOPStep` | 12 | `id`, `description`, `sop_doc`, `step_number` |
| `Equipment` | 8 | `id`, `type`, `description`, 機台專屬參數 |
| `Anomaly` | 2 | `id`, `description`, `threshold_*` |
| `ProcessCondition` | 4 | `id`, `parameter`, `target`, `tolerance`, `unit` |

**邊類型（Edge Types）與連接關係**

```
Anomaly ──[TRIGGERS_SOP]──────────────▶ SOPDocument
           觸發應執行的 SOP

SOPDocument ──[FIRST_STEP]────────────▶ SOPStep
               SOP 的第一個步驟

SOPStep ──[NEXT_STEP]─────────────────▶ SOPStep
           步驟順序鏈

SOPStep ──[DEPENDS_ON]────────────────▶ SOPStep
           執行前必須完成的前置步驟

SOPStep ──[DEFINED_IN]────────────────▶ SOPDocument
           步驟所屬文件

SOPStep ──[REQUIRES_STATUS {required_status}]──▶ Equipment
           步驟執行時設備必須處於的狀態

SOPDocument ──[PRECONDITION {required_status, condition_id}]──▶ Equipment
               整份 SOP 的設備前置條件

Equipment ──[INTERLOCK_WITH {interlock_id, trigger, action}]──▶ Equipment
              設備聯鎖（自動安全保護）

SOPDocument ──[CROSS_DOC_DEPENDENCY {reason}]──▶ SOPDocument
               跨文件依賴（某 SOP 引用另一份 SOP 的定義）
```

**完整圖譜示意**

```
 PressureAnomaly ──TRIGGERS_SOP──▶ SOP_Etch_001 ──FIRST_STEP──▶ CheckVacuumPump
 PumpDegradation ──TRIGGERS_SOP──▶ SOP_Pump_002                       │
                                        │                         NEXT_STEP
                                   CROSS_DOC                           ▼
                                   DEPENDENCY                  VerifyGasFlow
                                        │                              │
                                        ▼                         NEXT_STEP
                                   SOP_Pump_002 ──FIRST_STEP──▶        ▼
                                        │          ReadPumpStatus  InspectChamberLeak
                                   CROSS_DOC                           │
                                   DEPENDENCY                     NEXT_STEP
                                        │                              ▼
                                        ▼                    RestoreProcessCondition
                                   SOP_Vent_003

 CheckVacuumPump ──REQUIRES_STATUS {RUNNING}──▶ TurboVacuumPump
 SOP_Etch_001 ─────PRECONDITION {RUNNING}─────▶ TurboVacuumPump
 SOP_Etch_001 ─────PRECONDITION {STANDBY_OR_OFF}─▶ RFPowerSupply
 EtchStation ──INTERLOCK_WITH {IL-E001, pressure>10mTorr, disable RF}──▶ PressureInterlock
```

> 全部為**教學範例資料**，不代表任何真實製造商的操作規範。

### 新增自己的 SOP 文件（自動抽取）

手動維護 `nodes.json` / `edges.json` 在 SOP 數量多時不可行。
使用 `extract_graph_from_sop.py` 讓 LLM 自動從 Markdown 抽取圖譜：

```
SOP Markdown 文件
      │
      ▼
scripts/extract_graph_from_sop.py   ← LLM 兩次 pass（nodes → edges）
      │
      ├─ Pass 1：抽取節點（SOPDocument / SOPStep / Equipment / Anomaly / ProcessCondition）
      └─ Pass 2：抽取邊（TRIGGERS_SOP / FIRST_STEP / NEXT_STEP / REQUIRES_STATUS / …）
      │
      ▼
data/graph_seed/nodes_extracted.json
data/graph_seed/edges_extracted.json
      │
      ▼（人工 review 後執行 --merge）
data/graph_seed/nodes.json  ←  merge（dedup by id）
data/graph_seed/edges.json  ←  merge（dedup by type+from+to）
      │
      ▼
scripts/ingest_graph.py  →  Neo4j
```

**操作步驟：**

```bash
# 步驟 1：將新 SOP 放入 data/sop_docs/
cp my_new_sop.md data/sop_docs/

# 步驟 2：LLM 自動抽取（需要 vLLM 服務運行中）
docker compose run --rm api python scripts/extract_graph_from_sop.py

# 步驟 3：Review 抽取結果（確認節點/邊正確）
cat data/graph_seed/nodes_extracted.json
cat data/graph_seed/edges_extracted.json

# 步驟 4：確認無誤後合併進 graph seed
docker compose run --rm api python scripts/extract_graph_from_sop.py --merge

# 步驟 5：重新 ingest（idempotent，不會重複寫入）
docker compose run --rm api python scripts/ingest_all.py
```

**可用 flags：**

| Flag | 說明 |
|------|------|
| `--file <path>` | 只處理單一檔案 |
| `--dry-run` | 印出抽取結果，不寫檔案（preview 用） |
| `--merge` | 直接合併進 `nodes.json` / `edges.json` |
| `--output-dir <dir>` | 指定輸出目錄（預設 `data/graph_seed/`） |

### LLM 自動抽取的能力與限制

本系統以 3 份 SOP Markdown 實測，結果如下：

| | 手工標注（ground truth） | LLM 自動抽取（Qwen2.5-3B） |
|--|--|--|
| Nodes | 29 | 25（覆蓋率 ~86%） |
| Edges | 48 | 32（覆蓋率 ~67%） |

**已驗證可靠的部分：**
- SOPDocument / SOPStep 節點幾乎全抓到
- `TRIGGERS_SOP`、`FIRST_STEP`、`NEXT_STEP`、`REQUIRES_STATUS` 主要關係正確

**常見遺漏：**
- 細粒度 ProcessCondition 節點（如 `EtchGasFlow_HBr`、`ChamberLeakRate`）
- `PRECONDITION`、`INTERLOCK_WITH`、`CROSS_DOC_DEPENDENCY` 等跨文件關係
- 節點 ID 命名與原文稍有出入時，衍生 edge 因 validate_edges 檢查而被丟棄

**建議工作流程（Human-in-the-loop）：**

```
LLM 自動抽取（--dry-run preview）
        ↓
工程師 review nodes_extracted.json / edges_extracted.json
  → 補漏節點、修正 ID 命名、加入跨文件 edge
        ↓
--merge 合併進 graph seed
        ↓
docker compose exec api python scripts/ingest_graph.py
```

LLM 抽取做 **first-pass**，降低人工建圖門檻（估計節省 ~60% 手工時間）；
工程師只需 **review 與補漏**，而非從零撰寫 JSON。

### 抽取品質保障（type/label 白名單驗證）

`extract_graph_from_sop.py` 內建三層 edge 驗證，在寫入前過濾 LLM 輸出中的常見錯誤：

| 驗證層 | 檢查內容 | 常見 LLM 錯誤 |
|--------|---------|--------------|
| **type 白名單** | `type` 必須屬於 9 個已知 edge 類型 | LLM 自創 `LINKED_TO`、`PART_OF` 等不存在的 type |
| **label 白名單** | `from_label` / `to_label` 必須屬於 5 個節點標籤 | LLM 把節點 ID（如 `PressureAnomaly`）誤填到 `from_label` 欄位 |
| **node ID 存在性** | `from_id` / `to_id` 必須出現在已抽取的節點清單 | 邊引用了不存在的節點（懸空邊） |

被過濾的 edge 會以 WARNING 印出，方便 review。

### 並行處理多份 SOP（~2.6x 加速）

腳本預設使用 `ThreadPoolExecutor` 同時處理多份 SOP 文件（最多 4 個 worker）：

```
3 份 SOP 文件
 │
 ├── Thread 1: etch_pressure_anomaly.md
 ├── Thread 2: vacuum_pump_check.md
 └── Thread 3: chamber_vent_procedure.md
         │
         ▼（as_completed，哪個先好先 merge）
    merge_nodes / merge_edges（帶 dedup）
         │
         ▼
nodes_extracted.json / edges_extracted.json
```

實測（3 份 SOP，Qwen2.5-3B，RTX 3060）：

| 模式 | 耗時 |
|------|------|
| 循序（舊版） | ~90 秒 |
| 並行（4 workers） | ~34 秒 |
| **加速比** | **~2.6x** |

> SOP 文件數量越多，並行加速效果越明顯（I/O + LLM call 為主要等待時間）。

### 結構邊確定性推導（不依賴 LLM）

`DEFINED_IN`（步驟所屬文件）和 `FIRST_STEP`（SOP 的第一步）兩類 edge 可從節點屬性 100% 確定性推導，不需要 LLM：

- **DEFINED_IN**：所有 `SOPStep` 節點都有 `sop_doc` 屬性，直接生成 `(step) -[DEFINED_IN]-> (sop_doc)` 邊
- **FIRST_STEP**：`step_number == 1` 的步驟即為 FIRST_STEP，直接生成 `(sop_doc) -[FIRST_STEP]-> (step)` 邊

腳本會自動剔除 LLM 輸出中的這兩類 edge，改用確定性推導，避免 LLM 遺漏或命名不一致。

---

## 十二、服務端點總覽

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

## 十三、停止與重置

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

## 十四、常見問題

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

## 十五、併發壓力測試

### 測試環境

| 硬體 | 規格 |
|------|------|
| CPU | AMD Ryzen 7 3700X（8 核 16 線程） |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 3060 12 GB VRAM |
| LLM | Qwen2.5-3B-Instruct（vLLM v0.6.3，`gpu_memory_utilization=0.8`） |
| OS | Linux（Driver 550.163） |

### 測試方式

使用 Python `asyncio` + `httpx` 同時送出 N 個 POST `/v1/ask` 請求，問題固定為「蝕刻站發生壓力異常時應執行哪份SOP」（Graph RAG pipeline，`enable_guards=false`）。

### 結果

```
N（並發數）│   成功 │  Wall ms │  Avg ms │  Max ms │  備註
──────────────────────────────────────────────────────────
         1 │      1 │     1176 │    1176 │    1176 │  baseline
         2 │      2 │     1645 │    1645 │    1645 │  幾乎同時完成
         4 │      4 │     2450 │    2449 │    2450 │  ✅ 正常
         8 │      8 │     4020 │    4018 │    4020 │  ✅ 可接受
        16 │     16 │     7170 │    7162 │    7168 │  ⚠️ 開始排隊
        32 │     32 │    14336 │   10863 │   14333 │  ⚠️ 接近 timeout
```

### 分析

- **FastAPI 層不是瓶頸**：`/v1/ask` 使用 `asyncio.to_thread()` 將同步 pipeline 丟進 thread pool，event loop 保持非阻塞，多請求可真正並發執行。
- **瓶頸在 vLLM GPU KV cache**：RTX 3060 12 GB 上跑 Qwen2.5-3B，有效並發約 **8–16 個請求**在合理延遲（< 5s）內。
- **延遲線性成長**：每多一倍並發，延遲約多 1.5–2x，符合 vLLM 的批次推論佇列行為。

### 適用場景

| 場景 | 建議並發 | 延遲 | 結論 |
|------|---------|------|------|
| 單人 demo / PoC | 1–2 | ~1.2–1.6 s | ✅ |
| 小組內部工具（5–10 人） | 4–8 | ~2.4–4 s | ✅ 可接受 |
| 部門系統（~20 人） | 16 | ~7 s | ⚠️ 需換更大 GPU |
| 高併發生產環境 | 32+ | 14 s+ | ❌ 需多 GPU 或多實例 + LB |

### 進一步擴展方向

若需支援更高併發，建議依序：

1. **多 uvicorn worker**：`uvicorn app.main:app --workers 4`（利用多核 CPU）
2. **更大 GPU**：RTX 4090 24 GB 或 A100 40 GB，KV cache 容量倍增
3. **vLLM tensor parallelism**：多張 GPU 水平切割模型（`--tensor-parallel-size 2`）
4. **多實例 + load balancer**：多台機器各跑一套 stack，前端加 Nginx 做 round-robin

---

## 十六、架構說明

```
fab-sop-rag/
├── app/
│   ├── api/routes.py          # /health, /v1/health, /v1/ask
│   ├── services/
│   │   ├── pipeline.py           # 四道 guardrail 的主控流程（Graph RAG）
│   │   ├── baseline_pipeline.py  # 向量只讀 RAG（評測對照組）
│   │   ├── retrieval_service.py  # 圖譜 + 向量混合檢索
│   │   ├── guardrails.py         # guard_injection / guard_topic / guard_evidence / guard_grounding
│   │   ├── judge_service.py      # LLM-as-judge（主題過濾 + 事實接地）
│   │   └── answer_service.py     # LLM 答案生成
│   ├── middleware/request_id.py  # X-Request-ID 關聯 ID
│   ├── config.py              # pydantic-settings（.env 驅動）
│   └── schemas.py             # Pydantic 資料模型
├── scripts/
│   ├── extract_graph_from_sop.py  # SOP Markdown → nodes/edges JSON（LLM 自動抽取）
│   ├── ingest_graph.py            # nodes/edges JSON → Neo4j
│   ├── ingest_vector.py           # SOP Markdown → Chroma
│   ├── ingest_all.py              # ingest_graph + ingest_vector 一次跑完
│   └── eval_compare.py            # Baseline vs Graph RAG 評測（--mock 離線可用）
├── data/
│   ├── sop_docs/              # 原始 SOP Markdown
│   ├── graph_seed/            # 節點 + 邊 JSON
│   ├── sample_queries/        # 測試問題集（10 道含預期關鍵字）
│   └── eval_results/          # 評測輸出 JSON
├── docker-compose.yml
├── Dockerfile
└── .env.example
```
