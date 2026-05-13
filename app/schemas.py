from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Request / Response for POST /v1/ask ──────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="使用者提問")
    enable_guards: bool = Field(True, description="是否啟用 guardrails")
    debug: bool = Field(False, description="是否回傳 debug 資訊")
    max_hop: int = Field(2, ge=1, le=4, description="Graph 展開跳數")
    top_k: int = Field(4, ge=1, le=20, description="Vector retrieval top-k")


class GuardrailResult(BaseModel):
    """
    Python 屬性名稱使用 `passed`，序列化輸出時轉為 `pass`（JSON 關鍵字）。
    使用 model.model_dump(by_alias=True) 取得正確的 JSON 格式。
    """

    model_config = ConfigDict(populate_by_name=True)

    stage: str = Field(..., description="input | retrieval | output")
    name: str = Field(..., description="guardrail 名稱")
    passed: bool = Field(..., serialization_alias="pass", description="是否通過")
    reason: str = Field(..., description="原因說明")


class DebugInfo(BaseModel):
    context: str = Field(..., description="傳入 LLM 的 context 字串")
    llm_raw_output: str = Field(..., description="LLM 原始輸出")
    retrieval_count: int = Field(..., description="檢索到的三元組數量")
    stage_latencies: dict[str, int] = Field(
        default_factory=dict,
        description="各 pipeline 階段耗時 (ms)：guard_injection, guard_topic, retrieval, generation, guard_grounding",
    )


class AskResponse(BaseModel):
    question: str
    status: str = Field(..., description="answered | blocked")
    answer: str
    entities: list[str] = Field(default_factory=list, description="用於 graph 查詢的實體")
    evidence_triples: list[str] = Field(default_factory=list, description="圖譜三元組（graph traversal 全部結果）")
    model_triples: list[str] = Field(default_factory=list, description="實際傳入 LLM 的三元組（rerank + cap 後）")
    source_docs: list[str] = Field(
        default_factory=list,
        description="答案引用的 SOP 文件 ID，從 evidence_triples 提取，用於 citation traceability",
    )
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    reasoning_type: str = Field(
        ...,
        description=(
            "graph_rag | baseline_rag | blocked_injection | blocked_off_topic "
            "| blocked_low_evidence | answered_with_warning"
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    request_id: str = Field("-", description="X-Request-ID correlation token")
    debug: Optional[DebugInfo] = None


# ── POST /v1/ingest ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source_file: str = Field(..., min_length=1, max_length=200, description="來源檔案名稱，用於 graph versioning")
    nodes: list[dict] = Field(..., description="節點列表，格式同 data/graph_seed/nodes.json")
    edges: list[dict] = Field(..., description="邊列表，格式同 data/graph_seed/edges.json")


class IngestResponse(BaseModel):
    status: str = Field(..., description="ok | error")
    nodes_merged: int = Field(..., description="成功 MERGE 的節點數")
    edges_merged: int = Field(..., description="成功 MERGE 的邊數")
    edges_skipped: int = Field(..., description="因節點不存在而跳過的邊數")
    detail: Optional[str] = None


# ── GET /v1/health ────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    latency_ms: Optional[int] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    services: dict[str, ServiceStatus]
    request_id: str = "-"


# ── Standardised error envelope ───────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """
    Uniform error body returned by the global exception handlers.
    Every error includes the request_id so engineers can correlate
    a client-visible failure with the server log.
    """

    error: str = Field(..., description="Short error category")
    detail: Any = Field(..., description="Human-readable or structured detail")
    request_id: str = Field("-", description="X-Request-ID of the failed request")
