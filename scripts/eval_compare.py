"""
Baseline RAG vs Graph RAG — Quantitative Evaluation

Runs all queries from data/sample_queries/fab_queries.json through both
pipelines and produces a comparison report covering:

  • Keyword hit rate   — fraction of expected_keywords found in response
  • Correct-block rate — off-topic queries correctly rejected
  • Latency (ms)       — end-to-end pipeline time
  • Multi-hop delta    — keyword improvement on step-chain queries

Usage
-----
  # Online mode (requires Neo4j + vLLM + Chroma running):
  python scripts/eval_compare.py

  # Mock mode (offline, uses pre-computed results — no services needed):
  python scripts/eval_compare.py --mock

  # Save JSON results:
  python scripts/eval_compare.py --mock --output data/eval_results/latest.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

QUERY_TIMEOUT_SEC = 120  # per-query hard limit

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "data" / "sample_queries" / "fab_queries.json"
RESULTS_DIR = ROOT / "data" / "eval_results"

# Multi-hop query IDs (require graph traversal to answer correctly)
MULTIHOP_IDS = {"q02", "q05", "q07"}

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(resp: dict, expected: dict) -> dict:
    """Return scoring dict for one (response, expected) pair."""
    behavior = expected.get("expected_behavior", "retrieved_and_answered")
    keywords = expected.get("expected_keywords", [])

    if behavior == "blocked_by_topic_guard":
        correct = resp.get("status") == "blocked"
        return {
            "correct_block": correct,
            "keyword_hits": 0,
            "keyword_total": 0,
            "retrieval_hits": 0,
            "retrieval_total": 0,
            "is_block_query": True,
        }

    # Retrieval score: keywords found in model_triples — the triples actually passed
    # to the LLM (after rerank + cap). Falls back to evidence_triples for baseline
    # pipeline which has no model_triples field.
    model_triples = resp.get("model_triples") or resp.get("evidence_triples", [])
    evidence_text = " ".join(model_triples)
    retrieval_hits = sum(1 for kw in keywords if kw.lower() in evidence_text.lower())

    # Answer score: keywords found only in the model's generated answer (did the
    # LLM correctly use the retrieved context?).
    haystack = resp.get("answer", "")
    hits = sum(1 for kw in keywords if kw.lower() in haystack.lower())
    return {
        "correct_block": None,
        "keyword_hits": hits,
        "keyword_total": len(keywords),
        "retrieval_hits": retrieval_hits,
        "retrieval_total": len(keywords),
        "is_block_query": False,
    }


# ── Mock data (pre-computed realistic results) ────────────────────────────────
# These values reflect the expected behaviour of each pipeline on the
# 10 benchmark queries without requiring running services.

_MOCK_RESULTS = {
    "q01": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 231,
            "answer": "依據圖譜，PressureAnomaly 透過 TRIGGERS_SOP 關係觸發 SOP_Etch_001（蝕刻站壓力異常處置程序）。",
            "evidence_triples": [
                "(PressureAnomaly)-[:TRIGGERS_SOP]->(SOP_Etch_001)",
                "(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)",
            ],
            "entities": ["PressureAnomaly", "SOP_Etch_001"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 143,
            "answer": "蝕刻站壓力異常時，應執行 SOP_Etch_001 蝕刻站壓力異常處置程序。",
            "evidence_triples": [
                "# SOP_Etch_001 — 蝕刻站壓力異常處置程序\n異常類型 | PressureAnomaly",
            ],
            "entities": [],
        },
    },
    "q02": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 318,
            "answer": (
                "SOP_Etch_001 的步驟順序如下：\n"
                "1. CheckVacuumPump（FIRST_STEP）\n"
                "2. VerifyGasFlow（NEXT_STEP）\n"
                "3. InspectChamberLeak（NEXT_STEP）\n"
                "4. RestoreProcessCondition（NEXT_STEP）"
            ),
            "evidence_triples": [
                "(SOP_Etch_001)-[:FIRST_STEP]->(CheckVacuumPump)",
                "(CheckVacuumPump)-[:NEXT_STEP]->(VerifyGasFlow)",
                "(VerifyGasFlow)-[:NEXT_STEP]->(InspectChamberLeak)",
                "(InspectChamberLeak)-[:NEXT_STEP]->(RestoreProcessCondition)",
            ],
            "entities": ["SOP_Etch_001", "CheckVacuumPump"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 162,
            "answer": (
                "SOP_Etch_001 的第一個步驟是 CheckVacuumPump，"
                "接著執行 VerifyGasFlow。後續步驟需參閱完整文件。"
            ),
            "evidence_triples": [
                "步驟 1：CheckVacuumPump（第一步驟）確認真空泵浦運作狀態。\n"
                "步驟 2：VerifyGasFlow（下一步驟）確認製程氣體流量。",
            ],
            "entities": [],
        },
    },
    "q03": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 244,
            "answer": (
                "執行 SOP_Etch_001 前置條件：\n"
                "• TurboVacuumPump 狀態必須為 RUNNING（PRECONDITION-P01）\n"
                "• RFPowerSupply 狀態必須為 STANDBY 或 OFF（PRECONDITION-P02）"
            ),
            "evidence_triples": [
                "(SOP_Etch_001)-[:PRECONDITION {required_status: 'RUNNING', condition_id: 'PRECONDITION-P01'}]->(TurboVacuumPump)",
                "(SOP_Etch_001)-[:PRECONDITION {required_status: 'STANDBY_OR_OFF', condition_id: 'PRECONDITION-P02'}]->(RFPowerSupply)",
            ],
            "entities": ["SOP_Etch_001", "TurboVacuumPump", "RFPowerSupply"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 155,
            "answer": (
                "執行 SOP_Etch_001 前，TurboVacuumPump 狀態需為 RUNNING，"
                "RFPowerSupply 需為 STANDBY 或 OFF。"
            ),
            "evidence_triples": [
                "[PRECONDITION-P01] TurboVacuumPump 狀態 = RUNNING\n"
                "[PRECONDITION-P02] RFPowerSupply 狀態 = STANDBY 或 OFF",
            ],
            "entities": [],
        },
    },
    "q04": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 198,
            "answer": "CheckVacuumPump 步驟透過 REQUIRES_STATUS 關係要求 TurboVacuumPump 處於 RUNNING 狀態。",
            "evidence_triples": [
                "(CheckVacuumPump)-[:REQUIRES_STATUS {required_status: 'RUNNING'}]->(TurboVacuumPump)",
            ],
            "entities": ["CheckVacuumPump", "TurboVacuumPump"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 138,
            "answer": "CheckVacuumPump 步驟需確認 TurboVacuumPump 狀態為 RUNNING（泵浦正常運作）。",
            "evidence_triples": [
                "CheckVacuumPump REQUIRES_STATUS TurboVacuumPump.RUNNING",
            ],
            "entities": [],
        },
    },
    "q05": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 341,
            "answer": (
                "依據圖譜 CROSS_DOC_DEPENDENCY 關係：\n"
                "TurboVacuumPump 的狀態定義（RUNNING/FAULT 等）源自 SOP_Pump_002 第 4 節。\n"
                "此為跨文件設備依賴關係。"
            ),
            "evidence_triples": [
                "(SOP_Etch_001)-[:CROSS_DOC_DEPENDENCY {reason: 'TurboVacuumPump 狀態定義（RUNNING/FAULT 等）源自 SOP_Pump_002 第 4 節'}]->(SOP_Pump_002)",
            ],
            "entities": ["SOP_Etch_001", "SOP_Pump_002"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 149,
            "answer": "根據文件，SOP_Pump_002 為真空泵浦狀態檢查程序，定義了泵浦狀態代碼。",
            "evidence_triples": [
                "SOP_Etch_001 步驟 1 中引用本文件泵浦狀態定義",
            ],
            "entities": [],
        },
    },
    "q06": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 267,
            "answer": (
                "EtchStation 壓力 Interlock（IL-E001）：\n"
                "• 觸發條件：pressure > 10 mTorr\n"
                "• 觸發動作：disable RF power（自動關閉 RF 電源）\n"
                "• 聯鎖關係：EtchStation INTERLOCK_WITH PressureInterlock"
            ),
            "evidence_triples": [
                "(EtchStation)-[:INTERLOCK_WITH {interlock_id: 'IL-E001', trigger: 'pressure > 10 mTorr', action: 'disable RF power'}]->(PressureInterlock)",
            ],
            "entities": ["EtchStation", "PressureInterlock"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 152,
            "answer": "EtchStation 的壓力 Interlock 在壓力 > 10 mTorr 時觸發，自動關閉 RF 電源（IL-E001）。",
            "evidence_triples": [
                "IL-E001 | 壓力 > 10 mTorr | 自動關閉 RF 電源",
            ],
            "entities": [],
        },
    },
    "q07": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 356,
            "answer": (
                "OpenVentValve 步驟之前需完成：\n"
                "1. IsolateRFAndGas（DEPENDS_ON）\n"
                "2. SwitchPumpToStandby（DEPENDS_ON，NEXT_STEP）\n"
                "依據圖譜 DEPENDS_ON 和 NEXT_STEP 關係鏈。"
            ),
            "evidence_triples": [
                "(OpenVentValve)-[:DEPENDS_ON]->(SwitchPumpToStandby)",
                "(SwitchPumpToStandby)-[:DEPENDS_ON]->(IsolateRFAndGas)",
                "(SwitchPumpToStandby)-[:NEXT_STEP]->(OpenVentValve)",
                "(IsolateRFAndGas)-[:NEXT_STEP]->(SwitchPumpToStandby)",
            ],
            "entities": ["OpenVentValve", "SwitchPumpToStandby", "IsolateRFAndGas"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 157,
            "answer": "OpenVentValve 之前需先執行 IsolateRFAndGas 步驟，確保能源隔離。",
            "evidence_triples": [
                "步驟 1：IsolateRFAndGas 確認所有高風險能源已隔離\n步驟 3：OpenVentValve",
            ],
            "entities": [],
        },
    },
    "q08": {
        "graph": {
            "status": "blocked", "reasoning_type": "blocked_off_topic", "latency_ms": 89,
            "answer": "此問題不屬於晶圓廠 SOP 知識庫查詢範疇。",
            "evidence_triples": [], "entities": [],
        },
        "baseline": {
            "status": "blocked", "reasoning_type": "blocked_off_topic", "latency_ms": 87,
            "answer": "此問題不屬於晶圓廠 SOP 知識庫查詢範疇。",
            "evidence_triples": [], "entities": [],
        },
    },
    "q09": {
        "graph": {
            "status": "blocked", "reasoning_type": "blocked_off_topic", "latency_ms": 91,
            "answer": "此問題不屬於晶圓廠 SOP 知識庫查詢範疇。",
            "evidence_triples": [], "entities": [],
        },
        "baseline": {
            "status": "blocked", "reasoning_type": "blocked_off_topic", "latency_ms": 88,
            "answer": "此問題不屬於晶圓廠 SOP 知識庫查詢範疇。",
            "evidence_triples": [], "entities": [],
        },
    },
    "q10": {
        "graph": {
            "status": "answered", "reasoning_type": "graph_rag", "latency_ms": 287,
            "answer": (
                "SOP_Pump_002 的步驟順序（從 ReadPumpStatus 開始）：\n"
                "1. ReadPumpStatus\n"
                "2. CheckBearingVibration\n"
                "3. VerifyPumpCooling\n"
                "4. LogAndClearAlarm"
            ),
            "evidence_triples": [
                "(SOP_Pump_002)-[:FIRST_STEP]->(ReadPumpStatus)",
                "(ReadPumpStatus)-[:NEXT_STEP]->(CheckBearingVibration)",
                "(CheckBearingVibration)-[:NEXT_STEP]->(VerifyPumpCooling)",
                "(VerifyPumpCooling)-[:NEXT_STEP]->(LogAndClearAlarm)",
            ],
            "entities": ["SOP_Pump_002", "ReadPumpStatus", "CheckBearingVibration"],
        },
        "baseline": {
            "status": "answered", "reasoning_type": "baseline_rag", "latency_ms": 161,
            "answer": (
                "SOP_Pump_002 的泵浦檢查步驟：ReadPumpStatus → CheckBearingVibration → "
                "VerifyPumpCooling → LogAndClearAlarm。"
            ),
            "evidence_triples": [
                "步驟 1：ReadPumpStatus\n步驟 2：CheckBearingVibration\n"
                "步驟 3：VerifyPumpCooling\n步驟 4：LogAndClearAlarm",
            ],
            "entities": [],
        },
    },
}


# ── Live pipeline runner ──────────────────────────────────────────────────────

def _run_live(queries: list[dict]) -> list[dict]:
    """Run both pipelines live against real services."""
    from app.schemas import AskRequest
    from app.services.pipeline import run_pipeline
    from app.services.baseline_pipeline import run_baseline_pipeline

    # Pre-warm lazy singletons so the first query doesn't absorb cold-start
    print("[INFO] Pre-warming services (embedding model, Neo4j, LLM)...")
    try:
        from app.services.vector_store import _get_vector_store
        _get_vector_store()
        from app.services.graph_store import _get_driver
        _get_driver()
        from app.services.llm_client import chat_completion
        chat_completion("ping", max_tokens=1)
        print("[INFO] Pre-warm complete")
    except Exception as e:
        print(f"[WARNING] Pre-warm failed (non-fatal): {e}")

    results = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for q in queries:
            req = AskRequest(question=q["question"], enable_guards=True, top_k=4, max_hop=2)
            qid = q["id"]

            t0 = time.perf_counter()
            fut = pool.submit(run_pipeline, req)
            try:
                gr = fut.result(timeout=QUERY_TIMEOUT_SEC)
                g_ms = int((time.perf_counter() - t0) * 1000)
                graph_entry = {
                    "status": gr.status,
                    "reasoning_type": gr.reasoning_type,
                    "latency_ms": g_ms,
                    "answer": gr.answer,
                    "evidence_triples": gr.evidence_triples,
                    "model_triples": gr.model_triples,
                    "entities": gr.entities,
                }
            except FuturesTimeoutError:
                g_ms = QUERY_TIMEOUT_SEC * 1000
                print(f"[WARNING] Graph pipeline timed out for {qid} after {QUERY_TIMEOUT_SEC}s")
                graph_entry = {
                    "status": "error", "reasoning_type": "timeout", "latency_ms": g_ms,
                    "answer": f"[TIMEOUT after {QUERY_TIMEOUT_SEC}s]",
                    "evidence_triples": [], "model_triples": [], "entities": [],
                }

            t0 = time.perf_counter()
            fut = pool.submit(run_baseline_pipeline, req)
            try:
                br = fut.result(timeout=QUERY_TIMEOUT_SEC)
                b_ms = int((time.perf_counter() - t0) * 1000)
                baseline_entry = {
                    "status": br.status,
                    "reasoning_type": br.reasoning_type,
                    "latency_ms": b_ms,
                    "answer": br.answer,
                    "evidence_triples": br.evidence_triples,
                    "entities": br.entities,
                }
            except FuturesTimeoutError:
                b_ms = QUERY_TIMEOUT_SEC * 1000
                print(f"[WARNING] Baseline pipeline timed out for {qid} after {QUERY_TIMEOUT_SEC}s")
                baseline_entry = {
                    "status": "error", "reasoning_type": "timeout", "latency_ms": b_ms,
                    "answer": f"[TIMEOUT after {QUERY_TIMEOUT_SEC}s]",
                    "evidence_triples": [], "entities": [],
                }

            results.append({"id": qid, "graph": graph_entry, "baseline": baseline_entry})
    return results


# ── Report rendering ──────────────────────────────────────────────────────────

def _bar(hits: int, total: int) -> str:
    if total == 0:
        return "N/A"
    pct = hits / total * 100
    filled = int(pct / 10)
    return f"{'█' * filled}{'░' * (10 - filled)} {hits}/{total} ({pct:.0f}%)"


def render_report(queries: list[dict], results: dict) -> str:
    rows = []
    g_kw_hits = g_kw_total = 0
    b_kw_hits = b_kw_total = 0
    g_ret_hits = g_ret_total = 0
    b_ret_hits = b_ret_total = 0
    g_block_hits = b_block_hits = block_total = 0
    g_latencies = []
    b_latencies = []
    mh_g_hits = mh_g_total = mh_b_hits = mh_b_total = 0

    for q in queries:
        qid = q["id"]
        cat = q["category"]
        res = results.get(qid, {})
        g_resp = res.get("graph", {})
        b_resp = res.get("baseline", {})

        g_score = score_response(g_resp, q)
        b_score = score_response(b_resp, q)

        g_ms = g_resp.get("latency_ms", 0)
        b_ms = b_resp.get("latency_ms", 0)
        g_latencies.append(g_ms)
        b_latencies.append(b_ms)

        is_mh = qid in MULTIHOP_IDS
        tag = " [↑HOP]" if is_mh else ""

        if g_score["is_block_query"]:
            block_total += 1
            g_ok = "✅" if g_score["correct_block"] else "❌"
            b_ok = "✅" if b_score["correct_block"] else "❌"
            if g_score["correct_block"]:
                g_block_hits += 1
            if b_score["correct_block"]:
                b_block_hits += 1
            rows.append(
                f"  {qid} │ {cat[:26]:<26}{tag:<7} │ {g_ok} blocked                    │ {b_ok} blocked                    │ {g_ms:>5}ms │ {b_ms:>5}ms"
            )
        else:
            gh, gt = g_score["keyword_hits"], g_score["keyword_total"]
            bh, bt = b_score["keyword_hits"], b_score["keyword_total"]
            grh = g_score["retrieval_hits"]
            brh = b_score["retrieval_hits"]
            g_kw_hits += gh
            g_kw_total += gt
            b_kw_hits += bh
            b_kw_total += bt
            g_ret_hits += grh
            g_ret_total += gt
            b_ret_hits += brh
            b_ret_total += bt
            if is_mh:
                mh_g_hits += gh
                mh_g_total += gt
                mh_b_hits += bh
                mh_b_total += bt
            # R = retrieval (evidence_triples), A = answer
            g_r = "R✅" if grh == gt else ("R⚠" if grh > 0 else "R❌")
            b_r = "R✅" if brh == bt else ("R⚠" if brh > 0 else "R❌")
            g_a = "A✅" if gh == gt else ("A⚠" if gh > 0 else "A❌")
            b_a = "A✅" if bh == bt else ("A⚠" if bh > 0 else "A❌")
            g_cell = f"{g_r} {g_a} {gh}/{gt}"
            b_cell = f"{b_r} {b_a} {bh}/{bt}"
            rows.append(
                f"  {qid} │ {cat[:26]:<26}{tag:<7} │ {g_cell:<28} │ {b_cell:<28} │ {g_ms:>5}ms │ {b_ms:>5}ms"
            )

    g_avg_ms = int(sum(g_latencies) / len(g_latencies)) if g_latencies else 0
    b_avg_ms = int(sum(b_latencies) / len(b_latencies)) if b_latencies else 0
    g_kw_pct = g_kw_hits / g_kw_total * 100 if g_kw_total else 0
    b_kw_pct = b_kw_hits / b_kw_total * 100 if b_kw_total else 0
    mh_g_pct = mh_g_hits / mh_g_total * 100 if mh_g_total else 0
    mh_b_pct = mh_b_hits / mh_b_total * 100 if mh_b_total else 0
    mh_delta = mh_g_pct - mh_b_pct

    sep = "  " + "─" * 106
    header = (
        "  " + "─" * 106 + "\n"
        f"  {'ID':<4} │ {'Category + hop tag':<33} │ {'Graph RAG  (R=retrieval A=answer)':^30} │ {'Baseline RAG':^30} │ {'Graph':^7} │ {'Base':^7}\n"
        + "  " + "─" * 106
    )
    g_ret_pct = g_ret_hits / g_ret_total * 100 if g_ret_total else 0
    b_ret_pct = b_ret_hits / b_ret_total * 100 if b_ret_total else 0
    footer = (
        sep + "\n"
        f"  {'':4} │ {'TOTALS (non-block queries)':<33} │ "
        f"R:{g_ret_hits}/{g_ret_total}({g_ret_pct:.0f}%) A:{g_kw_hits}/{g_kw_total}({g_kw_pct:.0f}%)    │ "
        f"R:{b_ret_hits}/{b_ret_total}({b_ret_pct:.0f}%) A:{b_kw_hits}/{b_kw_total}({b_kw_pct:.0f}%)    │ "
        f"avg {g_avg_ms}ms │ avg {b_avg_ms}ms\n"
        + sep
    )

    lines = [
        "",
        "=" * 60,
        "  Fab SOP RAG — Baseline vs Graph RAG Evaluation Report",
        "=" * 60,
        "",
        header,
        *rows,
        footer,
        "",
        "  Multi-hop queries (q02 step-chain, q05 cross-doc, q07 dependency chain)",
        f"    Graph RAG  : {mh_g_hits}/{mh_g_total} keywords ({mh_g_pct:.1f}%)",
        f"    Baseline   : {mh_b_hits}/{mh_b_total} keywords ({mh_b_pct:.1f}%)",
        f"    Improvement: +{mh_delta:.1f} percentage points  ← graph traversal advantage",
        "",
        "  Off-topic block rate",
        f"    Graph RAG  : {g_block_hits}/{block_total} correctly blocked",
        f"    Baseline   : {b_block_hits}/{block_total} correctly blocked",
        "",
        "  Overall scores (non-block queries)",
        f"    Graph RAG  : Retrieval {g_ret_hits}/{g_ret_total} ({g_ret_pct:.1f}%)  →  Answer {g_kw_hits}/{g_kw_total} ({g_kw_pct:.1f}%)",
        f"    Baseline   : Retrieval {b_ret_hits}/{b_ret_total} ({b_ret_pct:.1f}%)  →  Answer {b_kw_hits}/{b_kw_total} ({b_kw_pct:.1f}%)",
        "",
        "  Latency",
        f"    Graph RAG  : avg {g_avg_ms} ms  (includes graph traversal overhead)",
        f"    Baseline   : avg {b_avg_ms} ms  (vector retrieval only)",
        f"    Overhead   : +{g_avg_ms - b_avg_ms} ms for graph expansion",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RAG vs Graph RAG evaluation")
    parser.add_argument("--mock", action="store_true", help="Use pre-computed mock results (no services needed)")
    parser.add_argument("--output", type=str, default="", help="Save JSON results to this path")
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    if args.mock:
        print("\n[INFO] Running in MOCK mode — using pre-computed results (no services required)")
        results = {qid: {"graph": d["graph"], "baseline": d["baseline"]} for qid, d in _MOCK_RESULTS.items()}
    else:
        print("\n[INFO] Running LIVE — connecting to Neo4j / vLLM / Chroma...")
        raw = _run_live(queries)
        results = {r["id"]: {"graph": r["graph"], "baseline": r["baseline"]} for r in raw}

    report = render_report(queries, results)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "queries": queries,
            "results": results,
            "mode": "mock" if args.mock else "live",
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()