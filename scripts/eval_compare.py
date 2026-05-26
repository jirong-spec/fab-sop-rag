"""
Graph RAG vs Vector RAG Evaluation

Runs all queries through both pipelines and reports:

  • R (Retrieval) — expected keywords in model_triples   [Graph RAG only]
  • A (Answer)    — expected keywords in answer
  • Correct-block rate — off-topic queries correctly rejected
  • Latency (ms) — side-by-side comparison

Usage
-----
  # Live mode (requires Neo4j + vLLM + Chroma running):
  docker compose exec -T api python scripts/eval_compare.py

  # Graph RAG only (original behaviour):
  docker compose exec -T api python scripts/eval_compare.py --graph-only

  # Save JSON results:
  docker compose exec -T api python scripts/eval_compare.py --output data/eval_results/latest.json

  # Log to MLflow:
  docker compose exec -T api python scripts/eval_compare.py --mlflow-uri http://mlflow:5000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

QUERY_TIMEOUT_SEC = 120

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "data" / "sample_queries" / "fab_queries.json"
MULTIHOP_IDS = {"q02", "q05", "q07"}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(resp: dict, expected: dict) -> dict:
    behavior = expected.get("expected_behavior", "retrieved_and_answered")
    keywords = expected.get("expected_keywords", [])

    if behavior == "blocked_by_topic_guard":
        return {
            "correct_block": resp.get("status") == "blocked",
            "keyword_hits": 0, "keyword_total": 0,
            "retrieval_hits": 0, "retrieval_total": 0,
            "is_block_query": True,
        }

    model_triples = resp.get("model_triples") or resp.get("evidence_triples", [])
    evidence_text = " ".join(model_triples)
    retrieval_hits = sum(1 for kw in keywords if kw.lower() in evidence_text.lower())
    haystack = resp.get("answer", "")
    hits = sum(1 for kw in keywords if kw.lower() in haystack.lower())
    return {
        "correct_block": None,
        "keyword_hits": hits, "keyword_total": len(keywords),
        "retrieval_hits": retrieval_hits, "retrieval_total": len(keywords),
        "is_block_query": False,
    }


# ── Live runner ───────────────────────────────────────────────────────────────

def _run_live(queries: list[dict], run_vector: bool = True) -> list[dict]:
    from app.schemas import AskRequest
    from app.services.pipeline import run_pipeline
    from app.services.vector_pipeline import run_vector_pipeline

    print("[INFO] Pre-warming services...")
    try:
        from app.services.vector_store import _get_vector_store, _get_embeddings, _get_reranker_embeddings
        _get_embeddings().embed_query("warmup")
        _get_reranker_embeddings().embed_query("warmup")
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

            # ── Graph RAG ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            fut = pool.submit(run_pipeline, req)
            try:
                gr = fut.result(timeout=QUERY_TIMEOUT_SEC)
                g_ms = int((time.perf_counter() - t0) * 1000)
                graph_entry = {
                    "status": gr.status, "reasoning_type": gr.reasoning_type,
                    "latency_ms": g_ms, "answer": gr.answer,
                    "evidence_triples": gr.evidence_triples,
                    "model_triples": gr.model_triples, "entities": gr.entities,
                }
            except FuturesTimeoutError:
                print(f"[WARNING] Graph RAG timed out: {qid}")
                graph_entry = {
                    "status": "error", "reasoning_type": "timeout",
                    "latency_ms": QUERY_TIMEOUT_SEC * 1000, "answer": "[TIMEOUT]",
                    "evidence_triples": [], "model_triples": [], "entities": [],
                }

            # ── Vector RAG ─────────────────────────────────────────────────
            vector_entry = None
            if run_vector:
                t0 = time.perf_counter()
                fut = pool.submit(run_vector_pipeline, req)
                try:
                    vr, sl = fut.result(timeout=QUERY_TIMEOUT_SEC)
                    v_ms = int((time.perf_counter() - t0) * 1000)
                    vector_entry = {
                        "status": vr.status, "reasoning_type": vr.reasoning_type,
                        "latency_ms": v_ms, "answer": vr.answer,
                        "model_triples": vr.model_triples,
                        "stage_latencies": sl,
                    }
                except FuturesTimeoutError:
                    print(f"[WARNING] Vector RAG timed out: {qid}")
                    vector_entry = {
                        "status": "error", "reasoning_type": "timeout",
                        "latency_ms": QUERY_TIMEOUT_SEC * 1000, "answer": "[TIMEOUT]",
                        "model_triples": [], "stage_latencies": {},
                    }

            results.append({"id": qid, "graph": graph_entry, "vector": vector_entry})
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def render_report(queries: list[dict], results: dict) -> str:
    has_vector = any(results.get(q["id"], {}).get("vector") is not None for q in queries)

    rows = []
    g_kw_hits = g_kw_total = g_ret_hits = g_ret_total = 0
    v_kw_hits = v_kw_total = 0
    block_hits = block_total = 0
    g_latencies = []
    v_latencies = []
    mh_hits = mh_total = 0

    for q in queries:
        qid = q["id"]
        g_resp = results.get(qid, {}).get("graph", {})
        v_resp = results.get(qid, {}).get("vector") or {}
        g_score = score_response(g_resp, q)
        v_score = score_response(v_resp, q) if v_resp else None
        g_ms = g_resp.get("latency_ms", 0)
        v_ms = v_resp.get("latency_ms", 0) if v_resp else 0
        g_latencies.append(g_ms)
        if v_resp:
            v_latencies.append(v_ms)
        tag = " [↑HOP]" if qid in MULTIHOP_IDS else ""

        if g_score["is_block_query"]:
            block_total += 1
            ok = "✅" if g_score["correct_block"] else "❌"
            if g_score["correct_block"]:
                block_hits += 1
            v_col = f"{v_ms:>5}ms" if v_resp else "  n/a "
            rows.append(f"  {qid} │ {q['category'][:28]:<28}{tag:<7} │ {ok} blocked   {g_ms:>5}ms │ {ok} blocked   {v_col}")
        else:
            gh, gt = g_score["keyword_hits"], g_score["keyword_total"]
            grh = g_score["retrieval_hits"]
            g_kw_hits += gh; g_kw_total += gt
            g_ret_hits += grh; g_ret_total += gt
            if qid in MULTIHOP_IDS:
                mh_hits += gh; mh_total += gt
            r = "R✅" if grh == gt else ("R⚠" if grh > 0 else "R❌")
            ga = "A✅" if gh == gt else ("A⚠" if gh > 0 else "A❌")
            g_col = f"{r} {ga} {gh}/{gt}  {g_ms:>5}ms"

            if v_score and v_resp:
                vh, vt = v_score["keyword_hits"], v_score["keyword_total"]
                v_kw_hits += vh; v_kw_total += vt
                va = "A✅" if vh == vt else ("A⚠" if vh > 0 else "A❌")
                delta = g_ms - v_ms
                sign = "+" if delta > 0 else ""
                v_col = f"     {va} {vh}/{vt}  {v_ms:>5}ms  Δ{sign}{delta}ms"
            else:
                v_col = "  n/a"

            rows.append(f"  {qid} │ {q['category'][:28]:<28}{tag:<7} │ {g_col} │ {v_col}")

    g_avg = int(sum(g_latencies) / len(g_latencies)) if g_latencies else 0
    v_avg = int(sum(v_latencies) / len(v_latencies)) if v_latencies else 0
    g_kw_pct = g_kw_hits / g_kw_total * 100 if g_kw_total else 0
    g_ret_pct = g_ret_hits / g_ret_total * 100 if g_ret_total else 0
    v_kw_pct = v_kw_hits / v_kw_total * 100 if v_kw_total else 0
    mh_pct = mh_hits / mh_total * 100 if mh_total else 0
    delta_avg = g_avg - v_avg

    sep = "  " + "─" * 88
    hdr_g = f"{'Graph RAG  R/A  Latency':^28}"
    hdr_v = f"{'Vector RAG  A  Latency  Δ':^32}" if has_vector else ""
    lines = [
        "", "=" * 60,
        "  Fab SOP RAG — Graph RAG vs Vector RAG Comparison",
        "=" * 60, "",
        sep,
        f"  {'ID':<4} │ {'Category':<35} │ {hdr_g} │ {hdr_v}",
        sep,
        *rows,
        sep,
        f"  Graph RAG  │ R:{g_ret_hits}/{g_ret_total}({g_ret_pct:.0f}%)  A:{g_kw_hits}/{g_kw_total}({g_kw_pct:.0f}%)  block:{block_hits}/{block_total}  avg {g_avg}ms",
    ]
    if has_vector:
        lines.append(
            f"  Vector RAG │ A:{v_kw_hits}/{v_kw_total}({v_kw_pct:.0f}%)  avg {v_avg}ms  │  Graph overhead: Δ+{delta_avg}ms"
        )
    lines += [
        sep, "",
        f"  Multi-hop (q02/q05/q07): {mh_hits}/{mh_total} ({mh_pct:.1f}%)  [Graph RAG]",
        "", "=" * 60,
    ]
    return "\n".join(lines)


# ── MLflow ────────────────────────────────────────────────────────────────────

def _log_to_mlflow(tracking_uri: str, queries: list[dict], results: dict) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fab-sop-eval")

    params = {"reranker": "bi-encoder-cosine", "cap_strategy": "dynamic_top50pct", "cot_method": "implicit"}
    try:
        from app.services.answer_service import _PROMPT_TEMPLATE
        params["cot_method"] = "implicit" if "心中逐一核對" in _PROMPT_TEMPLATE else "none"
    except Exception:
        pass

    ret_hits = ret_total = kw_hits = kw_total = 0
    latencies = []
    per_q: dict = {}
    for q in queries:
        qid = q["id"]
        resp = results.get(qid, {}).get("graph", {})
        score = score_response(resp, q)
        ms = resp.get("latency_ms", 0)
        latencies.append(ms)
        if not score["is_block_query"]:
            kw_hits += score["keyword_hits"]; kw_total += score["keyword_total"]
            ret_hits += score["retrieval_hits"]; ret_total += score["retrieval_total"]
            per_q[f"{qid}_answer"] = score["keyword_hits"] / score["keyword_total"] if score["keyword_total"] else 0
            per_q[f"{qid}_latency_ms"] = ms

    metrics = {
        "retrieval_rate": ret_hits / ret_total if ret_total else 0,
        "answer_rate": kw_hits / kw_total if kw_total else 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        **per_q,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"queries": queries, "results": results}, f, ensure_ascii=False, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="eval_results")
        os.unlink(tmp)

    print(f"[INFO] MLflow logged: answer={metrics['answer_rate']:.1%} latency={metrics['avg_latency_ms']:.0f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--mlflow-uri", type=str, default="")
    parser.add_argument("--graph-only", action="store_true", help="Skip Vector RAG baseline")
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    raw = _run_live(queries, run_vector=not args.graph_only)
    results = {r["id"]: {"graph": r["graph"], "vector": r.get("vector")} for r in raw}

    print(render_report(queries, results))

    if args.mlflow_uri and _MLFLOW_AVAILABLE:
        _log_to_mlflow(args.mlflow_uri, queries, results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"queries": queries, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved to {out}")


if __name__ == "__main__":
    main()
