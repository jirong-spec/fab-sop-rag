"""
Graph RAG Evaluation

Runs all queries from data/sample_queries/fab_queries.json through the
graph RAG pipeline and reports:

  • R (Retrieval) — expected keywords in model_triples
  • A (Answer)    — expected keywords in answer
  • Correct-block rate — off-topic queries correctly rejected
  • Latency (ms)

Usage
-----
  # Live mode (requires Neo4j + vLLM + Chroma running):
  docker compose exec -T api python scripts/eval_compare.py

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

def _run_live(queries: list[dict]) -> list[dict]:
    from app.schemas import AskRequest
    from app.services.pipeline import run_pipeline

    print("[INFO] Pre-warming services...")
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
                entry = {
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
                print(f"[WARNING] Timed out: {qid}")
                entry = {
                    "status": "error", "reasoning_type": "timeout",
                    "latency_ms": g_ms, "answer": f"[TIMEOUT]",
                    "evidence_triples": [], "model_triples": [], "entities": [],
                }

            results.append({"id": qid, "graph": entry})
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def render_report(queries: list[dict], results: dict) -> str:
    rows = []
    kw_hits = kw_total = ret_hits = ret_total = 0
    block_hits = block_total = 0
    latencies = []
    mh_hits = mh_total = 0

    for q in queries:
        qid = q["id"]
        resp = results.get(qid, {}).get("graph", {})
        score = score_response(resp, q)
        ms = resp.get("latency_ms", 0)
        latencies.append(ms)
        tag = " [↑HOP]" if qid in MULTIHOP_IDS else ""

        if score["is_block_query"]:
            block_total += 1
            ok = "✅" if score["correct_block"] else "❌"
            if score["correct_block"]:
                block_hits += 1
            rows.append(f"  {qid} │ {q['category'][:30]:<30}{tag:<7} │ {ok} blocked          │ {ms:>5}ms")
        else:
            gh, gt = score["keyword_hits"], score["keyword_total"]
            grh = score["retrieval_hits"]
            kw_hits += gh; kw_total += gt
            ret_hits += grh; ret_total += gt
            if qid in MULTIHOP_IDS:
                mh_hits += gh; mh_total += gt
            r = "R✅" if grh == gt else ("R⚠" if grh > 0 else "R❌")
            a = "A✅" if gh == gt else ("A⚠" if gh > 0 else "A❌")
            rows.append(f"  {qid} │ {q['category'][:30]:<30}{tag:<7} │ {r} {a} {gh}/{gt:<2}          │ {ms:>5}ms")

    avg_ms = int(sum(latencies) / len(latencies)) if latencies else 0
    kw_pct = kw_hits / kw_total * 100 if kw_total else 0
    ret_pct = ret_hits / ret_total * 100 if ret_total else 0
    mh_pct = mh_hits / mh_total * 100 if mh_total else 0

    sep = "  " + "─" * 72
    lines = [
        "", "=" * 50,
        "  Fab SOP RAG — Evaluation Report",
        "=" * 50, "",
        sep,
        f"  {'ID':<4} │ {'Category':<37} │ {'R / A':^18} │ {'Latency':^7}",
        sep,
        *rows,
        sep,
        f"  TOTALS │ R:{ret_hits}/{ret_total}({ret_pct:.0f}%)  A:{kw_hits}/{kw_total}({kw_pct:.0f}%)  block:{block_hits}/{block_total}  avg {avg_ms}ms",
        sep, "",
        f"  Multi-hop (q02/q05/q07): {mh_hits}/{mh_total} ({mh_pct:.1f}%)",
        "", "=" * 50,
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
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    raw = _run_live(queries)
    results = {r["id"]: {"graph": r["graph"]} for r in raw}

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
