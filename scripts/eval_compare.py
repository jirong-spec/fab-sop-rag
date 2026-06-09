"""
Graph RAG vs Vector RAG Evaluation

Runs all queries through both pipelines and reports:

  • R (Retrieval) — expected keywords in model_triples   [Graph RAG only]
  • A (Answer)    — expected keywords in answer
  • Correct-block rate — off-topic queries correctly rejected
  • Latency (ms) — side-by-side comparison

Usage
-----
  # Live mode (requires Neo4j + vLLM + Qdrant running):
  docker compose exec -T api python scripts/eval_compare.py

  # Graph RAG only (original behaviour):
  docker compose exec -T api python scripts/eval_compare.py --graph-only

  # Save JSON results:
  docker compose exec -T api python scripts/eval_compare.py --output data/eval_results/latest.json
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

QUERY_TIMEOUT_SEC = 120

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY_DIR = ROOT / "data" / "sample_queries"
QUERIES_PATHS = [QUERY_DIR / "fab_queries_dev.json", QUERY_DIR / "fab_queries_test.json"]
# Multi-hop questions are identified by their data label, not hardcoded IDs: the old
# {"q02","q05","q07"} set referred to a retired query file and silently matched nothing
# after the dev/test split. Driving off `category` keeps this correct as the set evolves.
MULTIHOP_CATEGORY = "multihop_dependency"

# Question-type buckets for the --by-structure report. The Graph-vs-Vector gap
# concentrates in questions whose answer must be assembled ACROSS graph edges
# (ordering, dependency chains, cross-document references) — no single retrieved
# chunk holds it — versus single-fact lookups where the answer is one contiguous
# span. Multi-hop alone is too small a subset (n=2) to headline; these buckets show
# where the graph advantage actually lives. Driven by `category` so a new question
# type that fits neither bucket is reported as uncategorized rather than silently dropped.
STRUCTURED_CATEGORIES = {
    "sop_step_sequence",
    "pump_check_sequence",
    "step_dependency",
    "multihop_dependency",
    "cross_doc_dependency",
    "defined_in",
    "interlock_condition",
}
LOOKUP_CATEGORIES = {
    "equipment_precondition",
    "step_requires_status",
    "anomaly_handling",
}


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_response(resp: dict, expected: dict) -> dict:
    behavior = expected.get("expected_behavior", "retrieved_and_answered")
    keywords = expected.get("expected_keywords", [])

    if behavior == "blocked_by_topic_guard":
        return {
            "correct_block": resp.get("status") == "blocked",
            "keyword_hits": 0,
            "keyword_total": 0,
            "retrieval_hits": 0,
            "retrieval_total": 0,
            "is_block_query": True,
        }

    model_triples = resp.get("model_triples") or resp.get("evidence_triples", [])
    evidence_text = " ".join(model_triples)
    retrieval_hits = sum(1 for kw in keywords if kw.lower() in evidence_text.lower())
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


# ── Live runner ───────────────────────────────────────────────────────────────


def _run_live(queries: list[dict], run_vector: bool = True) -> list[dict]:
    from app.schemas import AskRequest
    from app.services.pipeline import run_pipeline
    from app.services.vector_pipeline import run_vector_pipeline

    print("[INFO] Pre-warming services...")
    try:
        from app.services.vector_store import _get_embeddings, _get_reranker_embeddings, _get_vector_store

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

    # Timing wrappers measure elapsed time inside the thread, so latency
    # reflects actual execution and not time spent waiting for the sibling future.
    def _timed_graph(r):
        t = time.perf_counter()
        result = run_pipeline(r)
        return result, int((time.perf_counter() - t) * 1000)

    def _timed_vector(r):
        t = time.perf_counter()
        result, sl = run_vector_pipeline(r)
        return result, sl, int((time.perf_counter() - t) * 1000)

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for q in queries:
            req = AskRequest(question=q["question"], enable_guards=True, top_k=4, max_hop=2)
            qid = q["id"]

            # Submit both pipelines in parallel
            g_fut = pool.submit(_timed_graph, req)
            v_fut = pool.submit(_timed_vector, req) if run_vector else None

            # ── Graph RAG ──────────────────────────────────────────────────
            try:
                gr, g_ms = g_fut.result(timeout=QUERY_TIMEOUT_SEC)
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
                print(f"[WARNING] Graph RAG timed out: {qid}")
                graph_entry = {
                    "status": "error",
                    "reasoning_type": "timeout",
                    "latency_ms": QUERY_TIMEOUT_SEC * 1000,
                    "answer": "[TIMEOUT]",
                    "evidence_triples": [],
                    "model_triples": [],
                    "entities": [],
                }
            except Exception as exc:
                print(f"[WARNING] Graph RAG error on {qid}: {exc}")
                graph_entry = {
                    "status": "error",
                    "reasoning_type": "error",
                    "latency_ms": 0,
                    "answer": f"[ERROR] {exc}",
                    "evidence_triples": [],
                    "model_triples": [],
                    "entities": [],
                }

            # ── Vector RAG ─────────────────────────────────────────────────
            vector_entry = None
            if v_fut is not None:
                try:
                    vr, sl, v_ms = v_fut.result(timeout=QUERY_TIMEOUT_SEC)
                    vector_entry = {
                        "status": vr.status,
                        "reasoning_type": vr.reasoning_type,
                        "latency_ms": v_ms,
                        "answer": vr.answer,
                        "model_triples": vr.model_triples,
                        "evidence_triples": vr.evidence_triples,
                        "stage_latencies": sl,
                    }
                except FuturesTimeoutError:
                    print(f"[WARNING] Vector RAG timed out: {qid}")
                    vector_entry = {
                        "status": "error",
                        "reasoning_type": "timeout",
                        "latency_ms": QUERY_TIMEOUT_SEC * 1000,
                        "answer": "[TIMEOUT]",
                        "model_triples": [],
                        "evidence_triples": [],
                        "stage_latencies": {},
                    }
                except Exception as exc:
                    print(f"[WARNING] Vector RAG error on {qid}: {exc}")
                    vector_entry = {
                        "status": "error",
                        "reasoning_type": "error",
                        "latency_ms": 0,
                        "answer": f"[ERROR] {exc}",
                        "model_triples": [],
                        "evidence_triples": [],
                        "stage_latencies": {},
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
        tag = " [↑HOP]" if q.get("category") == MULTIHOP_CATEGORY else ""

        if g_score["is_block_query"]:
            block_total += 1
            ok = "✅" if g_score["correct_block"] else "❌"
            if g_score["correct_block"]:
                block_hits += 1
            if v_resp and v_score:
                v_ok = "✅" if v_score.get("correct_block") else "❌"
                v_col = f"{v_ok} blocked   {v_ms:>5}ms"
            else:
                v_col = "  n/a "
            rows.append(f"  {qid} │ {q['category'][:28]:<28}{tag:<7} │ {ok} blocked   {g_ms:>5}ms │ {v_col}")
        else:
            gh, gt = g_score["keyword_hits"], g_score["keyword_total"]
            grh = g_score["retrieval_hits"]
            g_kw_hits += gh
            g_kw_total += gt
            g_ret_hits += grh
            g_ret_total += gt
            if q.get("category") == MULTIHOP_CATEGORY:
                mh_hits += gh
                mh_total += gt
            r = "R✅" if grh == gt else ("R⚠" if grh > 0 else "R❌")
            ga = "A✅" if gh == gt else ("A⚠" if gh > 0 else "A❌")
            g_col = f"{r} {ga} {gh}/{gt}  {g_ms:>5}ms"

            if v_score and v_resp:
                vh, vt = v_score["keyword_hits"], v_score["keyword_total"]
                v_kw_hits += vh
                v_kw_total += vt
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
        "",
        "=" * 60,
        "  Fab SOP RAG — Graph RAG vs Vector RAG Comparison",
        "=" * 60,
        "",
        sep,
        f"  {'ID':<4} │ {'Category':<35} │ {hdr_g} │ {hdr_v}",
        sep,
        *rows,
        sep,
        f"  Graph RAG  │ R:{g_ret_hits}/{g_ret_total}({g_ret_pct:.0f}%)  A:{g_kw_hits}/{g_kw_total}({g_kw_pct:.0f}%)  block:{block_hits}/{block_total}  avg {g_avg}ms",
    ]
    if has_vector:
        lines.append(
            f"  Vector RAG │ A:{v_kw_hits}/{v_kw_total}({v_kw_pct:.0f}%)  avg {v_avg}ms  │  Graph overhead: Δ{'+' if delta_avg >= 0 else ''}{delta_avg}ms"
        )
    lines += [
        sep,
        "",
        f"  Multi-hop ({MULTIHOP_CATEGORY}): {mh_hits}/{mh_total} ({mh_pct:.1f}%)  [Graph RAG]",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def render_structure_report(queries: list[dict], results: dict) -> str:
    """Break the Graph-vs-Vector ANSWER gap down by question structure.

    Two buckets, by category:
      STRUCTURED — answer must be assembled across graph edges (ordering, dependency
                   chains, cross-document refs); no single retrieved chunk holds it.
      LOOKUP     — answer is a single contiguous fact attached to one entity.

    Reports keyword-level recall and question-level pass (all keywords present) for
    each bucket, with a per-category breakdown sorted worst-Vector-first. Graph RAG is
    ~100% throughout, so the spread lives in Vector — this shows WHERE the graph
    advantage concentrates without over-indexing on the tiny multi-hop subset.
    """
    from collections import defaultdict

    stat: dict = defaultdict(lambda: {"g_kw": 0, "v_kw": 0, "kw": 0, "g_q": 0, "v_q": 0, "n": 0, "has_v": False})
    uncategorized: dict = defaultdict(int)

    for q in queries:
        g_resp = results.get(q["id"], {}).get("graph", {}) or {}
        gs = score_response(g_resp, q)
        if gs["is_block_query"] or gs["keyword_total"] == 0:
            continue  # guardrail / refusal / injection — no answer keywords to score
        cat = q.get("category", "?")
        if cat not in STRUCTURED_CATEGORIES and cat not in LOOKUP_CATEGORIES:
            uncategorized[cat] += 1
        n = gs["keyword_total"]
        s = stat[cat]
        s["kw"] += n
        s["n"] += 1
        s["g_kw"] += gs["keyword_hits"]
        s["g_q"] += int(gs["keyword_hits"] == n)
        v_resp = results.get(q["id"], {}).get("vector")
        if v_resp:
            vs = score_response(v_resp, q)
            s["has_v"] = True
            s["v_kw"] += vs["keyword_hits"]
            s["v_q"] += int(vs["keyword_hits"] == n)

    def _pct(a: int, b: int) -> str:
        return f"{a / b * 100:.0f}%" if b else "  -"

    lines = ["", "=" * 64, "  Fab SOP RAG — Answer gap by question structure", "=" * 64]
    for title, cats in (
        ("STRUCTURED  (multi-edge: ordering / dependency / cross-doc)", STRUCTURED_CATEGORIES),
        ("LOOKUP  (single contiguous fact)", LOOKUP_CATEGORIES),
    ):
        present = [c for c in cats if c in stat]
        kw = sum(stat[c]["kw"] for c in present)
        if not kw:
            continue
        g_kw = sum(stat[c]["g_kw"] for c in present)
        v_kw = sum(stat[c]["v_kw"] for c in present)
        g_q = sum(stat[c]["g_q"] for c in present)
        v_q = sum(stat[c]["v_q"] for c in present)
        nq = sum(stat[c]["n"] for c in present)
        has_v = any(stat[c]["has_v"] for c in present)
        lines.append("")
        lines.append(f"  {title}    n={nq} questions, {kw} keywords")
        if has_v:
            lines.append(
                f"    keyword recall   Graph {g_kw}/{kw} ({_pct(g_kw, kw)})   "
                f"Vector {v_kw}/{kw} ({_pct(v_kw, kw)})   gap +{(g_kw - v_kw) / kw * 100:.0f}pp"
            )
            lines.append(
                f"    question pass    Graph {g_q}/{nq} ({_pct(g_q, nq)})   "
                f"Vector {v_q}/{nq} ({_pct(v_q, nq)})   gap +{(g_q - v_q) / nq * 100:.0f}pp"
            )
        else:
            lines.append(f"    keyword recall   Graph {g_kw}/{kw} ({_pct(g_kw, kw)})   (Vector baseline not run)")
            lines.append(f"    question pass    Graph {g_q}/{nq} ({_pct(g_q, nq)})")
        for c in sorted(present, key=lambda c: (stat[c]["v_kw"] / stat[c]["kw"]) if stat[c]["has_v"] else 1.0):
            s = stat[c]
            v_part = f"V kw {_pct(s['v_kw'], s['kw']):>4}  q {s['v_q']}/{s['n']}" if s["has_v"] else "V —"
            lines.append(
                f"      {c:24} {s['n']:>2}q   G kw {_pct(s['g_kw'], s['kw']):>4}  q {s['g_q']}/{s['n']}  │  {v_part}"
            )

    if uncategorized:
        lines.append("")
        lines.append("  [WARNING] answerable categories in NO bucket (excluded from the split above):")
        for c, n in sorted(uncategorized.items()):
            lines.append(f"            {c} ({n}q) — add to STRUCTURED_CATEGORIES or LOOKUP_CATEGORIES")
    lines += ["", "=" * 64]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--graph-only", action="store_true", help="Skip Vector RAG baseline")
    parser.add_argument(
        "--by-structure",
        action="store_true",
        help="Also break the answer gap down by question structure (structured vs lookup)",
    )
    args = parser.parse_args()

    queries = [q for p in QUERIES_PATHS for q in json.loads(p.read_text(encoding="utf-8"))]
    raw = _run_live(queries, run_vector=not args.graph_only)
    results = {r["id"]: {"graph": r["graph"], "vector": r.get("vector")} for r in raw}

    print(render_report(queries, results))

    if args.by_structure:
        print(render_structure_report(queries, results))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"queries": queries, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[INFO] Saved to {out}")


if __name__ == "__main__":
    main()
