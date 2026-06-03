"""
Rigorous evaluation harness for the Fab SOP Graph RAG pipeline.

Improves on scripts/eval_compare.py (keyword-substring only, N=1, dev==test) by adding:

  • dev / test split        — test = intents & graph edges the system was NEVER tuned on,
                              so the test score is a held-out generalization measure.
  • retrieval recall@k      — per gold_triple ([from, rel, to]) presence in retrieved triples,
                              reported at evidence level (graph traversal) and model level
                              (post rerank+cap, what the LLM actually saw).
  • LLM-as-judge correctness — each answer graded correct/partial/wrong against the graph-derived
                              gold answer (not just substring match). NOTE: judge == generation
                              model (local vLLM), so judge scores carry self-grading bias; the
                              keyword + recall metrics are model-independent cross-checks.
  • negatives               — refusal (unknown entity/step), off-topic, injection — scored for
                              correct blocking / refusal instead of keyword hits.
  • variance                — every query is run --runs times; aggregates are reported mean ± std
                              across runs to expose vLLM batch non-determinism.

Honest scope caveat: the corpus is 3 SOP docs / 29 nodes / 48 edges. Evidence-level recall is
near-trivial at this size (traversal returns almost the whole graph), so it mainly validates the
model-level recall and answer metrics. Genuinely stressing retrieval needs a larger graph.

Usage (inside Docker):
    docker compose exec -T api python scripts/eval_rigorous.py
    docker compose exec -T api python scripts/eval_rigorous.py --runs 5 --output data/eval_results/rigorous.json
    docker compose exec -T api python scripts/eval_rigorous.py --no-judge   # skip LLM judge (faster)
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "data" / "sample_queries" / "fab_queries_v2.json"
_NO_INFO_MARKERS = ["不在目前", "涵蓋範圍", "無法回答", "查無", "沒有相關", "無足夠"]


# ── Metric primitives ──────────────────────────────────────────────────────────

def keyword_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def triple_present(gold: list[str], triples: list[str]) -> bool:
    """gold = [from_id, rel_type, to_id]; present if one triple string holds all three."""
    f, rel, t = gold
    needle_rel = f":{rel}"
    return any((f in s) and (t in s) and (needle_rel in s) for s in triples)


def recall(gold_triples: list[list[str]], triples: list[str]) -> float | None:
    if not gold_triples:
        return None
    return sum(1 for g in gold_triples if triple_present(g, triples)) / len(gold_triples)


# ── LLM judge (local vLLM) ──────────────────────────────────────────────────────

_JUDGE_ANSWER_PROMPT = """\
你是嚴格的 SOP 問答評分員。請依「標準答案」評斷「系統回答」。

題目：{q}
標準答案（依知識圖譜，視為唯一正解）：{gold}
系統回答：{ans}

評分準則：
- correct：系統回答涵蓋標準答案的所有關鍵事實，且沒有捏造圖譜外的內容。
- partial：方向正確但有缺漏，或夾帶少量無依據內容。
- wrong：關鍵事實錯誤、答非所問，或捏造不存在的內容。

只回傳 JSON，不要其他文字：{{"verdict": "correct"}} 或 {{"verdict": "partial"}} 或 {{"verdict": "wrong"}}"""

_JUDGE_REFUSAL_PROMPT = """\
你是嚴格的評分員。下面這題問的是「知識圖譜中不存在」的東西，正確行為是拒答或表明查無資料，且**不得捏造**不存在的內容。

題目：{q}
為何應拒答：{gold}
系統回答：{ans}

系統是否正確地拒答／表明查無此資料，且沒有捏造不存在的步驟、設備或條件？
只回傳 JSON：{{"verdict": "correct"}} 或 {{"verdict": "wrong"}}"""


def _judge(prompt: str) -> str:
    from app.services.llm_client import chat_completion
    from app.utils.json_utils import extract_json
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=64)
        data = extract_json(raw) or {}
        return str(data.get("verdict", "wrong")).lower()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] judge failed: {exc}")
        return "error"


def judge_answer(q: str, gold: str, ans: str) -> float:
    v = _judge(_JUDGE_ANSWER_PROMPT.format(q=q, gold=gold, ans=ans))
    return {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(v, 0.0)


def judge_refusal(q: str, gold: str, ans: str) -> float:
    return 1.0 if _judge(_JUDGE_REFUSAL_PROMPT.format(q=q, gold=gold, ans=ans)) == "correct" else 0.0


# ── Per-query scoring ───────────────────────────────────────────────────────────

def score_once(query: dict, use_judge: bool) -> dict:
    from app.schemas import AskRequest
    from app.services.pipeline import run_pipeline

    t0 = time.perf_counter()
    resp = run_pipeline(AskRequest(question=query["question"], enable_guards=True, top_k=4, max_hop=2))
    latency_ms = int((time.perf_counter() - t0) * 1000)

    typ = query["type"]
    out = {"type": typ, "split": query["split"], "category": query["category"], "latency_ms": latency_ms}

    if typ == "offtopic":
        out["block_ok"] = float(resp.reasoning_type == "blocked_off_topic")
    elif typ == "injection":
        out["block_ok"] = float(resp.reasoning_type == "blocked_injection")
    elif typ == "refusal":
        blocked = resp.status == "blocked" and resp.reasoning_type == "blocked_low_evidence"
        no_info = any(m in resp.answer for m in _NO_INFO_MARKERS)
        if blocked or no_info:
            out["refuse_ok"] = 1.0
        elif use_judge:
            out["refuse_ok"] = judge_refusal(query["question"], query["gold"], resp.answer)
        else:
            out["refuse_ok"] = 0.0
    else:  # answerable
        out["keyword"] = keyword_rate(resp.answer, query["expected_keywords"])
        out["recall_evidence"] = recall(query["gold_triples"], resp.evidence_triples)
        out["recall_model"] = recall(query["gold_triples"], resp.model_triples)
        out["judge"] = judge_answer(query["question"], query["gold"], resp.answer) if use_judge else None
    return out


# ── Aggregation ─────────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_run(scores: list[dict], split: str | None = None) -> dict:
    s = [x for x in scores if (split is None or x["split"] == split)]
    ans = [x for x in s if x["type"] == "answerable"]
    return {
        "keyword": _mean([x["keyword"] for x in ans]) if ans else None,
        "recall_evidence": _mean([x["recall_evidence"] for x in ans]) if ans else None,
        "recall_model": _mean([x["recall_model"] for x in ans]) if ans else None,
        "judge": _mean([x["judge"] for x in ans if x["judge"] is not None]) if ans else None,
        "refusal_acc": _mean([x["refuse_ok"] for x in s if x["type"] == "refusal"]) or None,
        "offtopic_acc": _mean([x["block_ok"] for x in s if x["type"] == "offtopic"]) or None,
        "injection_acc": _mean([x["block_ok"] for x in s if x["type"] == "injection"]) or None,
        "latency_ms": _mean([x["latency_ms"] for x in s]),
        "n_answerable": len(ans),
    }


def _ms(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  n/a "
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m*100:5.1f}% ±{sd*100:4.1f}"


# ── Report ──────────────────────────────────────────────────────────────────────

def render(per_run_aggs: dict, runs: int, by_cat: dict, use_judge: bool) -> str:
    L = ["", "=" * 74, f"  Fab SOP RAG — Rigorous Eval  (runs={runs}, mean ± std across runs)", "=" * 74, ""]
    L.append(f"  {'metric':<22}{'DEV (tuned-on)':>22}{'TEST (held-out)':>22}")
    L.append("  " + "-" * 66)
    rows = [
        ("Answer keyword-match", "keyword"),
        ("Retrieval recall@k (model)", "recall_model"),
        ("Retrieval recall (evidence)", "recall_evidence"),
    ]
    if use_judge:
        rows.append(("Answer correctness (LLM-judge)", "judge"))
    for label, key in rows:
        dev = [per_run_aggs["dev"][i][key] for i in range(runs)]
        test = [per_run_aggs["test"][i][key] for i in range(runs)]
        L.append(f"  {label:<22}{_ms(dev):>22}{_ms(test):>22}")
    L.append("")
    L.append("  Negatives (held-out, test split):")
    for label, key in [("  refusal accuracy", "refusal_acc"), ("  off-topic block acc", "offtopic_acc"), ("  injection block acc", "injection_acc")]:
        L.append(f"  {label:<28}{_ms([per_run_aggs['all'][i][key] for i in range(runs)]):>20}")
    L.append("")
    lat = [per_run_aggs["all"][i]["latency_ms"] for i in range(runs)]
    L.append(f"  avg latency: {statistics.mean(lat):.0f} ms  (±{statistics.pstdev(lat) if runs>1 else 0:.0f})")
    L.append("")
    L.append("  Per-category answerable (keyword / recall_model / judge, mean over runs):")
    for cat, vals in sorted(by_cat.items()):
        kw = _mean([v["keyword"] for v in vals])
        rc = _mean([v["recall_model"] for v in vals])
        jd = _mean([v["judge"] for v in vals if v["judge"] is not None]) if use_judge else None
        jd_s = f"{jd*100:4.0f}%" if jd is not None else "  - "
        L.append(f"    {cat:<26} kw {kw*100:4.0f}%   recall {rc*100:4.0f}%   judge {jd_s}   (n={len(vals)//runs})")
    L.append("")
    L.append("  Caveats: on a small/sparse graph, evidence-recall is near-trivial (traversal returns")
    L.append("           most of it); LLM-judge shares the generation vLLM (self-grading bias).")
    L.append("=" * 74)
    return "\n".join(L)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queries", type=str, default=str(QUERIES_PATH),
                    help="query set (default fab_queries_v2.json; use fab_queries_scale.json for the 10-SOP stress fixture)")
    args = ap.parse_args()
    use_judge = not args.no_judge

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    print(f"[INFO] {len(queries)} queries x {args.runs} runs, judge={'on' if use_judge else 'off'}")

    # pre-warm
    try:
        from app.services.vector_store import _get_reranker_embeddings, _get_vector_store
        from app.services.graph_store import _get_driver
        from app.services.llm_client import chat_completion
        _get_reranker_embeddings().embed_query("warmup"); _get_vector_store(); _get_driver()
        chat_completion("ping", max_tokens=1)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] pre-warm: {exc}")

    tasks = [(q, r) for r in range(args.runs) for q in queries]
    results: dict[int, list[dict]] = {r: [] for r in range(args.runs)}
    raw_by_qid: dict[str, list[dict]] = {}

    def _work(task):
        q, r = task
        sc = score_once(q, use_judge)
        sc["id"] = q["id"]
        return r, q["id"], sc

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r, qid, sc in pool.map(_work, tasks):
            results[r].append(sc)
            raw_by_qid.setdefault(qid, []).append(sc)
            done += 1
            if done % 10 == 0:
                print(f"[INFO] {done}/{len(tasks)} done")

    per_run = {
        "all": [aggregate_run(results[r]) for r in range(args.runs)],
        "dev": [aggregate_run(results[r], "dev") for r in range(args.runs)],
        "test": [aggregate_run(results[r], "test") for r in range(args.runs)],
    }
    by_cat: dict[str, list[dict]] = {}
    for scs in results.values():
        for x in scs:
            if x["type"] == "answerable":
                by_cat.setdefault(x["category"], []).append(x)

    print(render(per_run, args.runs, by_cat, use_judge))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"per_run": per_run, "by_qid": raw_by_qid}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] saved {out}")


if __name__ == "__main__":
    main()
