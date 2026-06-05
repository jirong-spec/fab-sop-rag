"""
Chunking-strategy ablation for the Qdrant vector store.

Question: the production chunker (scripts/ingest_vector.py) splits each SOP doc into
fixed 400-char / 80-overlap windows. Was that a good choice? This harness ablates a
range of strategies and measures which one surfaces the most answer content under a
realistic constraint.

Metric — budget-normalised keyword recall (`r@<budget>ctx`):
    For each answerable question, retrieve the top chunks, concatenate them up to a
    fixed character budget (the LLM's real context constraint), and measure what
    fraction of the question's expected_keywords appear in that retrieved context.

    Plain recall@k is monotonic in chunk size (whole-doc trivially scores 100%), so it
    cannot rank chunkers fairly. Normalising by a fixed *context budget* removes that
    bias: it rewards chunkers that pack the most distinct answer content into a small,
    well-routed context — which is exactly what a token-limited RAG prompt needs.

    Keyword presence is also the right proxy for THIS system specifically: the vector
    store's job here is entity expansion (retrieved chunks are scanned for CamelCase /
    SOP_id tokens), which keys off literal token presence, not paraphrase.

Strategies:
    char-<size>/<overlap>  fixed character windows (the production family)
    paragraph              pack blank-line-separated paragraphs up to ~size
    md-header              one chunk per markdown heading section (## / ###)
    md-cap-<n>             md-header, then size-cap oversized sections (hybrid)
    whole-doc              no chunking (reference / sanity bound)

Rigour — dev/test discipline: choosing a chunk size by looking at the eval set IS
tuning, so it must respect the held-out split. We RANK strategies on the dev split
only (--select-split), pick the winner, then re-score that winner + baseline ONCE on
the held-out test split (--report-split) — that test number is the honest one. The
table prints both columns so you can see whether the dev ranking survives on test.
similarity_search is deterministic, so the only variance is the question sample; we
report bootstrap 95% CIs and a paired (winner − baseline) test-split CI, so "beats the
baseline" is a significance claim, not a point estimate.

Honest scope caveat: the corpus is 3 SOP docs and the answerable split is small
(8 dev / 19 test in fab_queries_v2). Model selection on 8 dev questions is statistically
weak — the dev "winner" among the top cluster is near coin-flip — so the durable claim is
"the 400/80 baseline is poorly placed", not a unique optimum. Inter-document routing is
near-trivial at this corpus size, so this mainly measures *within-document* chunking and
the budget-packing trade-off, and it is a retrieval-recall proxy, not end-to-end answer
quality (the graph, not the vector store, drives answers here).

Runs against a TEMPORARY collection (default "chunk_ablation"); the production
collection (settings.qdrant_collection) is never touched. The temp collection is
deleted on exit.

Usage (qdrant must be up; needs the embedding model, so run in the api image):
    docker compose up -d qdrant
    docker compose run --rm -T --no-deps api python scripts/eval_chunk_ablation.py
    docker compose run --rm -T --no-deps api python scripts/eval_chunk_ablation.py \
        --select-split dev --report-split test --output data/eval_results/chunk_ablation.json
"""

import argparse
import glob
import json
import logging
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DOCS_DIR = ROOT / "data" / "sop_docs"
_DEFAULT_QUERIES = ROOT / "data" / "sample_queries" / "fab_queries_v2.json"


# ── chunkers ────────────────────────────────────────────────────────────────
def char_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows with overlap (the production family)."""
    out, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        out.append(text[start : start + size])
        start += step
    return out


def para_chunk(text: str, size: int = 400) -> list[str]:
    """Greedily pack blank-line-separated paragraphs up to ~size chars."""
    out, cur = [], ""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def md_sections(text: str) -> list[str]:
    """One chunk per markdown heading section (split at lines starting with '#')."""
    out, cur = [], []
    for line in text.split("\n"):
        if line.lstrip().startswith("#") and cur:
            out.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


def md_cap(text: str, cap: int = 250, overlap: int = 50) -> list[str]:
    """md_sections, then size-cap any section longer than `cap` (structure + bound)."""
    out: list[str] = []
    for sec in md_sections(text):
        out += [sec] if len(sec) <= cap else char_chunk(sec, cap, overlap)
    return out


def build_strategies() -> dict:
    """Name -> chunker(text)->list[str]. '*' marks the current production baseline."""
    s = {
        f"char-{n}/{n // 5}": (lambda t, n=n: char_chunk(t, n, n // 5))
        for n in (150, 200, 250, 300, 350, 400, 500, 600)
    }
    s["char-400/80"] = lambda t: char_chunk(t, 400, 80)  # production baseline
    s["paragraph"] = lambda t: para_chunk(t, 400)
    s["md-header"] = md_sections
    s["md-cap-250"] = lambda t: md_cap(t, 250, 50)
    s["whole-doc"] = lambda t: [t]
    return s


# ── metric ──────────────────────────────────────────────────────────────────
def recall_at_budget(hits, keywords: list[str], budget: int) -> float:
    """Fraction of keywords present in the top retrieved chunks packed up to `budget` chars."""
    ctx = ""
    for h in hits:
        if ctx and len(ctx) + len(h.page_content) > budget:
            break
        ctx += h.page_content + " "
    return sum(kw in ctx for kw in keywords) / len(keywords)


def bootstrap_ci(values: list[float], n: int, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (deterministic given seed)."""
    random.seed(seed)
    m = len(values)
    means = sorted(sum(values[random.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# ── data ────────────────────────────────────────────────────────────────────
def load_corpus() -> dict[str, str]:
    return {Path(p).name: Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob(str(_DOCS_DIR / "*.md")))}


def load_questions(path: Path, corpus_text: str) -> list[dict]:
    """Answerable questions (tagged with their dev/test split) whose expected_keywords occur in the corpus."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r.get("type") != "answerable":
            continue
        kw = [k for k in r.get("expected_keywords", []) if k in corpus_text]
        if kw:
            out.append({"question": r["question"], "kw": kw, "split": r.get("split", "?")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate chunking strategies on budget-normalised keyword recall.")
    ap.add_argument("--budget", type=int, default=1500, help="retrieved-context budget in chars")
    ap.add_argument("--bootstrap", type=int, default=3000, help="bootstrap resamples for CIs")
    ap.add_argument("--baseline", type=str, default="char-400/80", help="strategy to compare the winner against")
    ap.add_argument("--select-split", type=str, default="dev", help="split used to RANK and pick the winner")
    ap.add_argument(
        "--report-split", type=str, default="test", help="held-out split the winner+baseline are re-scored on"
    )
    ap.add_argument("--queries", type=str, default=str(_DEFAULT_QUERIES))
    ap.add_argument("--collection", type=str, default="chunk_ablation", help="temporary qdrant collection name")
    ap.add_argument("--output", type=str, default="", help="optional path to dump results JSON")
    args = ap.parse_args()

    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    if args.collection == settings.qdrant_collection:
        raise SystemExit(f"refusing to use the production collection {settings.qdrant_collection!r}")

    corpus = load_corpus()
    questions = load_questions(Path(args.queries), "\n".join(corpus.values()))
    logger.info(
        "corpus: %d docs (avg %d chars) | %d answerable questions by split: %s",
        len(corpus),
        statistics.mean(len(t) for t in corpus.values()),
        len(questions),
        dict(Counter(q["split"] for q in questions)),
    )
    sel_idx = [i for i, q in enumerate(questions) if q["split"] == args.select_split]
    rep_idx = [i for i, q in enumerate(questions) if q["split"] == args.report_split]
    if not sel_idx or not rep_idx:
        raise SystemExit(
            f"need both --select-split {args.select_split!r} and --report-split {args.report_split!r} present"
        )

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model, model_kwargs={"device": "cuda"})
    client = QdrantClient(url=settings.qdrant_url)
    strategies = build_strategies()

    per_q: dict[str, list[float]] = {}  # strategy -> per-question recall at --budget (parallel to `questions`)
    meta: dict[str, dict] = {}
    try:
        for name, chunk_fn in strategies.items():
            docs = [
                Document(page_content=c, metadata={"source": fname})
                for fname, text in corpus.items()
                for c in chunk_fn(text)
            ]
            vs = QdrantVectorStore.from_documents(
                docs,
                embedding=emb,
                url=settings.qdrant_url,
                collection_name=args.collection,
                force_recreate=True,
            )
            per_q[name] = [
                recall_at_budget(vs.similarity_search(q["question"], k=10), q["kw"], args.budget) for q in questions
            ]
            meta[name] = {
                "n_chunks": len(docs),
                "avg_chunk_chars": round(statistics.mean(len(d.page_content) for d in docs), 1),
            }
    finally:
        client.delete_collection(args.collection)
        logger.info("cleaned up temporary collection %r", args.collection)

    def split_mean(name: str, idx: list[int]) -> float:
        return statistics.mean(per_q[name][i] for i in idx)

    def split_ci(name: str, idx: list[int]) -> tuple[float, float]:
        return bootstrap_ci([per_q[name][i] for i in idx], args.bootstrap)

    # ── table: select-split ranking, with the report-split column alongside for transparency ──
    order = sorted(strategies, key=lambda n: -split_mean(n, sel_idx))
    sel, rep = args.select_split, args.report_split
    print(
        f"\n{'strategy':14}{'#ch':>5}{'avg':>6}   {f'{sel}(n={len(sel_idx)}) r@{args.budget}':>22}   {f'{rep}(n={len(rep_idx)}) r@{args.budget}':>22}"
    )
    print("-" * 76)
    for name in order:
        slo, shi = split_ci(name, sel_idx)
        rlo, rhi = split_ci(name, rep_idx)
        mark = " <- baseline" if name == args.baseline else ""
        print(
            f"{name:14}{meta[name]['n_chunks']:5}{meta[name]['avg_chunk_chars']:6.0f}   "
            f"{split_mean(name, sel_idx) * 100:5.1f}% [{slo * 100:4.1f},{shi * 100:4.1f}]   "
            f"{split_mean(name, rep_idx) * 100:5.1f}% [{rlo * 100:4.1f},{rhi * 100:4.1f}]{mark}"
        )

    # ── honest protocol: pick on select-split, then report the held-out result once ──
    winner = order[0]
    base = args.baseline
    print(f"\n── model selection (pick on {sel}, report on held-out {rep}) ──")
    print(
        f"selected on {sel} (n={len(sel_idx)}): winner = {winner}  ({sel} r@{args.budget} = {split_mean(winner, sel_idx) * 100:.1f}%)"
    )
    held = {}
    if base in per_q:
        for name in dict.fromkeys([winner, base]):  # winner first, then baseline (dedup if winner==baseline)
            lo, hi = split_ci(name, rep_idx)
            held[name] = {
                "recall": round(split_mean(name, rep_idx) * 100, 1),
                "ci95": [round(lo * 100, 1), round(hi * 100, 1)],
            }
        if winner != base:
            diff = [per_q[winner][i] - per_q[base][i] for i in rep_idx]
            w = sum(d > 0 for d in diff)
            t = sum(d == 0 for d in diff)
            lo, hi = bootstrap_ci(diff, args.bootstrap)
            sig = "significant" if lo > 0 else "n.s. (CI includes 0)"
            print(f"held-out {rep} (n={len(rep_idx)}):")
            print(
                f"  {winner:14} r@{args.budget} = {held[winner]['recall']:.1f}%   vs baseline {held[base]['recall']:.1f}%"
            )
            print(
                f"  paired (winner-baseline) win/tie/loss = {w}/{t}/{len(diff) - w - t}   diff 95%CI [{lo * 100:+.1f},{hi * 100:+.1f}]  {sig}"
            )
        else:
            print(f"held-out {rep}: winner == baseline; nothing beats it on {sel}.")

    if args.output:
        out = {
            "corpus_docs": len(corpus),
            "split_counts": dict(Counter(q["split"] for q in questions)),
            "budget": args.budget,
            "select_split": sel,
            "report_split": rep,
            "baseline": base,
            "winner": winner,
            "held_out": held,
            "strategies": {
                n: {
                    **meta[n],
                    f"{sel}_recall": round(split_mean(n, sel_idx) * 100, 1),
                    f"{rep}_recall": round(split_mean(n, rep_idx) * 100, 1),
                }
                for n in order
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
