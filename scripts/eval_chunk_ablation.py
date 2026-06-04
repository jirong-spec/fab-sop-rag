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

Rigour: similarity_search is deterministic, so the only variance is the question
sample. We report a bootstrap 95% CI per strategy and a paired bootstrap CI of
(strategy − baseline) per question, so "X beats the baseline" is a significance
claim, not a single point estimate.

Honest scope caveat: the corpus is 3 SOP docs / 27 answerable questions. Inter-document
routing is near-trivial at this size, so this mainly measures *within-document* chunking
(how to split one rich doc) and the budget-packing trade-off. The CIs among non-baseline
strategies overlap; the robust, significant finding is "the 400/80 baseline is poorly
placed", not a unique optimum. It is a retrieval-recall proxy, not end-to-end answer
quality (the graph, not the vector store, drives answers here).

Runs against a TEMPORARY collection (default "chunk_ablation"); the production
collection (settings.qdrant_collection) is never touched. The temp collection is
deleted on exit.

Usage (qdrant must be up; needs the embedding model, so run in the api image):
    docker compose up -d qdrant
    docker compose run --rm -T --no-deps api python scripts/eval_chunk_ablation.py
    docker compose run --rm -T --no-deps api python scripts/eval_chunk_ablation.py \
        --budget 1500 --bootstrap 3000 --output data/eval_results/chunk_ablation.json
"""

import argparse
import glob
import json
import logging
import random
import statistics
import sys
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
    """Answerable questions whose expected_keywords actually occur in the corpus."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r.get("type") != "answerable":
            continue
        kw = [k for k in r.get("expected_keywords", []) if k in corpus_text]
        if kw:
            out.append({"question": r["question"], "kw": kw})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate chunking strategies on budget-normalised keyword recall.")
    ap.add_argument("--budget", type=int, default=1500, help="primary retrieved-context budget in chars")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 1500, 2000], help="budgets shown in the table")
    ap.add_argument("--bootstrap", type=int, default=3000, help="bootstrap resamples for CIs")
    ap.add_argument("--baseline", type=str, default="char-400/80", help="strategy to compare others against")
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
        "corpus: %d docs (avg %d chars) | %d answerable questions",
        len(corpus),
        statistics.mean(len(t) for t in corpus.values()),
        len(questions),
    )
    if not questions:
        raise SystemExit("no answerable questions with in-corpus keywords")

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model, model_kwargs={"device": "cuda"})
    client = QdrantClient(url=settings.qdrant_url)
    strategies = build_strategies()

    per_q: dict[str, list[float]] = {}  # strategy -> per-question recall at --budget
    summary: dict[str, dict] = {}
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
            budget_recall = {b: [] for b in args.budgets}
            primary = []
            for q in questions:
                hits = vs.similarity_search(q["question"], k=10)
                for b in args.budgets:
                    budget_recall[b].append(recall_at_budget(hits, q["kw"], b))
                primary.append(recall_at_budget(hits, q["kw"], args.budget))
            per_q[name] = primary
            summary[name] = {
                "n_chunks": len(docs),
                "avg_chunk_chars": round(statistics.mean(len(d.page_content) for d in docs), 1),
                "recall_by_budget": {b: round(statistics.mean(v) * 100, 1) for b, v in budget_recall.items()},
            }
            logger.info(
                "  %-14s %3d chunks  avg %4.0f chars  r@%d=%.1f%%",
                name,
                len(docs),
                summary[name]["avg_chunk_chars"],
                args.budget,
                statistics.mean(primary) * 100,
            )
    finally:
        client.delete_collection(args.collection)
        logger.info("cleaned up temporary collection %r", args.collection)

    # ── report ───────────────────────────────────────────────────────────────
    order = sorted(summary, key=lambda k: -summary[k]["recall_by_budget"][args.budget])
    bud_hdr = "".join(f"r@{b:<6}" for b in args.budgets)
    print(f"\n{'strategy':14}{'#ch':>5}{'avg':>6}  {bud_hdr}  r@{args.budget} 95%CI")
    print("-" * (40 + 8 * len(args.budgets)))
    for name in order:
        s = summary[name]
        lo, hi = bootstrap_ci(per_q[name], args.bootstrap)
        s["recall_ci95"] = [round(lo * 100, 1), round(hi * 100, 1)]
        cells = "".join(f"{s['recall_by_budget'][b]:5.1f}% " for b in args.budgets)
        mark = " <- baseline" if name == args.baseline else ""
        print(f"{name:14}{s['n_chunks']:5}{s['avg_chunk_chars']:6.0f}  {cells} [{lo * 100:4.1f},{hi * 100:4.1f}]{mark}")

    base = args.baseline
    if base in per_q:
        print(f"\npaired vs baseline {base!r} (per-question, r@{args.budget}):")
        for name in order:
            if name == base:
                continue
            diff = [a - b for a, b in zip(per_q[name], per_q[base], strict=True)]
            w = sum(d > 0 for d in diff)
            t = sum(d == 0 for d in diff)
            lo, hi = bootstrap_ci(diff, args.bootstrap)
            summary[name]["vs_baseline"] = {
                "win": w,
                "tie": t,
                "loss": len(diff) - w - t,
                "diff_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
            }
            sig = "significant" if lo > 0 else "n.s."
            print(
                f"  {name:14} win/tie/loss = {w:2}/{t:2}/{len(diff) - w - t:2}"
                f"   diff 95%CI [{lo * 100:+.1f},{hi * 100:+.1f}]  {sig}"
            )

    if args.output:
        out = {
            "corpus_docs": len(corpus),
            "n_questions": len(questions),
            "primary_budget": args.budget,
            "baseline": base,
            "strategies": summary,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
