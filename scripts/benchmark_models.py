#!/usr/bin/env python3
"""
Model benchmark: loop through Qwen2.5 variants in ~/models,
swap vLLM for each, run eval_compare.py, print comparison table.

Skips Qwen2.5-7B-Instruct (FP16 ~14GB > RTX 3060 12GB VRAM).

Usage (from project root):
    python scripts/benchmark_models.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = [
    {"name": "Qwen2.5-3B-Instruct",          "quantization": None,   "max_model_len": 4096},
    {"name": "Qwen2.5-3B-Instruct-GPTQ-int8", "quantization": "gptq", "max_model_len": 4096},
    {"name": "Qwen2.5-3B-Instruct-GPTQ-int4", "quantization": "gptq", "max_model_len": 4096},
    {"name": "Qwen2.5-3B-Instruct-AWQ-int4",  "quantization": "awq",  "max_model_len": 4096},
    {"name": "Qwen2.5-7B-Instruct-AWQ-int4",  "quantization": "awq",  "max_model_len": 4096},
]

VLLM_READY_TIMEOUT = 900   # seconds to wait for vllm to load a model
VLLM_POLL_INTERVAL = 10


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT, **kwargs)


def _write_override(model: dict) -> Path:
    """Write a docker-compose override that changes only the vllm command."""
    cmd = [
        "--model",        f"/llm/{model['name']}",
        "--host",         "0.0.0.0",
        "--port",         "8000",
        "--served-model-name", "Qwen2.5-3B-Instruct",   # fixed alias so api finds it
        "--device",       "cuda",
        "--max-model-len", str(model["max_model_len"]),
        "--kv-cache-dtype", "auto",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
    ]
    if model["quantization"]:
        cmd += ["--quantization", model["quantization"]]
    # AWQ requires float16; vLLM v0.6.3 does not support bfloat16 + AWQ
    if model["quantization"] == "awq":
        cmd += ["--dtype", "float16"]

    lines = ["services:", "  vllm:", "    command:"]
    for part in cmd:
        lines.append(f"      - \"{part}\"")

    override_path = PROJECT_ROOT / "docker-compose.benchmark-override.yml"
    override_path.write_text("\n".join(lines) + "\n")
    return override_path


def _wait_for_vllm(timeout: int = VLLM_READY_TIMEOUT) -> bool:
    """Poll localhost:8299/v1/models until 200 or timeout."""
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8299/v1/models", timeout=5)
            print(f"\n  vLLM ready")
            return True
        except Exception:
            print(".", end="", flush=True)
            dots += 1
            if dots % 60 == 0:
                elapsed = int(time.time() - (deadline - timeout))
                print(f" [{elapsed}s]")
            time.sleep(VLLM_POLL_INTERVAL)
    print(f"\n  vLLM did not become ready within {timeout}s")
    return False


def _run_eval(model_name: str) -> dict | None:
    """Run eval_compare.py inside the api container, return parsed result dict."""
    container_path = f"/app/data/eval_results/benchmark_{model_name}.json"

    proc = _run([
        "docker", "compose", "exec", "-T", "api",
        "python", "scripts/eval_compare.py",
        "--output", container_path,
    ], capture_output=True, text=True)

    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"  [ERROR] eval failed:\n{proc.stderr[-500:]}")
        return None

    # Read result directly from inside the container (no host mount needed)
    cat = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "cat", container_path],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if cat.returncode != 0:
        print(f"  [ERROR] could not read result from container: {cat.stderr[:200]}")
        return None

    return json.loads(cat.stdout)


def _parse_result(data: dict) -> dict:
    """Extract summary metrics from eval result JSON using the same logic as render_report."""
    if not data:
        return {}

    queries  = data.get("queries", [])
    results  = data.get("results", {})

    g_kw_hits = g_kw_total = 0
    g_latencies: list[int] = []

    for q in queries:
        qid  = q["id"]
        res  = results.get(qid, {})
        g_resp = res.get("graph", {})
        behavior = q.get("expected_behavior", "retrieved_and_answered")
        keywords = q.get("expected_keywords", [])

        g_latencies.append(g_resp.get("latency_ms", 0))

        if behavior == "blocked_by_topic_guard":
            continue   # block queries counted separately; skip for keyword score

        haystack = g_resp.get("answer", "")
        hits = sum(1 for kw in keywords if kw.lower() in haystack.lower())
        g_kw_hits += hits
        g_kw_total += len(keywords)

    avg_ms = int(sum(g_latencies) / len(g_latencies)) if g_latencies else 0
    pct    = g_kw_hits / g_kw_total * 100 if g_kw_total else 0

    return {
        "score":  f"{g_kw_hits}/{g_kw_total}",
        "pct":    f"{pct:.1f}%",
        "avg_ms": avg_ms,
    }


def main():
    print("=" * 60)
    print("  Fab SOP RAG — Model Benchmark")
    print("=" * 60)

    # Ensure neo4j + api are up (don't touch vllm yet)
    print("\n[1/3] Starting neo4j + api...")
    _run(["docker", "compose", "up", "-d", "neo4j", "api"])
    time.sleep(5)

    results = []

    for i, model in enumerate(MODELS, 1):
        name = model["name"]
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(MODELS)}] {name}")
        print("=" * 60)

        # Write override and restart vllm
        override = _write_override(model)
        print(f"\n[Step 1] Restarting vLLM with {name}...")
        _run([
            "docker", "compose",
            "-f", "docker-compose.yml",
            "-f", str(override.name),
            "up", "-d", "--no-deps", "--force-recreate", "vllm",
        ])

        print(f"[Step 2] Waiting for vLLM to load model (up to {VLLM_READY_TIMEOUT}s)...")
        if not _wait_for_vllm():
            results.append({"model": name, "score": "TIMEOUT", "pct": "-", "avg_ms": "-"})
            continue

        print(f"[Step 3] Running eval_compare.py...")
        data = _run_eval(name)
        metrics = _parse_result(data)
        metrics["model"] = name
        results.append(metrics)
        print(f"  Result: {metrics.get('score')} ({metrics.get('pct')})  avg {metrics.get('avg_ms')} ms")

    # Restore original docker-compose (restart vllm with original model)
    override_path = PROJECT_ROOT / "docker-compose.benchmark-override.yml"
    if override_path.exists():
        override_path.unlink()
    print("\n\n[Restore] Restarting vLLM with original model (Qwen2.5-3B-Instruct)...")
    _run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "vllm"])

    # Print summary table
    print("\n")
    print("=" * 60)
    print("  BENCHMARK RESULTS (Graph RAG, 10 questions)")
    print("=" * 60)
    header = f"{'Model':<40} {'Score':>8} {'Pct':>8} {'Avg ms':>8}"
    print(header)
    print("-" * 66)
    for r in results:
        print(f"{r['model']:<40} {str(r.get('score','?')):>8} {str(r.get('pct','?')):>8} {str(r.get('avg_ms','?')):>8}")
    print("=" * 60)

    # Save summary
    out = PROJECT_ROOT / "data" / "eval_results" / "benchmark_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
