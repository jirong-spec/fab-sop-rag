"""
SOP-to-Graph Extraction Pipeline

Reads SOP Markdown files and uses an LLM to automatically extract
structured nodes and edges for the knowledge graph.

This script is the upstream pipeline that produces the graph_seed JSON
consumed by ingest_graph.py → Neo4j.  Without it, someone would have to
hand-write nodes.json and edges.json for every new SOP document.

Extraction is done in two LLM passes per document:
  Pass 1 — Nodes : SOPDocument, SOPStep, Equipment, Anomaly, ProcessCondition
  Pass 2 — Edges : TRIGGERS_SOP, FIRST_STEP, NEXT_STEP, DEPENDS_ON,
                   DEFINED_IN, REQUIRES_STATUS, PRECONDITION,
                   INTERLOCK_WITH, CROSS_DOC_DEPENDENCY

Output
------
  Default (--merge not set):
    data/graph_seed/nodes_extracted.json
    data/graph_seed/edges_extracted.json

  With --merge:
    Merges into data/graph_seed/nodes.json + edges.json (deduplicates by id)

Usage
-----
  # Extract from all SOP docs (requires vLLM running):
  python scripts/extract_graph_from_sop.py

  # Single file:
  python scripts/extract_graph_from_sop.py --file data/sop_docs/etch_pressure_anomaly.md

  # Preview without writing:
  python scripts/extract_graph_from_sop.py --dry-run

  # Merge directly into graph seed:
  python scripts/extract_graph_from_sop.py --merge

  # Full pipeline (extract → merge → ingest):
  python scripts/extract_graph_from_sop.py --merge && python scripts/ingest_graph.py
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.llm_client import chat_completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOP_DOCS_DIR = ROOT / "data" / "sop_docs"
GRAPH_SEED_DIR = ROOT / "data" / "graph_seed"


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict | None:
    """
    Robustly extract a JSON object from LLM output.

    LLMs commonly wrap JSON in markdown fences (```json ... ```) or add
    explanatory text before/after.  This function strips those and finds
    the outermost { ... } block using a bracket counter so that nested
    objects are not truncated.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # Find the outermost JSON object via bracket counting
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    logger.debug("JSON decode error: %s", exc)
                    return None
    return None


# ── Prompts ───────────────────────────────────────────────────────────────────

_NODE_PROMPT = """\
你是一位知識圖譜建構專家，專門從晶圓廠 SOP 文件中提取結構化節點。

請從以下 SOP 文件中識別所有實體，分類為下列五種節點標籤：

【節點標籤規格】

SOPDocument — SOP 文件本身
  必要屬性：id（英文代碼，如 SOP_Etch_001）, title（中文名稱）, version（版本號）, equipment（主要適用設備 id）

SOPStep — SOP 中的操作步驟
  必要屬性：id（TitleCase 英文，如 CheckVacuumPump）, description（步驟說明）, sop_doc（所屬 SOP id）, step_number（整數，從 1 開始）

Equipment — 機台或設備（包含閥門、電源供應器、感測器等）
  必要屬性：id（TitleCase 英文，如 TurboVacuumPump）, type（設備類型，如 VacuumPump / Valve / SafetyInterlock）, description（說明）
  選用屬性：rated_rpm, normal_temp_max_C, exhaust_pressure_max_Pa 等機台規格數值

Anomaly — 異常事件或故障類型
  必要屬性：id（TitleCase 英文，如 PressureAnomaly）, description（說明）
  選用屬性：threshold_high_mTorr, threshold_low_mTorr, duration_sec, threshold_rpm_pct 等數值門檻

ProcessCondition — 製程條件或參數規格
  必要屬性：id（英文代碼，如 EtchGasFlow_Cl2）, parameter（參數名稱）, unit（單位）
  選用屬性：target, tolerance, max_allowed 等數值

【提取原則】
1. id 只用英文，TitleCase 或 Code_格式，不含中文或空格
2. 每個節點 id 必須唯一
3. 只提取文件中明確出現的實體，不要推測或補充
4. 步驟 id 直接使用文件中的步驟名稱（如 CheckVacuumPump, VerifyGasFlow）
5. 回傳純 JSON，不要有任何說明文字或 markdown 格式

【輸出格式】
{{
  "nodes": [
    {{"label": "SOPDocument", "properties": {{"id": "SOP_Etch_001", "title": "蝕刻站壓力異常處置程序", "version": "Rev. 2.1", "equipment": "EtchStation"}}}},
    {{"label": "SOPStep", "properties": {{"id": "CheckVacuumPump", "description": "確認真空泵浦運作狀態", "sop_doc": "SOP_Etch_001", "step_number": 1}}}},
    {{"label": "Equipment", "properties": {{"id": "TurboVacuumPump", "type": "VacuumPump", "description": "渦輪分子泵浦"}}}},
    {{"label": "Anomaly", "properties": {{"id": "PressureAnomaly", "description": "腔體壓力超出允許範圍", "threshold_high_mTorr": 5.0}}}},
    {{"label": "ProcessCondition", "properties": {{"id": "EtchGasFlow_Cl2", "parameter": "Cl2_flow_sccm", "target": 50, "tolerance": 5, "unit": "sccm"}}}}
  ]
}}

【SOP 文件內容】
{content}"""


_EDGE_PROMPT = """\
你是一位知識圖譜建構專家，專門從晶圓廠 SOP 文件中提取節點間的關係。

【已識別節點清單】
{node_list}

【邊類型規格】

TRIGGERS_SOP      Anomaly → SOPDocument
  異常事件觸發應執行的 SOP
  properties: {{}}

FIRST_STEP        SOPDocument → SOPStep
  SOP 文件的第一個步驟（每份 SOP 只有一條）
  properties: {{}}

NEXT_STEP         SOPStep → SOPStep
  步驟執行順序（每步驟只指向緊鄰的下一步）
  properties: {{}}

DEPENDS_ON        SOPStep → SOPStep
  執行此步驟前必須已完成的前置步驟（通常與 NEXT_STEP 方向相反）
  properties: {{}}

DEFINED_IN        SOPStep → SOPDocument
  步驟所屬的 SOP 文件
  properties: {{}}

REQUIRES_STATUS   SOPStep → Equipment
  步驟執行時設備必須處於的狀態
  properties: {{"required_status": "RUNNING"}}

PRECONDITION      SOPDocument → Equipment
  整份 SOP 執行前的設備前置條件
  properties: {{"required_status": "RUNNING", "condition_id": "PRECONDITION-P01"}}

INTERLOCK_WITH    Equipment → Equipment
  設備間的安全聯鎖關係
  properties: {{"interlock_id": "IL-E001", "trigger": "pressure > 10 mTorr", "action": "disable RF power"}}

CROSS_DOC_DEPENDENCY  SOPDocument → SOPDocument
  此 SOP 引用另一份 SOP 中的定義或程序
  properties: {{"reason": "說明引用原因"}}

【提取原則】
1. from_id 和 to_id 必須來自上方節點清單，不得自創 id
2. NEXT_STEP 和 DEPENDS_ON 通常成對出現在連續步驟之間
3. 每個步驟都應有一條 DEFINED_IN 邊指向其所屬 SOP
4. 只提取文件中明確記載的關係，不要推測
5. 回傳純 JSON，不要有任何說明文字或 markdown 格式

【輸出格式】
{{
  "edges": [
    {{"type": "TRIGGERS_SOP", "from_label": "Anomaly", "from_id": "PressureAnomaly", "to_label": "SOPDocument", "to_id": "SOP_Etch_001", "properties": {{}}}},
    {{"type": "FIRST_STEP", "from_label": "SOPDocument", "from_id": "SOP_Etch_001", "to_label": "SOPStep", "to_id": "CheckVacuumPump", "properties": {{}}}},
    {{"type": "NEXT_STEP", "from_label": "SOPStep", "from_id": "CheckVacuumPump", "to_label": "SOPStep", "to_id": "VerifyGasFlow", "properties": {{}}}},
    {{"type": "REQUIRES_STATUS", "from_label": "SOPStep", "from_id": "CheckVacuumPump", "to_label": "Equipment", "to_id": "TurboVacuumPump", "properties": {{"required_status": "RUNNING"}}}}
  ]
}}

【SOP 文件內容】
{content}"""


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_nodes(content: str) -> list[dict]:
    prompt = _NODE_PROMPT.format(content=content)
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=2048)
    except RuntimeError as exc:
        logger.error("Node extraction LLM call failed: %s", exc)
        return []
    data = _parse_llm_json(raw)
    if data is None:
        logger.warning("Node extraction: could not parse JSON from LLM output")
        logger.debug("Raw output: %s", raw[:300])
        return []
    nodes = data.get("nodes", [])
    logger.info("  Extracted %d nodes", len(nodes))
    return nodes


def extract_edges(content: str, nodes: list[dict]) -> list[dict]:
    node_list = "\n".join(
        f"  {n['properties']['id']} ({n['label']})"
        for n in nodes
        if "id" in n.get("properties", {})
    )
    prompt = _EDGE_PROMPT.format(content=content, node_list=node_list)
    try:
        raw = chat_completion(prompt, temperature=0.0, max_tokens=2048)
    except RuntimeError as exc:
        logger.error("Edge extraction LLM call failed: %s", exc)
        return []
    data = _parse_llm_json(raw)
    if data is None:
        logger.warning("Edge extraction: could not parse JSON from LLM output")
        logger.debug("Raw output: %s", raw[:300])
        return []
    edges = data.get("edges", [])
    logger.info("  Extracted %d edges", len(edges))
    return edges


def validate_edges(edges: list[dict], nodes: list[dict]) -> list[dict]:
    """Drop edges whose from_id or to_id don't appear in the node list."""
    valid_ids = {n["properties"]["id"] for n in nodes if "id" in n.get("properties", {})}
    valid, dropped = [], []
    for e in edges:
        fid, tid = e.get("from_id"), e.get("to_id")
        if fid in valid_ids and tid in valid_ids:
            valid.append(e)
        else:
            dropped.append((fid, tid))
    if dropped:
        logger.warning("  Dropped %d edges with unknown node IDs: %s", len(dropped), dropped)
    return valid


def derive_structural_edges(nodes: list[dict]) -> list[dict]:
    """
    Deterministically generate DEFINED_IN and FIRST_STEP edges from node properties.

    These edges are 100% derivable from what the LLM already extracted —
    no second-guessing needed. Every SOPStep has sop_doc and step_number;
    DEFINED_IN and FIRST_STEP follow directly from those values.

    Why not rely on LLM for these?
    Small models frequently miss or mis-label these structural edges.
    Deterministic derivation guarantees complete coverage and avoids the
    validate_edges drop rate that plagues LLM-generated structural edges.
    """
    edges = []
    sop_ids = {
        n["properties"]["id"]
        for n in nodes
        if n["label"] == "SOPDocument" and "id" in n.get("properties", {})
    }

    for n in nodes:
        if n["label"] != "SOPStep":
            continue
        props = n.get("properties", {})
        step_id = props.get("id")
        sop_doc = props.get("sop_doc")
        step_num = props.get("step_number")
        if not step_id or not sop_doc or sop_doc not in sop_ids:
            continue

        # DEFINED_IN — every step belongs to its SOP document
        edges.append({
            "type": "DEFINED_IN",
            "from_label": "SOPStep",
            "from_id": step_id,
            "to_label": "SOPDocument",
            "to_id": sop_doc,
            "properties": {},
        })

        # FIRST_STEP — step_number 1 is the entry point of the SOP
        if step_num == 1:
            edges.append({
                "type": "FIRST_STEP",
                "from_label": "SOPDocument",
                "from_id": sop_doc,
                "to_label": "SOPStep",
                "to_id": step_id,
                "properties": {},
            })

    return edges


# ── Merge helpers ─────────────────────────────────────────────────────────────

def merge_nodes(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    seen = {n["properties"]["id"] for n in existing if "id" in n.get("properties", {})}
    merged = list(existing)
    added = 0
    for node in new:
        nid = node.get("properties", {}).get("id")
        if nid and nid not in seen:
            merged.append(node)
            seen.add(nid)
            added += 1
    return merged, added


def merge_edges(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    seen = {(e["type"], e["from_id"], e["to_id"]) for e in existing}
    merged = list(existing)
    added = 0
    for edge in new:
        key = (edge.get("type"), edge.get("from_id"), edge.get("to_id"))
        if None not in key and key not in seen:
            merged.append(edge)
            seen.add(key)
            added += 1
    return merged, added


# ── Per-file pipeline ─────────────────────────────────────────────────────────

def process_file(md_path: Path) -> tuple[list[dict], list[dict]]:
    logger.info("Processing: %s", md_path.name)
    content = md_path.read_text(encoding="utf-8")
    nodes = extract_nodes(content)
    if not nodes:
        logger.warning("  No nodes extracted — skipping edge extraction")
        return [], []

    # LLM pass for non-structural edges
    llm_edges = extract_edges(content, nodes)
    llm_edges = validate_edges(llm_edges, nodes)

    # Remove DEFINED_IN and FIRST_STEP from LLM output — derived deterministically below
    llm_edges = [e for e in llm_edges if e.get("type") not in {"DEFINED_IN", "FIRST_STEP"}]

    # Deterministic structural edges (100% recall, no hallucination)
    structural = derive_structural_edges(nodes)
    logger.info("  Derived %d structural edges (DEFINED_IN + FIRST_STEP)", len(structural))

    edges, _ = merge_edges(structural, llm_edges)
    return nodes, edges


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph nodes and edges from SOP Markdown files"
    )
    parser.add_argument(
        "--file", type=str,
        help="Process a single file (default: all *.md in data/sop_docs/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print extracted JSON to stdout, do not write any files",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge results into nodes.json / edges.json (default: write to *_extracted.json)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(GRAPH_SEED_DIR),
        help=f"Output directory (default: {GRAPH_SEED_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect files ─────────────────────────────────────────────────────────
    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(SOP_DOCS_DIR.glob("*.md"))

    if not files:
        logger.error("No SOP Markdown files found in %s", SOP_DOCS_DIR)
        sys.exit(1)

    logger.info("Files to process: %d", len(files))

    # ── Extract ───────────────────────────────────────────────────────────────
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    for f in files:
        nodes, edges = process_file(f)
        all_nodes, n_added = merge_nodes(all_nodes, nodes)
        all_edges, e_added = merge_edges(all_edges, edges)
        logger.info("  Running total: %d nodes (+%d), %d edges (+%d)",
                    len(all_nodes), n_added, len(all_edges), e_added)

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"  Nodes : {len(all_nodes)}")
    print(f"  Edges : {len(all_edges)}")
    print(f"{'='*55}")

    if not all_nodes:
        logger.error("Nothing extracted — check that vLLM is running and the SOP files exist")
        sys.exit(1)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n=== NODES ===")
        print(json.dumps(all_nodes, ensure_ascii=False, indent=2))
        print("\n=== EDGES ===")
        print(json.dumps(all_edges, ensure_ascii=False, indent=2))
        print("\n[dry-run] No files written.")
        return

    # ── Write / merge ─────────────────────────────────────────────────────────
    if args.merge:
        nodes_path = out_dir / "nodes.json"
        edges_path = out_dir / "edges.json"

        existing_nodes = (
            json.loads(nodes_path.read_text(encoding="utf-8"))
            if nodes_path.exists() else []
        )
        existing_edges = (
            json.loads(edges_path.read_text(encoding="utf-8"))
            if edges_path.exists() else []
        )

        all_nodes, added_n = merge_nodes(existing_nodes, all_nodes)
        all_edges, added_e = merge_edges(existing_edges, all_edges)
        logger.info("After merge: %d nodes (%+d new), %d edges (%+d new)",
                    len(all_nodes), added_n, len(all_edges), added_e)
    else:
        nodes_path = out_dir / "nodes_extracted.json"
        edges_path = out_dir / "edges_extracted.json"

    nodes_path.write_text(
        json.dumps(all_nodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    edges_path.write_text(
        json.dumps(all_edges, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n  nodes → {nodes_path.relative_to(ROOT)}")
    print(f"  edges → {edges_path.relative_to(ROOT)}")

    if not args.merge:
        print(f"\n  Review the files, then merge into graph seed:")
        print(f"  python scripts/extract_graph_from_sop.py --merge")

    print(f"\n  Ingest into Neo4j:")
    print(f"  python scripts/ingest_graph.py")
    print(f"  (or: docker compose run --rm api python scripts/ingest_all.py)")


if __name__ == "__main__":
    main()
