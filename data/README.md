# Sample Data — Fab SOP Knowledge Query

> All files in this directory are **sample data for educational purposes only**.
> They represent fictitious SOP content and do not reflect any real manufacturer's procedures.

---

## Directory structure

```
data/
├── sop_docs/                     # Source SOP documents (Traditional Chinese Markdown)
│   ├── etch_pressure_anomaly.md  # SOP_Etch_001: 蝕刻站壓力異常處置程序
│   ├── vacuum_pump_check.md      # SOP_Pump_002: 真空泵浦狀態檢查程序
│   └── chamber_vent_procedure.md # SOP_Vent_003: 腔體洩壓程序
│
├── graph_seed/                   # Structured graph data for Neo4j
│   ├── nodes.json                # Node definitions (label + properties)
│   └── edges.json                # Edge definitions (type + from/to + properties)
│
└── sample_queries/
    └── fab_queries.json          # 10 test queries with expected behavior labels
```

---

## Graph schema overview

### Node labels

| Label | Count | Example id |
|-------|-------|------------|
| `SOPDocument` | 3 | `SOP_Etch_001` |
| `SOPStep` | 12 | `CheckVacuumPump` |
| `Equipment` | 8 | `TurboVacuumPump` |
| `Anomaly` | 2 | `PressureAnomaly` |
| `ProcessCondition` | 4 | `ChamberLeakRate` |

### Edge types

| Type | Meaning |
|------|---------|
| `TRIGGERS_SOP` | Anomaly → triggers this SOP document |
| `FIRST_STEP` | SOPDocument → its first SOPStep |
| `NEXT_STEP` | SOPStep → next SOPStep in sequence |
| `DEPENDS_ON` | SOPStep → must complete this step first |
| `REQUIRES_STATUS` | SOPStep → Equipment must be in this state |
| `PRECONDITION` | SOPDocument → Equipment precondition before starting |
| `DEFINED_IN` | SOPStep → SOPDocument it belongs to |
| `INTERLOCK_WITH` | Equipment → Equipment interlock relationship |
| `CROSS_DOC_DEPENDENCY` | SOPDocument → SOPDocument referenced by it |

---

## How ingestion works

1. **`scripts/ingest_graph.py`** reads `graph_seed/nodes.json` and `edges.json`,
   then MERGEs them into Neo4j using the `id` property as the unique key.

2. **`scripts/ingest_vector.py`** reads every `*.md` file in `sop_docs/`,
   splits each file into ~400-character overlapping chunks,
   embeds them with HuggingFace sentence-transformers,
   and upserts them into the `sop_docs` Chroma collection.

3. **`scripts/ingest_all.py`** runs both scripts in sequence.

Both graph and vector stores are queried at `/v1/ask` time via hybrid retrieval:
- Graph: entity extraction → Cypher BFS traversal
- Vector: semantic similarity search on embedded chunks
- Results are merged, deduplicated, and passed to the LLM for grounded answer generation.
