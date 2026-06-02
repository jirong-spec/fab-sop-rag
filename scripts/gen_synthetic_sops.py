"""
Deterministically generate synthetic SOPs to enlarge the knowledge graph so that
retrieval is actually stress-tested (the 3-SOP / 48-edge seed is too small —
2-hop traversal returns nearly the whole graph, so recall@k is trivially 100%).

Design for retrieval pressure
-----------------------------
Several pieces of equipment are SHARED across many SOPs (RFPowerSupply,
N2PurgeSystem, TurboVacuumPump, DIWaterSystem). They become high-degree hubs:
a query about one SOP's use of a shared hub must be answered by selecting the
correct edge among many structurally identical edges from OTHER SOPs. That is
exactly the disambiguation the bi-encoder rerank + dynamic cap must get right.

Everything (nodes, edges, markdown) is generated from one SPEC, so the graph,
the source docs, and any gold labels derived from SPEC stay mutually consistent.
Merges into data/graph_seed/{nodes,edges}.json (dedup by id / (type,from,to))
and writes one Markdown file per new SOP into data/sop_docs/. Idempotent.

Run (from project root):  python scripts/gen_synthetic_sops.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "graph_seed"
DOCS = ROOT / "data" / "sop_docs"

# Equipment shared with the EXISTING seed (already present as nodes) — referenced, not redefined.
EXISTING_SHARED = {"TurboVacuumPump", "RFPowerSupply", "N2PurgeSystem", "DryPump"}

# New shared equipment (defined once, reused across several new SOPs → hubs).
SHARED_EQUIPMENT = {
    "GasDeliverySystem": "製程氣體輸送系統（多機台共用）",
    "DIWaterSystem": "去離子水供應系統（多機台共用）",
}

# Each SOP: id, title, anomaly(id,desc), equipment[{id,type,desc}], steps[{id,desc,requires?}],
# preconditions[(equip,status,cond_id)], interlock?(a,b,id,trigger,action), crossdoc[(target,reason)].
SOPS = [
    {
        "id": "SOP_CVD_010", "title": "CVD 薄膜沉積異常處置程序",
        "anomaly": ("FilmThicknessDrift", "CVD 薄膜厚度漂移超出規格範圍"),
        "equipment": [{"id": "CVDChamber", "type": "ProcessChamber", "desc": "化學氣相沉積腔體"}],
        "steps": [
            {"id": "CheckCVDVacuum", "desc": "確認沉積腔真空度", "requires": ("TurboVacuumPump", "RUNNING")},
            {"id": "CalibrateGasFlow", "desc": "校正製程氣體流量", "requires": ("GasDeliverySystem", "READY")},
            {"id": "MeasureFilmThickness", "desc": "量測薄膜厚度"},
            {"id": "AdjustDepositionRate", "desc": "調整沉積速率"},
            {"id": "VerifyFilmUniformity", "desc": "驗證薄膜均勻性"},
        ],
        "preconditions": [("TurboVacuumPump", "RUNNING", "PRECONDITION-CVD01"),
                          ("RFPowerSupply", "STANDBY_OR_OFF", "PRECONDITION-CVD02")],
        "interlock": ("CVDChamber", "GasDeliverySystem", "IL-C010", "chamber_pressure > 5 Torr", "close gas inlet"),
        "crossdoc": [("SOP_Pump_002", "TurboVacuumPump 狀態定義（RUNNING/FAULT）源自 SOP_Pump_002")],
    },
    {
        "id": "SOP_CMP_020", "title": "CMP 平坦化研磨異常處置程序",
        "anomaly": ("PlanarityDeviation", "CMP 研磨後平坦度偏離規格"),
        "equipment": [{"id": "CMPPolisher", "type": "Polisher", "desc": "化學機械研磨機台"},
                      {"id": "SlurrySupply", "type": "ChemicalSupply", "desc": "研磨液供應系統"},
                      {"id": "EndpointDetector", "type": "Sensor", "desc": "研磨終點偵測器"}],
        "steps": [
            {"id": "ReadPolishRecipe", "desc": "讀取研磨配方"},
            {"id": "CheckSlurryFlow", "desc": "檢查研磨液流量", "requires": ("SlurrySupply", "READY")},
            {"id": "MonitorEndpoint", "desc": "監看研磨終點", "requires": ("EndpointDetector", "ACTIVE")},
            {"id": "MeasurePostPolish", "desc": "量測研磨後厚度"},
            {"id": "CleanPolishedWafer", "desc": "清潔研磨後晶圓", "requires": ("DIWaterSystem", "RUNNING")},
        ],
        "preconditions": [("SlurrySupply", "READY", "PRECONDITION-CMP01")],
        "crossdoc": [("SOP_Clean_060", "研磨後須銜接 SOP_Clean_060 濕式清洗去除殘留研磨液")],
    },
    {
        "id": "SOP_Implant_030", "title": "離子佈植劑量異常處置程序",
        "anomaly": ("DoseUniformityError", "離子佈植劑量均勻度誤差超標"),
        "equipment": [{"id": "IonSource", "type": "Source", "desc": "離子源"},
                      {"id": "BeamlineScanner", "type": "Scanner", "desc": "束流掃描器"},
                      {"id": "FaradayCup", "type": "Sensor", "desc": "法拉第杯劑量偵測器"}],
        "steps": [
            {"id": "VerifyBeamCurrent", "desc": "驗證束流電流", "requires": ("FaradayCup", "CALIBRATED")},
            {"id": "CheckScanUniformity", "desc": "檢查掃描均勻度"},
            {"id": "AdjustImplantDose", "desc": "調整佈植劑量"},
            {"id": "ConfirmImplantLog", "desc": "確認佈植記錄"},
        ],
        "preconditions": [("RFPowerSupply", "OFF", "PRECONDITION-IMP01")],
    },
    {
        "id": "SOP_Litho_040", "title": "微影疊對誤差處置程序",
        "anomaly": ("OverlayError", "微影層間疊對誤差超出規格"),
        "equipment": [{"id": "Stepper", "type": "Scanner", "desc": "步進曝光機"},
                      {"id": "PhotoresistTrack", "type": "Track", "desc": "光阻塗佈顯影軌道"},
                      {"id": "AlignmentSensor", "type": "Sensor", "desc": "對準感測器"}],
        "steps": [
            {"id": "CheckAlignmentMarks", "desc": "檢查對準標記", "requires": ("AlignmentSensor", "READY")},
            {"id": "MeasureOverlay", "desc": "量測疊對誤差"},
            {"id": "AdjustStagePosition", "desc": "調整載台位置"},
            {"id": "ReexposeTestWafer", "desc": "重曝測試晶圓"},
        ],
        "preconditions": [("PhotoresistTrack", "READY", "PRECONDITION-LIT01")],
    },
    {
        "id": "SOP_Anneal_050", "title": "快速熱退火溫度異常處置程序",
        "anomaly": ("ThermalNonUniformity", "退火爐溫度均勻度異常"),
        "equipment": [{"id": "RTAChamber", "type": "ProcessChamber", "desc": "快速熱退火腔體"},
                      {"id": "TempController", "type": "Controller", "desc": "溫度控制器"}],
        "steps": [
            {"id": "CheckPurgeFlow", "desc": "確認吹淨氣體流量", "requires": ("N2PurgeSystem", "READY")},
            {"id": "RampTemperature", "desc": "升溫"},
            {"id": "HoldSoakTime", "desc": "恆溫保持"},
            {"id": "CoolDownChamber", "desc": "降溫"},
            {"id": "LogThermalProfile", "desc": "記錄熱製程曲線"},
        ],
        "preconditions": [("N2PurgeSystem", "READY", "PRECONDITION-ANN01")],
        "crossdoc": [("SOP_Vent_003", "退火後腔體洩壓須依 SOP_Vent_003 程序執行")],
    },
    {
        "id": "SOP_Clean_060", "title": "濕式清洗粒子污染處置程序",
        "anomaly": ("ParticleContamination", "濕式清洗後粒子污染超標"),
        "equipment": [{"id": "WetBench", "type": "WetStation", "desc": "濕式清洗槽"},
                      {"id": "ChemicalSupply", "type": "ChemicalSupply", "desc": "清洗化學品供應系統"}],
        "steps": [
            {"id": "PrepareChemicalBath", "desc": "配置清洗藥液", "requires": ("ChemicalSupply", "READY")},
            {"id": "ImmerseWafer", "desc": "晶圓浸泡"},
            {"id": "DIWaterRinse", "desc": "去離子水沖洗", "requires": ("DIWaterSystem", "RUNNING")},
            {"id": "SpinDryWafer", "desc": "甩乾晶圓"},
            {"id": "InspectParticles", "desc": "檢查粒子數"},
        ],
        "preconditions": [("ChemicalSupply", "READY", "PRECONDITION-CLN01")],
    },
    {
        "id": "SOP_Metro_070", "title": "關鍵尺寸量測漂移處置程序",
        "anomaly": ("MeasurementDrift", "CD-SEM 關鍵尺寸量測漂移"),
        "equipment": [{"id": "CDSEM", "type": "Metrology", "desc": "關鍵尺寸掃描電鏡"},
                      {"id": "OverlayMetrology", "type": "Metrology", "desc": "疊對量測機"}],
        "steps": [
            {"id": "CalibrateCDSEM", "desc": "校正 CD-SEM", "requires": ("CDSEM", "CALIBRATED")},
            {"id": "MeasureCriticalDimension", "desc": "量測關鍵尺寸"},
            {"id": "CompareToSpec", "desc": "比對規格"},
            {"id": "FlagOutOfSpec", "desc": "標記超規結果"},
        ],
        "preconditions": [("CDSEM", "CALIBRATED", "PRECONDITION-MET01")],
        "crossdoc": [("SOP_Litho_040", "量測超規須回饋 SOP_Litho_040 微影調整")],
    },
]


def build():
    nodes, edges = [], []

    def node(label, props):
        nodes.append({"label": label, "properties": props})

    def edge(t, fl, fi, tl, ti, props=None):
        edges.append({"type": t, "from_label": fl, "from_id": fi, "to_label": tl, "to_id": ti, "properties": props or {}})

    for eid, desc in SHARED_EQUIPMENT.items():
        node("Equipment", {"id": eid, "type": "SharedUtility", "description": desc})

    for s in SOPS:
        sid = s["id"]
        node("SOPDocument", {"id": sid, "title": s["title"], "version": "Rev. 1.0"})
        aid, adesc = s["anomaly"]
        node("Anomaly", {"id": aid, "description": adesc})
        edge("TRIGGERS_SOP", "Anomaly", aid, "SOPDocument", sid)
        for eq in s["equipment"]:
            node("Equipment", {"id": eq["id"], "type": eq["type"], "description": eq["desc"]})

        steps = s["steps"]
        for i, st in enumerate(steps):
            node("SOPStep", {"id": st["id"], "description": st["desc"], "sop_doc": sid, "step_number": i + 1})
            edge("DEFINED_IN", "SOPStep", st["id"], "SOPDocument", sid)
            if i == 0:
                edge("FIRST_STEP", "SOPDocument", sid, "SOPStep", st["id"])
            else:
                prev = steps[i - 1]["id"]
                edge("NEXT_STEP", "SOPStep", prev, "SOPStep", st["id"],
                     {"description": f"{prev} 完成後，下一步執行 {st['id']}"})
                edge("DEPENDS_ON", "SOPStep", st["id"], "SOPStep", prev,
                     {"description": f"{st['id']} 執行前必須先完成前置依賴步驟 {prev}"})
            if "requires" in st:
                eq, status = st["requires"]
                edge("REQUIRES_STATUS", "SOPStep", st["id"], "Equipment", eq, {"required_status": status})

        for eq, status, cid in s.get("preconditions", []):
            edge("PRECONDITION", "SOPDocument", sid, "Equipment", eq, {"required_status": status, "condition_id": cid})
        if "interlock" in s:
            a, b, ilid, trig, act = s["interlock"]
            edge("INTERLOCK_WITH", "Equipment", a, "Equipment", b, {"interlock_id": ilid, "trigger": trig, "action": act})
        for target, reason in s.get("crossdoc", []):
            edge("CROSS_DOC_DEPENDENCY", "SOPDocument", sid, "SOPDocument", target, {"reason": reason})

    return nodes, edges


def markdown(s):
    L = [f"# {s['id']}：{s['title']}", "", "> 合成範例資料，僅供測試/教學，非真實機台程序。", ""]
    aid, adesc = s["anomaly"]
    L += [f"## 觸發異常", f"- **{aid}**：{adesc}（觸發本 SOP {s['id']}）", ""]
    if s.get("preconditions"):
        L += ["## 前置條件（PRECONDITION）"]
        for eq, status, cid in s["preconditions"]:
            L.append(f"- {eq} 必須為 `{status}`（{cid}）")
        L.append("")
    L += ["## 步驟順序"]
    for i, st in enumerate(s["steps"], 1):
        line = f"{i}. **{st['id']}** — {st['desc']}"
        if "requires" in st:
            line += f"（需要 {st['requires'][0]} 狀態為 `{st['requires'][1]}`）"
        L.append(line)
    L.append("")
    if s.get("interlock"):
        a, b, ilid, trig, act = s["interlock"]
        L += ["## 設備聯鎖（INTERLOCK_WITH）", f"- {a} → {b}（{ilid}）：當 `{trig}` 時，動作為 `{act}`", ""]
    if s.get("crossdoc"):
        L += ["## 跨文件依賴（CROSS_DOC_DEPENDENCY）"]
        for target, reason in s["crossdoc"]:
            L.append(f"- {s['id']} → {target}：{reason}")
        L.append("")
    return "\n".join(L)


def merge_into(path, items, key):
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    seen = {key(x) for x in existing}
    added = 0
    for it in items:
        k = key(it)
        if k not in seen:
            existing.append(it); seen.add(k); added += 1
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return added, len(existing)


def main():
    nodes, edges = build()
    na, nt = merge_into(SEED / "nodes.json", nodes, lambda n: n["properties"]["id"])
    ea, et = merge_into(SEED / "edges.json", edges, lambda e: (e["type"], e["from_id"], e["to_id"]))
    print(f"nodes: +{na} (total {nt})   edges: +{ea} (total {et})")
    for s in SOPS:
        fn = "_".join(s["id"].lower().split("_")[1:])  # e.g. cvd_010
        (DOCS / f"{fn}.md").write_text(markdown(s), encoding="utf-8")
        print(f"  wrote {DOCS.name}/{fn}.md")


if __name__ == "__main__":
    main()
