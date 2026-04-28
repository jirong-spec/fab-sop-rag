# SOP_Etch_001 — 蝕刻站壓力異常處置程序

> **文件類型：範例資料（Sample Data）**  
> 本文件為教學用途之虛構 SOP，不代表任何真實廠商之操作規範。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_Etch_001 |
| 文件名稱 | 蝕刻站壓力異常處置程序 |
| 適用設備 | EtchStation |
| 版本 | Rev. 2.1 |
| 異常類型 | PressureAnomaly |

---

## 1. 適用範圍

本程序適用於蝕刻站（EtchStation）腔體壓力超出製程允許範圍之異常事件處置。
壓力異常門檻：腔體壓力 > 5 mTorr 或 < 0.5 mTorr 連續超過 30 秒。

---

## 2. 前置確認條件（PRECONDITION）

在啟動異常排查程序前，操作員必須確認以下所有條件：

- **[PRECONDITION-P01]** TurboVacuumPump 狀態 = RUNNING
- **[PRECONDITION-P02]** RFPowerSupply 狀態 = STANDBY 或 OFF（禁止在 RF 啟動狀態下進行排查）
- **[PRECONDITION-P03]** 腔體 PressureInterlock 未觸發
- **[PRECONDITION-P04]** 操作員已登入 EtchStation 操作介面並取得 Level-2 權限

---

## 3. 異常排查步驟

### 步驟 1：CheckVacuumPump（第一步驟）

確認真空泵浦（TurboVacuumPump）運作狀態。

- 查看泵浦狀態指示燈：綠色 = 正常，黃色 = 警告，紅色 = 故障
- 確認泵浦轉速 ≥ 90% 額定轉速
- 如泵浦狀態異常，執行 `VacuumPump_Check` 子程序後返回本步驟

**依賴關係：** CheckVacuumPump REQUIRES_STATUS TurboVacuumPump.RUNNING

---

### 步驟 2：VerifyGasFlow（下一步驟）

確認製程氣體流量是否在正常範圍。

- 檢查 MFC（Mass Flow Controller）讀值：
  - Cl₂：50 ± 5 sccm
  - HBr：20 ± 3 sccm
  - O₂：5 ± 1 sccm
- 若流量偏差 > 10%，記錄並通報 Process Engineer

**依賴關係：** VerifyGasFlow DEPENDS_ON CheckVacuumPump

---

### 步驟 3：InspectChamberLeak（下一步驟）

執行腔體洩漏測試。

- 關閉所有氣體供應閥
- 監測腔體壓力上升速率（Leak Rate）
- 允許值：Leak Rate ≤ 0.1 mTorr/min
- 若 Leak Rate > 0.1 mTorr/min，執行 `ChamberVent` 程序（參考 SOP_Vent_003）

**依賴關係：** InspectChamberLeak DEPENDS_ON VerifyGasFlow

---

### 步驟 4：RestoreProcessCondition（下一步驟）

恢復製程條件並確認壓力穩定。

- 開啟氣體供應閥至設定流量
- 等待腔體壓力穩定（< 30 秒達到目標壓力 ± 5%）
- 記錄壓力穩定時間至系統日誌

**依賴關係：** RestoreProcessCondition DEPENDS_ON InspectChamberLeak

---

## 4. Interlock 條件

| Interlock ID | 觸發條件 | 聯鎖動作 |
|-------------|----------|----------|
| IL-E001 | 壓力 > 10 mTorr | 自動關閉 RF 電源 |
| IL-E002 | TurboVacuumPump 轉速 < 80% | 暫停製程並發出警報 |
| IL-E003 | Leak Rate > 0.5 mTorr/min | 啟動緊急洩壓程序 |

**聯鎖關係：** EtchStation INTERLOCK_WITH PressureInterlock

---

## 5. 相關文件

- SOP_Pump_002：真空泵浦狀態檢查程序
- SOP_Vent_003：腔體洩壓程序
- SOP_Etch_001 定義於文件庫 FabSOP_Library
