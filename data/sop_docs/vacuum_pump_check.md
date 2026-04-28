# SOP_Pump_002 — 真空泵浦狀態檢查程序

> **文件類型：範例資料（Sample Data）**  
> 本文件為教學用途之虛構 SOP，不代表任何真實廠商之操作規範。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_Pump_002 |
| 文件名稱 | 真空泵浦狀態檢查程序 |
| 適用設備 | TurboVacuumPump, DryPump |
| 版本 | Rev. 1.4 |
| 異常類型 | PumpDegradation |

---

## 1. 適用範圍

本程序適用於 TurboVacuumPump（渦輪泵浦）與 DryPump（乾式泵浦）之例行狀態確認，
以及因壓力異常觸發之計劃外檢查。

---

## 2. 前置確認條件（PRECONDITION）

- **[PRECONDITION-PMP01]** EtchStation 製程狀態 = IDLE（非製程進行中）
- **[PRECONDITION-PMP02]** 操作員已取得設備維護介面登入權限（Level-3）
- **[PRECONDITION-PMP03]** PumpControlPanel 可正常存取

---

## 3. 檢查步驟

### 步驟 1：ReadPumpStatus（第一步驟）

從設備控制介面讀取泵浦即時狀態。

- TurboVacuumPump 轉速（RPM）：正常值 ≥ 90% 額定轉速（27,000 RPM）
- TurboVacuumPump 溫度：正常值 ≤ 55°C
- DryPump 排氣壓力：正常值 ≤ 100 Pa

**依賴關係：** ReadPumpStatus REQUIRES_STATUS PumpControlPanel.ACCESSIBLE

---

### 步驟 2：CheckBearingVibration（下一步驟）

使用振動感測器確認軸承狀態。

- 振動加速度 ≤ 2.5 mm/s²（符合 ISO 10816-1 Class I）
- 若振動值超出範圍，標記 TurboVacuumPump 狀態為 MAINTENANCE_REQUIRED

**依賴關係：** CheckBearingVibration DEPENDS_ON ReadPumpStatus

---

### 步驟 3：VerifyPumpCooling（下一步驟）

確認冷卻水迴路正常運作。

- 冷卻水入口溫度：15°C ~ 25°C
- 冷卻水流量：≥ 2.0 L/min
- 若冷卻水異常，立即通知 Facility Engineer 並暫停泵浦使用

**依賴關係：** VerifyPumpCooling DEPENDS_ON CheckBearingVibration

---

### 步驟 4：LogAndClearAlarm（下一步驟）

記錄檢查結果並清除警報。

- 將所有讀值記錄至 PM（Preventive Maintenance）日誌
- 若所有項目正常，清除 PumpDegradation 警報
- 若有異常項目，填寫 MAR（Maintenance Action Request）單

**依賴關係：** LogAndClearAlarm DEPENDS_ON VerifyPumpCooling

---

## 4. 設備狀態定義

| 狀態代碼 | 含義 |
|---------|------|
| RUNNING | 泵浦正常運作中，達到額定轉速 |
| STARTING | 泵浦啟動中，尚未達到額定轉速 |
| STANDBY | 泵浦待機，低速旋轉保溫 |
| MAINTENANCE_REQUIRED | 需要維護，禁止用於製程 |
| FAULT | 故障停機，需工程師介入 |

---

## 5. 相關文件

- SOP_Etch_001：蝕刻站壓力異常處置程序（引用本文件泵浦狀態定義）
- SOP_Vent_003：腔體洩壓程序
- SOP_Pump_002 定義於文件庫 FabSOP_Library

**跨文件依賴說明：**
SOP_Etch_001 步驟 1（CheckVacuumPump）中，TurboVacuumPump 的狀態代碼（RUNNING、FAULT 等）
定義於本文件（SOP_Pump_002）第 4 節。此為典型「跨文件設備依賴關係」。
