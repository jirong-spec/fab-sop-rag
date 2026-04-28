# SOP_Vent_003 — 腔體洩壓程序

> **文件類型：範例資料（Sample Data）**  
> 本文件為教學用途之虛構 SOP，不代表任何真實廠商之操作規範。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_Vent_003 |
| 文件名稱 | 腔體洩壓程序 |
| 適用設備 | EtchStation, VentValve, N2PurgeSystem |
| 版本 | Rev. 3.0 |
| 適用情境 | 計劃維護洩壓 / 緊急聯鎖洩壓 |

---

## 1. 適用範圍

本程序定義 EtchStation 腔體之標準洩壓（Vent）操作步驟，適用於：
- 計劃性維護前的腔體開啟準備
- 壓力 Interlock（IL-E003）觸發後的緊急洩壓
- 腔體洩漏率超標後的安全排氣

---

## 2. 前置確認條件（PRECONDITION）

- **[PRECONDITION-VNT01]** RFPowerSupply 狀態 = OFF
- **[PRECONDITION-VNT02]** 所有製程氣體供應閥已關閉
- **[PRECONDITION-VNT03]** TurboVacuumPump 已切換至 STANDBY（非 FAULT）
- **[PRECONDITION-VNT04]** N2PurgeSystem 已就緒（N2 壓力 ≥ 5 psig）

---

## 3. 洩壓步驟

### 步驟 1：IsolateRFAndGas（第一步驟）

確認所有高風險能源已隔離。

- 確認 RFPowerSupply 開關位於 OFF 並上鎖（LOTO 程序）
- 關閉所有氣體供應管線上的手動截止閥
- 確認 EtchStation 操作介面顯示「Process Stopped」

**依賴關係：**
- IsolateRFAndGas REQUIRES_STATUS RFPowerSupply.OFF
- IsolateRFAndGas REQUIRES_STATUS GasSupplyValves.CLOSED

---

### 步驟 2：SwitchPumpToStandby（下一步驟）

將 TurboVacuumPump 切換至 STANDBY 模式。

- 在 PumpControlPanel 點擊「Standby」
- 等待轉速降至 20% 額定轉速以下（約 60 秒）
- **不可直接關閉泵浦電源**，必須經由 STANDBY 過渡

**依賴關係：**
- SwitchPumpToStandby DEPENDS_ON IsolateRFAndGas
- SwitchPumpToStandby REQUIRES_STATUS TurboVacuumPump.RUNNING

---

### 步驟 3：OpenVentValve（下一步驟）

緩慢打開洩壓閥，引入 N₂ 置換腔體殘氣。

- 以 10% 開度增量緩慢開啟 VentValve（每步間隔 5 秒）
- N₂ 流量目標：20 slm（Standard Liter per Minute）
- 監測腔體壓力上升速率：不超過 5 Torr/sec
- 目標壓力：760 Torr（大氣壓）

**依賴關係：**
- OpenVentValve DEPENDS_ON SwitchPumpToStandby
- OpenVentValve REQUIRES_STATUS N2PurgeSystem.READY
- EtchStation INTERLOCK_WITH VentValve

---

### 步驟 4：ConfirmAtmPressure（下一步驟）

確認腔體壓力達到大氣壓並穩定。

- 腔體壓力讀值應在 755 ~ 765 Torr 範圍內（30 秒穩定）
- 記錄最終壓力值與洩壓完成時間
- 在維護日誌中標記「Chamber Vented」

**依賴關係：** ConfirmAtmPressure DEPENDS_ON OpenVentValve

---

## 4. 緊急洩壓（Interlock IL-E003 觸發）

當 Interlock IL-E003（Leak Rate > 0.5 mTorr/min）觸發時，系統自動執行：

1. 關閉所有氣體閥
2. 切斷 RF 電源
3. 啟動緊急洩壓序列（等同本程序步驟 2 ~ 4，但由系統自動控制）

操作員應立即進行安全確認，不得手動干預自動洩壓序列。

---

## 5. 相關文件

- SOP_Etch_001：蝕刻站壓力異常處置程序（步驟 3 引用本文件）
- SOP_Pump_002：真空泵浦狀態檢查程序（TurboVacuumPump 狀態定義）
- SOP_Vent_003 定義於文件庫 FabSOP_Library
