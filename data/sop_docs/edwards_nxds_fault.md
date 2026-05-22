# SOP_nXDS_002 — Edwards nXDS 乾式渦卷泵異常故障排查程序

> **文件類型：真實設備參考資料（Real Equipment Reference）**  
> 本文件依據 Edwards nXDS 系列乾式渦卷泵操作手冊第 5.11 節整理。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_nXDS_002 |
| 文件名稱 | Edwards nXDS 乾式渦卷泵異常故障排查程序 |
| 適用設備 | nXDS_ScrollPump, ControlPanel, InletValve, ExhaustLine |
| 版本 | Rev. 1.0 |
| 異常類型 | PumpStartupFailure, PumpPoorPerformance, PumpOverheat |

---

## 1. 適用範圍

本程序適用於 Edwards nXDS 乾式渦卷泵無法啟動、真空效能不佳、溫度過高之故障排查。

---

## 2. 前置確認條件（PRECONDITION）

- **[PRECONDITION-FT01]** nXDS_ScrollPump 狀態 = STOPPED 或 FAULT
- **[PRECONDITION-FT02]** IsolatorSwitch 狀態 = OFF（進行實體檢查前必須斷電）
- **[PRECONDITION-FT03]** 操作員已完成 Lock-Out/Tag-Out（LOTO）程序

---

## 3. 故障排查步驟

### 步驟 1：ReadFlashCode（第一步驟）

讀取 ControlPanel 服務指示燈閃爍次數：1 次 = 溫度警告；2 次 = 溫度故障自動停機；3 次 = 馬達過載自動停機。

**依賴關係：** ReadFlashCode REQUIRES_STATUS ControlPanel.ACCESSIBLE

---

### 步驟 2：VerifyStartupFault（下一步驟）

若泵浦無法啟動：確認電源電壓（額定 ±10%），確認入口無異物阻塞（對應 3 次閃爍），
確認 60 秒內達到額定轉速（1450 RPM，標準見 SOP_nXDS_001），否則通報 Equipment Engineer。

**依賴關係：** VerifyStartupFault DEPENDS_ON ReadFlashCode

---

### 步驟 3：CheckVacuumPerformance（下一步驟）

若啟動後無法達到基礎壓力：關閉 InletValve，監測入口壓力回升速率。
正常 ≤ 0.1 mTorr/min；異常 > 0.1 mTorr/min 表示泵浦本體或管路洩漏。
確認排氣管路背壓 ≤ 0.5 bar gauge（背壓過高降低抽氣效能）。

**依賴關係：** CheckVacuumPerformance DEPENDS_ON VerifyStartupFault

---

### 步驟 4：InvestigateOverheat（下一步驟）

若出現溫度警告（1 次閃）或溫度故障（2 次閃）：
確認環境溫度 ≤ 40°C，確認通風孔未堵塞，確認入口氣體溫度 ≤ 60°C。
2 次閃爍（溫度故障）須待泵浦冷卻後方可重啟。

**依賴關係：** InvestigateOverheat DEPENDS_ON CheckVacuumPerformance
**依賴關係：** InvestigateOverheat REQUIRES_STATUS nXDS_ScrollPump.FAULT

---

### 步驟 5：LogAndEscalate（下一步驟）

記錄閃爍代碼、故障時間、排查結果至 PM 日誌。
若故障可排除：填寫 MAR 單後依 SOP_nXDS_001 重新啟動。
若涉及泵浦本體損壞：通報 Edwards 原廠服務工程師，禁止自行拆解。

**依賴關係：** LogAndEscalate DEPENDS_ON InvestigateOverheat

---

## 4. Interlock 條件

| Interlock ID | 觸發條件 | 聯鎖動作 |
|-------------|----------|----------|
| IL-NX004 | 入口壓力回升速率 > 0.1 mTorr/min（InletValve 關閉狀態） | 發出洩漏警告，通知 Process Engineer |
| IL-NX005 | 排氣背壓 > 0.5 bar gauge | 暫停製程抽氣，防止泵浦過載 |

**聯鎖關係：** nXDS_ScrollPump INTERLOCK_WITH ExhaustLine

---

## 5. 相關文件

- SOP_nXDS_001：Edwards nXDS 乾式渦卷泵啟動程序
- SOP_nXDS_003：Edwards nXDS 乾式渦卷泵關機程序
- SOP_nXDS_002 定義於文件庫 FabSOP_Library

**跨文件依賴說明：**
本文件步驟 2（VerifyStartupFault）中，nXDS_ScrollPump 正常轉速標準（1450 RPM）
定義於 SOP_nXDS_001 第 4 節（VerifyNormalSpeed）。
