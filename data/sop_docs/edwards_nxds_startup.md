# SOP_nXDS_001 — Edwards nXDS 乾式渦卷泵啟動程序

> **文件類型：真實設備參考資料（Real Equipment Reference）**  
> 本文件依據 Edwards nXDS 系列乾式渦卷泵操作手冊第 4.7 節整理。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_nXDS_001 |
| 文件名稱 | Edwards nXDS 乾式渦卷泵啟動程序 |
| 適用設備 | nXDS_ScrollPump, InletValve, IsolatorSwitch, ControlPanel |
| 版本 | Rev. 1.0 |
| 異常類型 | PumpStartupFailure |

---

## 1. 適用範圍

本程序適用於 Edwards nXDS 乾式渦卷泵之正常啟動作業。

---

## 2. 前置確認條件（PRECONDITION）

- **[PRECONDITION-NX01]** nXDS_ScrollPump 電源已接至正確電壓（100–230 VAC）
- **[PRECONDITION-NX02]** InletValve 狀態 = CLOSED
- **[PRECONDITION-NX03]** IsolatorSwitch 狀態 = OFF
- **[PRECONDITION-NX04]** 排氣管路已連接且背壓 ≤ 0.5 bar gauge

---

## 3. 啟動步驟

### 步驟 1：ConnectInlet（第一步驟）

確認真空入口法蘭已正確連接、管路連接件鎖緊。

**依賴關係：** ConnectInlet REQUIRES_STATUS InletValve.CLOSED

---

### 步驟 2：SwitchOnIsolator（下一步驟）

將 IsolatorSwitch 旋轉至 ON，確認 ControlPanel 電源指示燈亮起並完成自診（3 秒）。

**依賴關係：** SwitchOnIsolator DEPENDS_ON ConnectInlet

---

### 步驟 3：StartPump（下一步驟）

按下 ControlPanel 綠色啟動鈕，泵浦加速至額定轉速（1450 RPM）。
啟動瞬間電流 ≤ 6 A，穩態 ≤ 2.5 A。

**依賴關係：** StartPump DEPENDS_ON SwitchOnIsolator

---

### 步驟 4：VerifyNormalSpeed（下一步驟）

60 秒內確認泵浦達到額定轉速，ControlPanel 狀態燈顯示 RUNNING（綠燈）。
若 60 秒後未達轉速，執行故障排查（參考 SOP_nXDS_002）。

**依賴關係：** VerifyNormalSpeed DEPENDS_ON StartPump
**依賴關係：** VerifyNormalSpeed REQUIRES_STATUS nXDS_ScrollPump.STARTING

---

### 步驟 5：OpenInletAndMonitor（下一步驟）

緩慢開啟 InletValve 至全開，確認系統壓力下降，記錄啟動時間至操作日誌。

**依賴關係：** OpenInletAndMonitor DEPENDS_ON VerifyNormalSpeed

---

## 4. 設備狀態定義

| 狀態代碼 | 含義 |
|---------|------|
| RUNNING | 正常運轉，達到額定轉速 1450 RPM |
| STARTING | 啟動加速中，未達額定轉速 |
| FAULT | 故障停機，ControlPanel 服務指示燈閃爍 |

---

## 5. Interlock 條件

| Interlock ID | 觸發條件 | 聯鎖動作 |
|-------------|----------|----------|
| IL-NX001 | nXDS_ScrollPump 溫度 > 70°C（1 次閃爍） | 發出溫度警告，記錄事件 |
| IL-NX002 | nXDS_ScrollPump 溫度 > 85°C（2 次閃爍） | 自動停機，觸發 FAULT 狀態 |
| IL-NX003 | 馬達電流過載（3 次閃爍） | 自動停機，需檢查入口阻塞 |

**聯鎖關係：** nXDS_ScrollPump INTERLOCK_WITH ControlPanel

---

## 6. 相關文件

- SOP_nXDS_002：Edwards nXDS 泵浦異常故障排查程序
- SOP_nXDS_003：Edwards nXDS 泵浦關機程序
- SOP_nXDS_001 定義於文件庫 FabSOP_Library
