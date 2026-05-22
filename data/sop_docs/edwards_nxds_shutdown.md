# SOP_nXDS_003 — Edwards nXDS 乾式渦卷泵關機程序

> **文件類型：真實設備參考資料（Real Equipment Reference）**  
> 本文件依據 Edwards nXDS 系列乾式渦卷泵操作手冊第 4.10 節整理。

---

## 文件資訊

| 欄位 | 內容 |
|------|------|
| 文件編號 | SOP_nXDS_003 |
| 文件名稱 | Edwards nXDS 乾式渦卷泵關機程序 |
| 適用設備 | nXDS_ScrollPump, InletValve, IsolatorSwitch, ControlPanel |
| 版本 | Rev. 1.0 |
| 異常類型 | PumpShutdownError |

---

## 1. 適用範圍

本程序適用於 Edwards nXDS 乾式渦卷泵之計劃性正常關機作業。

---

## 2. 前置確認條件（PRECONDITION）

- **[PRECONDITION-SD01]** nXDS_ScrollPump 狀態 = RUNNING（泵浦正在運轉中）
- **[PRECONDITION-SD02]** 製程腔體已完成製程並處於 IDLE 狀態
- **[PRECONDITION-SD03]** 操作員已取得設備維護介面登入權限（Level-2）

---

## 3. 關機步驟

### 步驟 1：CloseInletBeforeStop（第一步驟）

緩慢關閉 InletValve 至全關，等待腔體洩壓至大氣壓，確認入口無製程氣體殘留。

**依賴關係：** CloseInletBeforeStop REQUIRES_STATUS nXDS_ScrollPump.RUNNING

---

### 步驟 2：StopPump（下一步驟）

按下 ControlPanel 紅色停止鈕，泵浦開始減速。
禁止在泵浦轉速 > 100 RPM 時斷電。

**依賴關係：** StopPump DEPENDS_ON CloseInletBeforeStop

---

### 步驟 3：WaitForDeceleration（下一步驟）

等待泵浦轉速降至 0 RPM（約需 30–60 秒）。
確認 ControlPanel 狀態燈顯示 STOPPED。期間禁止搬動或維修泵浦。

**依賴關係：** WaitForDeceleration DEPENDS_ON StopPump

---

### 步驟 4：SwitchOffIsolator（下一步驟）

將 IsolatorSwitch 旋轉至 OFF，確認 ControlPanel 電源指示燈熄滅。
若進行維護作業，加裝 Lock-Out/Tag-Out（LOTO）。

**依賴關係：** SwitchOffIsolator DEPENDS_ON WaitForDeceleration

---

### 步驟 5：LogShutdown（下一步驟）

記錄關機時間與累積運轉小時數至 PM 日誌。
若因異常關機，填寫 MAR 單並更新設備狀態至 EHS 系統。

**依賴關係：** LogShutdown DEPENDS_ON SwitchOffIsolator

---

## 4. 設備狀態定義

| 狀態代碼 | 含義 |
|---------|------|
| RUNNING | 正常運轉，達到額定轉速 1450 RPM |
| STOPPING | 減速停止中 |
| STOPPED | 完全停止，轉速 = 0 RPM |

---

## 5. 相關文件

- SOP_nXDS_001：Edwards nXDS 乾式渦卷泵啟動程序
- SOP_nXDS_002：Edwards nXDS 泵浦異常故障排查程序
- SOP_nXDS_003 定義於文件庫 FabSOP_Library

**跨文件依賴說明：**
本文件（SOP_nXDS_003）步驟 1（CloseInletBeforeStop）依賴 SOP_nXDS_001
所定義之 nXDS_ScrollPump RUNNING 狀態標準（額定轉速 1450 RPM）。
