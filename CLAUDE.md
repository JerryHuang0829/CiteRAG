# CLAUDE.md — CiteRAG

本檔為 **CiteRAG repo 專屬** Claude Code 守則，每次在本資料夾的新 session 開場載入。
**權威計畫＝[PLAN.md](PLAN.md)（SSOT），動工前先讀。** 本檔只放 invariant 規則，不複製 PLAN 細節。

---

## 0. 專案一句話

學習導向的**通用 RAG + Agent 文件助手**，轉職 AI 應用工程師作品集。CPU-only / 零預算 / 單人週末。語料預設金融/投資公開文件、可抽換。

---

## 1. 硬體與環境約束（鐵律）

- 開發機 **i7-1260P CPU-only**（無 CUDA）、16GB RAM、**零預算**（雲端僅用免費額度）。
- **模型 / 容器 / cache 一律放 E:**（C: 僅剩 ~33GB）：`OLLAMA_MODELS` / `HF_HOME` / Docker disk image / conda envs 全指 E:。
- conda envs：`rag`（核心）、`rageval`（評測，隔離）、對照線（隔離）。**Ollama 跑 Windows host（非容器）**。

---

## 2. Agent 設計鐵律（可用 vs 不可用的分水嶺；詳 PLAN §5）

- **禁止讓 4B 自由多步 ReAct**（端到端成功率 ~16%）。
- 用**程式碼編排單步**：流程順序寫在 Python（你規劃），每步只問 LLM 一件小事（吃單步 ~90%）。
- 護欄全開：**grammar / `format=json`**（壞 JSON 歸零）＋ **Pydantic 驗證每步** ＋ `max_turns` ＋ 重複呼叫偵測 ＋ **引用機制驗證**（被引頁碼/chunk_id ∈ 本次檢索命中）。
- **router 逃生艙**：難步驟 / 視覺 env 開關外送免費 Gemini/Groq，本地 4B 只做簡單步。

---

## 3. W0 Benchmark Gate 鐵律（詳 PLAN §7）

- **未在本機實測的數字（tok/s、成功率、延遲…）一律不得寫入文件**（呼應「未實測就寫文件」教訓）。
- W0 量完才定 agent demo 野心；**專案本身是 GO，野心隨實測伸縮**。
- tok/s 一律標註單/雙通道 RAM；i7-1260P **無 AVX-512**（勿照抄誤稱）。

---

## 4. 語言 / 命名慣例

- 回覆繁體中文、技術術語英文（首次出現附括號中譯）；日期 ISO 8601。
- 敘述用 `NT$` / code 用 `TWD`；敘述用 `%` / code 存 decimal。
- 版本一律線性 `vN.M`；禁舊代號（Phase / Round / Sprint / V0.x）。
- code 註解只留 **WHY**，不留 dated / agent 痕跡；commit **不加** Co-Authored-By。

---

## 5. 不主動邊界（Plan 核准後）

| 可直接做 | 必先問 |
|---|---|
| Plan 列的檔案（stub / placeholder / docstring / `__init__.py`）、Plan 列的 memory | 邏輯 code（任何 `if` / loop / 計算）、新 skill / agent、Plan 外 memory、Plan 沒列但發現必要的檔案 |

---

## 6. 誠實補充（固定格式）

技術推薦 / 時間估算 / 架構決策後固定加：
- **Time estimate**：預估 + 不確定性 + 最大卡點
- **Failure modes**：哪種情況會壞 / 已知 tail risk
- **Assumption boundary**：所依賴的假設

---

## 7. Checkpoint-driven 執行

每個執行單元結束後 4 步：①自我 review（有無偏題）②回報做了什麼（檔案清單 + 關鍵 decision）③下一步是什麼 ④**等 user 核可，不自動繼續**。偵測偏題立即停下、不再改檔。

---

## 8. 檔案角色

| 檔案 | 用途 |
|---|---|
| `PLAN.md` | 計畫 SSOT（細節都在這） |
| `CLAUDE.md`（本檔）| 本 repo 守則（invariant 規則） |
| `HANDOFF.md` | 當前 session snapshot（每 session end 覆寫更新）|
| `README.md` | 對外公開介紹 |

---

最後更新：2026-06-22（v2.0 隨 PLAN 同步建立）
