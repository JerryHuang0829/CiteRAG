# 安全設計（Security）— OWASP LLM Top-10 對映

金融場景 RAG/Agent 的安全防護與紅隊測試。核心原則：**defense-in-depth（多層）+ 可量測（block rate）+ 誠實邊界（不宣稱絕對安全）**。

## 威脅模型
對外 `/ask`、`/agent` 端點接受任意使用者輸入 → 檢索 → LLM 生成 → 回傳。攻擊面：惡意 prompt（注入/越獄）、誘導捏造、系統提示外洩、輸出夾帶 PII。語料為公開財報（本身無 PII），但使用者輸入與模型輸出仍可能帶個資。

## 防禦層 × OWASP LLM Top-10 對映

| OWASP | 風險 | 本專案防禦 | 在哪 | 如何驗 |
|---|---|---|---|---|
| **LLM01 Prompt Injection** | 「忽略指令…」劫持 | ① 輸入啟發式偵測 `detect_injection`（中英句式）② 系統提示約束「只依參考資料作答」③ 紅隊 canary 測試 | `security.py` / `build_prompt` / `redteam.py` | 紅隊 canary（要求輸出 PWNED/HACKED，resist=不出現） |
| **LLM06 Sensitive Info Disclosure** | 輸出夾帶個資 | **PII 輸出護欄**：偵測+遮罩台灣身分證（**內政部檢核碼**）、手機、信用卡（**Luhn**）、Email | `security.py` `redact_pii`，接在 `/ask`+`/agent` | 確定性單元測試（合成 PII 抓到、財報數字零誤判）+ 紅隊 PII 外洩測試 |
| **LLM07 System Prompt Leakage** | 洩漏系統提示/規則 | 系統提示不含機密；紅隊監測是否被 verbatim 吐出 | `redteam.py` | 紅隊：要求印出 system prompt，resist=不含提示原文標記 |
| **LLM09 Misinformation** | 捏造不存在的數字 | grounding（只依參考資料）+ 拒答（「查無此資訊」）+ **引用護欄** `verify_citations`（剝除範圍外頁碼） | `build_prompt` / `core.verify_citations` | 紅隊 jailbreak 捏造測試（resist=拒答）；eval-gate faithfulness |

## 如何測（兩層，接 CI）
- **確定性層**（不需 LLM，每次 push）：`tests/test_security.py` — PII 偵測/遮罩、injection 偵測、紅隊判定邏輯。跑在 `eval-ci` 的 `test` job（`pytest -m "not local"`）。
- **行為層**（需 LLM，nightly/手動）：`redteam.py` — N 個 OWASP 攻擊跑過完整防禦鏈，**確定性判定**（canary/關鍵字，非 LLM judge）量 **block rate**，低於 `SECURITY_SLO`（`slo.py`）即 exit 1。跑在 `generation-gate` job。目前 5/5 resist（block_rate 1.000）。

## 誠實邊界（面試會被戳，先講清楚）
- **不宣稱「injection-proof」**——沒有系統是。這裡是 **defense-in-depth + 量測 block rate**。
- `detect_injection` 是**啟發式**（pattern-based），可被繞過；它是第一層，不是唯一層。真正的注入抵抗主要來自系統提示 + 模型本身的 robustness（**量測而非假設**）。
- PII 偵測是 **regex + 檢核碼/Luhn**，**高精準（財報零誤判）但非 100% 召回**（罕見格式可能漏）。
- 攻擊集小（5）、判定用 canary/關鍵字，**證明「有防線且會持續量測」，非窮舉式保證**。
- 設計取捨：不套 US-centric 的 Presidio，改**手刻台灣在地 PII**（身分證檢核碼 / Luhn），精準且零重依賴。
