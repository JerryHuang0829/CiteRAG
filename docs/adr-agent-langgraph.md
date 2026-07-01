# ADR：手刻 Agent vs LangGraph（兩者並存）

**狀態**：已決定 ｜ **對象**：`rag/agent.py`（手刻，prod 預設）vs `rag/agent_lg.py`（LangGraph，展示 + 可觀測性）

## 背景
Agent 要在**小模型/免費雲端**上可靠地跑「單步 function-calling loop」，並帶多層護欄（引用 / 數值溯源 / 重複偵測 / 卡住提早收尾 / 逾步強制收尾）。同時，履歷/面試需要業界標準框架的訊號與**可視化軌跡（tracing）**。

## 決策
**兩者並存**：手刻版是**部署預設**；LangGraph 版是**平行實作**，用於 tracing/observability 與 portfolio，並以**行為等價測試**證明兩者一致。

## 理由
| 面向 | 手刻 `agent.py`（prod） | LangGraph `agent_lg.py`（展示） |
|---|---|---|
| 護欄控制 | **細緻**（每個 edge 都自己寫，好 debug） | 用節點/條件路由表達，較抽象 |
| 依賴 | 零框架依賴（純 Python，部署精簡） | +langgraph/langchain-core（選配、不進 Docker） |
| 標準/生態 | 非標準（自訂） | **業界標準圖結構 + tracing 生態（Langfuse）** |
| 可觀測性 | 需自己加 | `@observe` 一接就有節點級 trace 儀表板 |
| 面試訊號 | 「懂原理」 | **「手刻也用框架、且用測試證明等價」＝判斷力** |

- **prod 用手刻**：對護欄要細緻控制、要最少依賴、要完全透明可 debug。
- **另建 LangGraph**：拿業界框架訊號 + tracing 可視化 + 證明設計是**框架可移植**的（不是綁死在手刻）。

## 一致性保證
`agent_lg` **複用** `agent.py` 的 `TOOLS`/`SYSTEM`/`MAX_TURNS` + `core.chat` + 引用/數值護欄（單一事實來源，不重複邏輯）。`tests/test_agent_lg.py` 用**與手刻版相同的 scripted 回應**斷言同樣行為（拒答 / grounded-accept / corrective-retry / 跨輪污染），mutation 驗護欄咬得住。

## 取捨與後果
- 好處：兩套並存 → 部署精簡（手刻）+ 面試有框架/儀表板可秀（LangGraph）。
- 代價：需維護兩份；緩解＝共用工具/護欄邏輯、以等價測試綁定行為。
- 誠實邊界：行為等價指「**測到的關鍵護欄行為一致**」，非 bit-identical（如 stuck 計數細節略異）。
