"""P3-A：Agent 的 LangGraph 版（與手刻 `agent.py` 並存、行為對齊）。

同一組工具 + 同一個 `core.chat` + 同樣的引用/數值護欄，改用 LangGraph `StateGraph` 表達單步 tool-loop，
以取得標準圖結構 + tracing 生態。手刻版仍是部署預設；本版為 portfolio 展示 + 可觀測性（tracing 後續接）。

圖：agent（LLM 決策）─┬─(final)→ finalize ─┬─(未溯源數字，逼查證)→ agent
                      └─(tool/invalid)→ tools → agent          └─(done)→ END
langgraph 為選配依賴（requirements-langgraph.txt），不進核心/Docker。用法（rag/）：python agent_lg.py "..."
"""
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

import agent as A          # 複用 SYSTEM / TOOLS / MAX_TURNS / 工具函式（單一事實來源）
import core

sys.stdout.reconfigure(encoding="utf-8")


# ---- Langfuse tracing（選配、env-gated）：有 LANGFUSE_PUBLIC_KEY 才開；無則 no-op，零開銷/零警告、不影響測試 ----
def _load_langfuse_env():
    env = Path(__file__).resolve().parent.parent / ".env"
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LANGFUSE_") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


_load_langfuse_env()
# 有 LANGFUSE key 才開；但 pytest 下強制關（避免單元測試送 trace 污染儀表板/拖慢）
TRACE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and "pytest" not in sys.modules
if TRACE_ON:
    # conda 常把 SSL_CERT_FILE 指到 base env 不存在的 cacert → httpx/langfuse 會 FileNotFoundError（urllib 不受影響）；指到 certifi 修正
    _cert = os.environ.get("SSL_CERT_FILE")
    if not _cert or not os.path.exists(_cert):
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            os.environ.pop("SSL_CERT_FILE", None)
    from langfuse import observe as _observe
    def _obs(name):
        return _observe(name=name)
else:
    def _obs(name):
        return lambda fn: fn

_ASK_TOOL = ("你的答案出現未經工具查證的數字 {n}，可能是臆測。請務必先呼叫對應工具"
             "（stock_price 查股價、lookup_metric 查財報、compare 跨公司比較）取得真實數值，"
             "不要直接給 final；查不到就如實說無法確定。")
_REFUSE = "（無法由工具查證此問題的數值，請稍後再試或提供更明確的查詢條件。）"


class AgentState(TypedDict):
    messages: list
    called: set
    allowed_pages: set
    grounded: str
    corrections: int
    steps: int
    trace: list
    final: str            # None＝未完成（loop）；字串＝最終答案（→END）
    pending: dict         # 上一個 agent node 解出的 step
    pending_raw: str


@_obs("agent")
def _agent_node(state: AgentState) -> dict:
    raw = core.chat(state["messages"], as_json=True)
    try:
        step = json.loads(raw)
        if not isinstance(step, dict):
            step = {"action": "invalid"}
    except Exception:
        step = {"action": "invalid"}
    return {"pending": step, "pending_raw": raw, "steps": state["steps"] + 1}


def _route_after_agent(state: AgentState) -> str:
    if state["steps"] > A.MAX_TURNS:
        return "finalize"                       # 逾步 → 強制收尾
    return "finalize" if state["pending"].get("action") == "final" else "tools"


@_obs("tools")
def _tools_node(state: AgentState) -> dict:
    step, raw = state["pending"], state["pending_raw"]
    msgs = list(state["messages"])

    def _corr(text):                            # 驗證失敗：加更正訊息、回 agent 重試
        msgs.append({"role": "user", "content": text})
        return {"messages": msgs}

    if step.get("action") != "tool":
        return _corr("action 必須是 tool 或 final，請只輸出規定格式的 JSON。")
    name, args = step.get("tool"), (step.get("args") or {})
    if not isinstance(name, str):
        return _corr("tool 必須是工具名字串，請從清單選，或改用 final。")
    entry = A.TOOLS.get(name)
    if entry is None:
        return _corr(f"工具「{name}」不存在，請從清單選，或改用 final。")
    fn, required, optional = entry
    missing = [k for k in required if k not in args]
    if missing:
        return _corr(f"工具 {name} 的參數必須是 {required}，你缺 {missing}。請重發，或改用 final。")
    passed = required + [o for o in optional if o in args]
    bad = [k for k in passed if not isinstance(args[k], (str, int, float))]
    if bad:
        return _corr(f"工具 {name} 的參數 {bad} 必須是單一文字值（不可物件/陣列），請重發。")
    sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
    if sig in state["called"]:
        return _corr("已呼叫過相同工具與參數，請改用 final 給最終答案。")

    result = fn(**{k: args[k] for k in passed})
    allowed = set(state["allowed_pages"])
    if name == "search_filings":
        result, pages = result                  # (顯示字串, 命中頁碼集)
        allowed |= pages
    msgs.append({"role": "assistant", "content": raw})
    msgs.append({"role": "user", "content": f"工具 {name} 結果：{result}"})
    return {
        "messages": msgs,
        "called": state["called"] | {sig},
        "allowed_pages": allowed,
        "grounded": state["grounded"] + " " + str(result),
        "trace": state["trace"] + [{"tool": name, "args": args, "result": result}],
    }


@_obs("finalize")
def _finalize_node(state: AgentState) -> dict:
    step, forced = state["pending"], state["steps"] > A.MAX_TURNS
    if forced and step.get("action") != "final":       # 逾步強制收尾：逼一次 final
        msgs = state["messages"] + [{"role": "user", "content":
                                     "請『現在』只輸出 final 的 JSON 給最終答案，禁止再呼叫工具；資料不足就說查無。"}]
        try:
            step = json.loads(core.chat(msgs, as_json=True))
        except Exception:
            step = {}
    ans = core.verify_citations(step.get("answer", "") if isinstance(step, dict) else "",
                                state["allowed_pages"])[0]
    ungrounded = core.verify_numbers(ans, state["grounded"])
    if ungrounded and state["corrections"] < 1 and not forced:   # 未溯源數字 → 逼一次工具查證（loop 回 agent）
        msgs = state["messages"] + [{"role": "user", "content": _ASK_TOOL.format(n=ungrounded)}]
        return {"messages": msgs, "corrections": state["corrections"] + 1}   # 不設 final ＝ 續跑
    if ungrounded:
        ans = _REFUSE if not state["called"] else ans + "（註：部分數字未能由工具佐證，請以實際查詢為準。）"
    return {"final": ans or "（已盡力處理，但資料不足或無法完成此請求。）"}


def _route_after_finalize(state: AgentState) -> str:
    return END if state["final"] else "agent"


def _build():
    g = StateGraph(AgentState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", _tools_node)
    g.add_node("finalize", _finalize_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("finalize", _route_after_finalize, {"agent": "agent", END: END})
    return g.compile()


_GRAPH = _build()


@_obs("agent_lg.run")
def run(user_msg: str, history=None):
    """與 agent.run 同介面：回 (final_answer, trace)。history＝前幾輪 [{role,content}]。"""
    messages = [{"role": "system", "content": A.SYSTEM}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_msg})
    init: AgentState = {
        "messages": messages, "called": set(), "allowed_pages": set(),
        "grounded": user_msg or "",   # 數值溯源只含本輪題目＋工具結果（不含 history，防跨輪自我洗白）
        "corrections": 0, "steps": 0, "trace": [], "final": None,
        "pending": {}, "pending_raw": "",
    }
    out = _GRAPH.invoke(init, {"recursion_limit": 40})
    return out["final"] or "", out["trace"]


def main():
    if len(sys.argv) >= 2:
        final, trace = run(" ".join(sys.argv[1:]))
        for t in trace:
            print(f"[TOOL] {t['tool']}({t['args']}) -> {str(t['result'])[:90]}")
        print(f"\n[FINAL] {final}")
        if TRACE_ON:
            from langfuse import get_client
            get_client().flush()
            print("[trace] 已送出到 Langfuse")


if __name__ == "__main__":
    main()
