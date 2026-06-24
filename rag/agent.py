"""W3 Agent：程式碼編排的「單步」function-calling loop（非自由 ReAct）。

每一步只讓 4B 做一個決定（呼叫一個工具 / 給最終答案），用 format=json 約束輸出 +
程式驗證工具名與參數 + max_turns 防迴圈 + 重複呼叫偵測。
利用 W0 實證的「單步高可靠」而非「自由多步 ~16%」。

工具：search_filings(RAG 查財報/法說會) / lookup_metric(mock 查財務指標) / create_note(mock 建追蹤筆記)。
用法（rag/）：python agent.py "鴻海 2022 全年 EPS 多少？幫我記一筆下季追蹤毛利率"
"""
import json
import sys

import core

sys.stdout.reconfigure(encoding="utf-8")

# ---- mock 後端（demo 用）----
_METRICS_DB = {
    ("鴻海", "EPS"): "2022 全年 EPS 10.21 元（15 年新高）",
    ("鴻海", "營收"): "2022 全年合併營收 6.6 兆元（年增逾一成）",
    ("鴻海", "毛利率"): "2022 第四季毛利率 5.66%",
    ("鴻海", "現金"): "2022 帳上現金約 1.06 兆元",
}
_NOTES = []


def search_filings(query: str) -> str:
    hits = core.retrieve(query, k=3)
    # 把可靠命中頁碼掛在函式上供引用護欄取用（不從顯示字串 re-parse，避免內文 p.N 污染白名單）
    search_filings.last_pages = {h["page"] for h in hits}
    if not hits:
        return "文件查無相關內容。"
    return " ｜ ".join(f"(p.{h['page']}) {h['text'][:120]}" for h in hits)


search_filings.last_pages = set()


def lookup_metric(company: str, metric: str) -> str:
    return _METRICS_DB.get((company, metric), f"指標庫查無「{company}／{metric}」。")


def create_note(topic: str, content: str) -> str:
    nid = f"NOTE-{len(_NOTES) + 1:03d}"
    _NOTES.append({"id": nid, "topic": topic, "content": content})
    return f"已建立追蹤筆記 {nid}（主題：{topic}）。"


TOOLS = {
    "search_filings": (search_filings, ["query"]),
    "lookup_metric": (lookup_metric, ["company", "metric"]),
    "create_note": (create_note, ["topic", "content"]),
}

SYSTEM = """你是金融文件研究 Agent。每一步只輸出「一個」JSON，二選一：
1) 呼叫工具：{"action":"tool","tool":"<工具名>","args":{...}}
2) 給最終答案：{"action":"final","answer":"<繁體中文答案，引用文件頁碼如 (p.3)>"}

可用工具：
- search_filings(query): 查財報/法說會等文件原文，回傳帶頁碼段落（要看原文、解釋、出處時用）
- lookup_metric(company, metric): 快速查單一財務指標數字（company 如「鴻海」；metric 如 EPS／營收／毛利率／現金）
- create_note(topic, content): 建立一筆研究追蹤筆記

規則：一次只做一件事；缺資料先呼叫工具拿到後再繼續；資料齊了才給 final。
若使用者要求「記下／儲存／記成筆記」任何內容，務必實際呼叫 create_note 建立，不可只在 final 聲稱已記。
只輸出 JSON、不要多餘文字。"""

MAX_TURNS = 6


def _events(user_msg: str):
    """單一事件流（CLI 與 UI 共用）。yield:
    ("tool", name, args, result) / ("final", answer)。
    護欄：連續 2 步無進展即提早收尾；迴圈結束一律強制逼出 final，永不空手回。"""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]
    called = set()
    allowed_pages = set()   # 本軌跡 search_filings 命中的頁碼（用於剝除 final 的幻覺頁碼）
    stuck = 0   # 連續「沒成功執行工具/沒給 final」的步數
    for _ in range(MAX_TURNS):
        if stuck >= 2:
            break   # 卡住 → 跳去強制收尾，不浪費剩餘步數
        raw = core.chat(messages, as_json=True)
        try:
            step = json.loads(raw)
        except Exception:
            stuck += 1
            messages.append({"role": "user", "content": "請只輸出規定格式的 JSON。"})
            continue
        if not isinstance(step, dict):
            stuck += 1
            messages.append({"role": "user", "content": "請輸出 JSON 物件（{...}），不是陣列或單一值。"})
            continue

        if step.get("action") == "final":
            yield ("final", core.verify_citations(step.get("answer", ""), allowed_pages)[0])
            return

        if step.get("action") == "tool":
            name, args = step.get("tool"), (step.get("args") or {})
            if not isinstance(name, str):
                stuck += 1
                messages.append({"role": "user", "content": "tool 必須是工具名字串，請從清單選，或改用 final。"})
                continue
            entry = TOOLS.get(name)
            if entry is None:
                stuck += 1
                messages.append({"role": "user", "content": f"工具「{name}」不存在，請從清單選，或改用 final。"})
                continue
            fn, required = entry
            missing = [k for k in required if k not in args]
            if missing:
                stuck += 1
                messages.append({"role": "user",
                                 "content": f"工具 {name} 的參數必須是 {required}，你缺 {missing}。"
                                            f"請用正確的參數名重發，或改用 final 給最終答案。"})
                continue
            bad = [k for k in required if not isinstance(args[k], (str, int, float))]
            if bad:
                stuck += 1
                messages.append({"role": "user",
                                 "content": f"工具 {name} 的參數 {bad} 必須是單一文字值（不可是物件或陣列），請重發。"})
                continue
            # 4B 在 format=json 下仍可能吐巢狀 args；用 JSON 字串當去重 key，天然可 hash 且容忍巢狀
            sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if sig in called:
                stuck += 1
                messages.append({"role": "user", "content": "已呼叫過相同工具與參數，請改用 final 給最終答案。"})
                continue
            called.add(sig)
            stuck = 0
            result = fn(**{k: args[k] for k in required})
            if name == "search_filings":
                allowed_pages |= search_filings.last_pages
            yield ("tool", name, args, result)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"工具 {name} 結果：{result}"})
            continue

        stuck += 1
        messages.append({"role": "user", "content": "action 必須是 tool 或 final。"})

    # 強制收尾：用目前資訊逼出 final，永不回空的「達上限」
    messages.append({"role": "user",
                     "content": "請『現在』只輸出 final 的 JSON 給最終答案，禁止再呼叫工具；"
                                "若資料不足就在 answer 直接說明查無。"})
    try:
        step = json.loads(core.chat(messages, as_json=True))
        ans = step.get("answer") if isinstance(step, dict) else None
        if ans:
            yield ("final", core.verify_citations(ans, allowed_pages)[0])
            return
    except Exception:
        pass
    yield ("final", "（已盡力處理，但資料不足或無法完成此請求。）")


def run(user_msg: str, verbose: bool = True):
    """CLI / 程式呼叫：回 (final_answer, trace)。"""
    trace = []
    for ev in _events(user_msg):
        if ev[0] == "tool":
            _, name, args, result = ev
            trace.append({"tool": name, "args": args, "result": result})
            if verbose:
                print(f"[TOOL] {name}({args}) -> {result[:90]}")
        elif ev[0] == "final":
            if verbose:
                print(f"\n[FINAL] {ev[1]}")
            return ev[1], trace
    return "", trace


def run_iter(user_msg: str):
    """UI 用：yield 逐步累積的 markdown（顯示工具軌跡 + 最終答案）。"""
    log = ""
    for ev in _events(user_msg):
        if ev[0] == "tool":
            _, name, args, result = ev
            log += f"🔧 **{name}**（{args}）\n\n→ {result}\n\n"
            yield log
        elif ev[0] == "final":
            log += f"\n---\n✅ **最終答案**\n\n{ev[1]}"
            yield log


def main():
    if len(sys.argv) < 2:
        print('用法：python agent.py "你的需求"')
        return
    try:
        run(" ".join(sys.argv[1:]))
    except (core.OllamaError, FileNotFoundError) as e:
        print(f"\n[錯誤] {e}")


if __name__ == "__main__":
    main()
