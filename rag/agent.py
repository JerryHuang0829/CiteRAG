"""W3 Agent：程式碼編排的「單步」function-calling loop（非自由 ReAct）。

每一步只讓 4B 做一個決定（呼叫一個工具 / 給最終答案），用 format=json 約束輸出 +
程式驗證工具名與參數 + max_turns 防迴圈 + 重複呼叫偵測。
利用 W0 實證的「單步高可靠」而非「自由多步 ~16%」。

工具：search_filings(RAG 查文字) / lookup_metric(FinMind 查結構化財務數字) / create_note(mock 建筆記)。
用法（rag/）：python agent.py "鴻海 2022 全年 EPS 多少？幫我記一筆下季追蹤毛利率"
"""
import json
import sys

import core
import findata

sys.stdout.reconfigure(encoding="utf-8")

_NOTES = []


def search_filings(query: str) -> str:
    hits = core.retrieve(query, k=3)
    # 把可靠命中頁碼掛在函式上供引用護欄取用（不從顯示字串 re-parse，避免內文 p.N 污染白名單）
    search_filings.last_pages = {h["page"] for h in hits}
    if not hits:
        return "文件查無相關內容。"
    return " ｜ ".join(f"(p.{h['page']}) {h['text'][:120]}" for h in hits)


search_filings.last_pages = set()


def lookup_metric(company: str, metric: str, year=None) -> str:
    # 真實結構化查詢（FinMind 財報）：數字/事件走這＝精確、可彙總、零幻覺（對照 RAG 走文字）
    return findata.lookup(company, metric, year)


def compare(metric: str, companies=None, year=None, threshold=None) -> str:
    # 多家公司彙總/排名/篩選（「誰最高/比較/哪些公司>X」）——RAG 做不到、資料庫才能做
    return findata.compare(metric, companies, year, threshold)


def stock_price(company: str) -> str:
    # 即時/近期股價（每天變、文件裡不會有）——只有資料庫能給
    return findata.price(company)


def create_note(topic: str, content: str) -> str:
    nid = f"NOTE-{len(_NOTES) + 1:03d}"
    _NOTES.append({"id": nid, "topic": topic, "content": content})
    return f"已建立追蹤筆記 {nid}（主題：{topic}）。"


TOOLS = {  # (函式, 必填參數, 選填參數)
    "search_filings": (search_filings, ["query"], []),
    "lookup_metric": (lookup_metric, ["company", "metric"], ["year"]),
    "compare": (compare, ["metric"], ["companies", "year", "threshold"]),
    "stock_price": (stock_price, ["company"], []),
    "create_note": (create_note, ["topic", "content"], []),
}

SYSTEM = """你是金融文件研究 Agent。每一步只輸出「一個」JSON，二選一：
1) 呼叫工具：{"action":"tool","tool":"<工具名>","args":{...}}
2) 給最終答案：{"action":"final","answer":"<繁體中文答案，引用文件頁碼如 (p.3)>"}

可用工具：
- search_filings(query): 查財報/法說會等文件「原文文字」，回傳帶頁碼段落（要解釋、原文、出處、質性內容時用）
- lookup_metric(company, metric, year): 查「單一公司精確財務數字」（FinMind；company 如「台積電」、metric 如 EPS／營收／毛利率／淨利、year 如 2023）。問單一數字優先用這個＝精確不幻覺
- compare(metric, companies, year, threshold): 「多家比較/排名/篩選」（companies 逗號分隔如「台積電,聯發科」，省略＝主要大公司；threshold 如「>30」）。問「誰最高／比較／哪些公司…」時用
- stock_price(company): 查「即時/近期股價」（最新收盤、當日漲跌、近一年漲跌幅）。問股價/漲跌時用＝財報沒有、只能查資料庫
- create_note(topic, content): 建立一筆研究追蹤筆記

規則：一次只做一件事；缺資料先呼叫工具拿到後再繼續；資料齊了才給 final。
若使用者要求「記下／儲存／記成筆記」任何內容，務必實際呼叫 create_note 建立，不可只在 final 聲稱已記。
若使用者用代名詞（它／那家／這個）或省略主詞，依前文對話判斷指的是哪家公司／哪個主題再查。
只輸出 JSON、不要多餘文字。"""

MAX_TURNS = 6


def _events(user_msg: str, history=None):
    """單一事件流（CLI 與 UI 共用）。yield:
    ("tool", name, args, result) / ("final", answer)。
    history＝前幾輪 [{role,content}]（讓「它/那家」等代名詞可解析），不含本輪工具軌跡。
    護欄：連續 2 步無進展即提早收尾；迴圈結束一律強制逼出 final，永不空手回。"""
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_msg})
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
            fn, required, optional = entry
            missing = [k for k in required if k not in args]
            if missing:
                stuck += 1
                messages.append({"role": "user",
                                 "content": f"工具 {name} 的參數必須是 {required}，你缺 {missing}。"
                                            f"請用正確的參數名重發，或改用 final 給最終答案。"})
                continue
            passed = required + [o for o in optional if o in args]
            bad = [k for k in passed if not isinstance(args[k], (str, int, float))]
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
            result = fn(**{k: args[k] for k in passed})
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


def run(user_msg: str, history=None, verbose: bool = True):
    """CLI / 程式呼叫：回 (final_answer, trace)。history＝前幾輪 [{role,content}]。"""
    trace = []
    for ev in _events(user_msg, history):
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


def run_iter(user_msg: str, history=None):
    """UI 用：yield 逐步累積的 markdown（顯示工具軌跡 + 最終答案）。history＝前幾輪。"""
    log = ""
    for ev in _events(user_msg, history):
        if ev[0] == "tool":
            _, name, args, result = ev
            log += f"🔧 **{name}**（{args}）\n\n→ {result}\n\n"
            yield log
        elif ev[0] == "final":
            log += f"\n---\n✅ **最終答案**\n\n{ev[1]}"
            yield log


def main():
    try:
        if len(sys.argv) >= 2:          # 一次性：python agent.py "你的需求"
            run(" ".join(sys.argv[1:]))
            return
        # 多輪互動：python agent.py（空白行離開）。可用代名詞接續，如「那它營收呢」。
        print('多輪對話模式（直接 Enter 離開）。例：先問「台積電 2023 EPS」，再問「那它營收呢」。')
        history = []
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            final, _ = run(q, history)
            history += [{"role": "user", "content": q},
                        {"role": "assistant", "content": final}]
            history = history[-6:]       # 只保留最近 3 輪，控制 context（4B num_ctx=4096）
    except (core.OllamaError, FileNotFoundError) as e:
        print(f"\n[錯誤] {e}")


if __name__ == "__main__":
    main()
