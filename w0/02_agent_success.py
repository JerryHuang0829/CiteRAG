"""W0-02 Agent 可用性實測（最關鍵）：量「程式碼編排單步 + format=json」下，
4B 選對工具 + 填對參數的端到端成功率，附 bootstrap 95% CI。

這個數決定 agent demo 野心（PLAN §5.3：≥80% 主秀 / 70-80% 勉強 / <70% 降級）。
別信 BFCL 通用數，要量你自己的工具集。
前置：Ollama 跑、qwen3:4b-instruct 已 pull。零額外套件。
輸出：成功率 + CI + w0_results_agent.json
"""
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")   # Windows cp950 無法編 ≥ 等字元，強制 UTF-8 輸出

# 共用統計原語（rule-of-three / 線性插值 percentile）；純 stdlib，不破壞 W0「零額外套件」
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from stats import bootstrap_ci   # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b-instruct"
REPS = 3   # 每題重複；題數 × REPS = bootstrap 樣本數

TOOLS_DESC = """你是設備維運助手。一次只選一個工具，只輸出 JSON。
可用工具：
1. search_manual(query: str)                查維修手冊
2. query_device(device_id: str)             查設備狀態
3. create_work_order(device_id: str, problem: str)  開工單
輸出格式（不要任何多餘文字）：{"tool": "<工具名>", "args": {...}}"""

CASES = [
    {"msg": "幫我查 3 號泵浦現在的狀態", "tool": "query_device", "args": {"device_id": "3"}},
    {"msg": "P-102 在漏油，幫我開一張工單", "tool": "create_work_order", "args": {"device_id": "P-102"}},
    {"msg": "換濾芯的步驟在手冊哪裡？", "tool": "search_manual", "args": {}},
    {"msg": "看一下 CNC-7 的運轉狀況", "tool": "query_device", "args": {"device_id": "CNC-7"}},
    {"msg": "幫 5 號機開單，問題是軸承異音", "tool": "create_work_order", "args": {"device_id": "5"}},
    {"msg": "手冊裡關於壓力錶校正怎麼說？", "tool": "search_manual", "args": {}},
    {"msg": "查設備編號 AHU-12", "tool": "query_device", "args": {"device_id": "AHU-12"}},
    {"msg": "馬達過熱，幫 M-9 報修", "tool": "create_work_order", "args": {"device_id": "M-9"}},
    {"msg": "聯軸器對心的標準是多少？查手冊", "tool": "search_manual", "args": {}},
    {"msg": "B 棟冰水主機 CH-2 現在跑得正常嗎", "tool": "query_device", "args": {"device_id": "CH-2"}},
]


def ask(msg):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": TOOLS_DESC},
                     {"role": "user", "content": msg}],
        "stream": False,
        "format": "json",   # grammar 約束：保證輸出合法 JSON
        "options": {"num_ctx": 2048, "temperature": 0},
        "keep_alive": "30m",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def grade(raw, case):
    try:
        obj = json.loads(raw)
    except Exception:
        return False, False
    tool_ok = obj.get("tool") == case["tool"]
    a = obj.get("args", {}) or {}
    args_ok = all(str(a.get(k, "")).strip() == str(v).strip() for k, v in case["args"].items())
    return True, (tool_ok and args_ok)


def main():
    per_case, json_valid = [], []
    unstable = 0
    print(f"Model={MODEL}  format=json ON  題數={len(CASES)} × REPS={REPS}\n")
    for case in CASES:
        reps_ok = []
        for _ in range(REPS):
            try:
                raw = ask(case["msg"])
            except Exception as e:
                print(f"  [ERROR ] {case['msg'][:22]}  {e}")
                json_valid.append(0)
                reps_ok.append(0)
                continue
            j, ok = grade(raw, case)
            json_valid.append(1 if j else 0)
            reps_ok.append(1 if ok else 0)
            mark = "OK" if ok else ("JSON壞" if not j else "選錯/填錯")
            print(f"  [{mark:>7}] {case['msg'][:22]}")
        per_case.append(sum(reps_ok) / len(reps_ok))
        if min(reps_ok) != max(reps_ok):
            unstable += 1

    n = len(per_case)   # 有效樣本 = 不重複工具題數（非題×REPS；temperature=0 下 REPS 非獨立）
    sr = sum(per_case) / n
    jr = sum(json_valid) / len(json_valid)
    lo, hi = bootstrap_ci(per_case)
    degen = " (0/1 退化→rule-of-three 下界)" if min(per_case) == max(per_case) else ""
    print("\n========== 結果 ==========")
    print(f"JSON 合法率   : {jr:.1%}  (grammar 應接近 100%)")
    print(f"端到端成功率  : {sr:.1%}   95% CI [{lo:.1%}, {hi:.1%}]{degen}  "
          f"有效 n={n}（REPS={REPS} 不一致 {unstable}）")
    if sr >= 0.80:
        verdict = "≥80% → agent 多步可當主秀"
    elif sr >= 0.70:
        verdict = "70-80% → 勉強進場，難步驟路由雲端"
    else:
        verdict = "<70% → 主秀降為 RAG+router，agent 縮為學習章節"
    print("判定(PLAN §5.3):", verdict)
    with open("w0_results_agent.json", "w", encoding="utf-8") as f:
        json.dump({"success_rate": sr, "ci": [lo, hi], "json_valid_rate": jr,
                   "n_effective": n, "reps": REPS}, f,
                  ensure_ascii=False, indent=2)
    print("→ 已存 w0_results_agent.json")


if __name__ == "__main__":
    main()
