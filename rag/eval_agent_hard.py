"""W3 agent 難情境壓測：故意用會踩雷的情境找 agent 極限（不是要 100%，是要看它在哪裡壞）。

涵蓋：拒答(不在資料)、fallback(指標庫無→改查文件)、工具選擇(要原文/頁→search)、
長鏈(多查+建筆記)、不過度動作(純記筆記不亂查)、部分可用、模糊需求、條件式。
用法（rag/）：python eval_agent_hard.py   （多步×CPU，數十分鐘）
"""
import agent
from eval_retrieval import norm
from stats import bootstrap_ci

REPS = 2   # temperature=0 下為穩定度 sanity-check，非獨立樣本；有效 n 以「不重複情境數」計
REFUSE = ["查無", "查不到", "無法", "沒有", "未提供", "未提及", "未明確", "未揭露",
          "未列", "找不到", "無資料", "無相關", "不到", "無此"]

# req, 可選: refuse / contains(任一) / must(全要) / must_not(全不可)
SCENARIOS = [
    {"req": "台積電 2023 年的 EPS 是多少？", "refuse": True,
     "說明": "不在資料→應拒答不可亂編"},
    {"req": "鴻海 2022 全年資本支出是多少？", "contains": ["979"],
     "說明": "指標庫無此項→需 fallback 去 search_filings 找"},
    {"req": "鴻海 2022 EPS 的原文怎麼說？在第幾頁？", "contains": ["p."], "must": ["search_filings"],
     "說明": "要原文/頁碼→須用 search_filings 而非 lookup_metric"},
    {"req": "幫我查鴻海 2022 的 EPS 和營收，並把兩個數字記成一筆筆記",
     "contains": ["10.21"], "must": ["create_note"],
     "說明": "長鏈：多次查 + 建筆記"},
    {"req": "幫我記一筆：下週要看鴻海法說會",
     "must": ["create_note"], "must_not": ["lookup_metric", "search_filings"],
     "說明": "純記筆記→不該多餘查詢"},
    {"req": "查鴻海和台積電的 2022 EPS 各是多少", "contains": ["10.21"],
     "說明": "部分可用：鴻海有、台積電無"},
    {"req": "鴻海賺多少錢？", "contains": ["10.21", "6.6", "兆", "億"],
     "說明": "模糊需求→給相關財務數字即可"},
    {"req": "查興櫃市場成立時間，如果是 2002 年才幫我記一筆筆記",
     "contains": ["91", "2002"], "must": ["create_note"],
     "說明": "條件式推理（4B 易失敗）"},
]


def score(sc, final, called):
    ok = True
    if sc.get("refuse"):
        ok = any(w in final for w in REFUSE)
    if sc.get("contains"):
        ok = ok and any(norm(c) in norm(final) for c in sc["contains"])
    if sc.get("must"):
        ok = ok and all(t in called for t in sc["must"])
    if sc.get("must_not"):
        ok = ok and all(t not in called for t in sc["must_not"])
    return 1 if ok else 0


def main():
    per_scenario = []
    unstable = 0
    print(f"{'結果':>4}  情境 | 工具 | final 摘要")
    for sc in SCENARIOS:
        reps_ok = []
        for _ in range(REPS):
            final, trace = agent.run(sc["req"], verbose=False)
            called = [t["tool"] for t in trace]
            ok = score(sc, final, called)
            reps_ok.append(ok)
            print(f"{('OK' if ok else '××'):>4}  {sc['req'][:20]} | {called} | {final[:46]}")
        per_scenario.append(sum(reps_ok) / len(reps_ok))
        if min(reps_ok) != max(reps_ok):
            unstable += 1
    n = len(per_scenario)   # 有效樣本 = 不重複情境數（非情境×REPS）
    lo, hi = bootstrap_ci(per_scenario)
    degen = " (0/1 退化→rule-of-three)" if min(per_scenario) == max(per_scenario) else ""
    print(f"\nhard task success: {sum(per_scenario)/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{degen}  "
          f"(有效 n={n} 情境；REPS={REPS} 不一致 {unstable})")
    print("註：難情境（拒答/fallback/長鏈/條件式/模糊），失敗是預期內、用來定位極限與『該救哪裡』。"
          "有效 n 以不重複情境數計（temperature=0 下 REPS 非獨立樣本）。")


if __name__ == "__main__":
    main()
