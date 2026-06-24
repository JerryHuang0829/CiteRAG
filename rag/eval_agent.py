"""W3 agent trajectory eval：多情境量 agent 的端到端表現（像 W0 但針對多步）。

每情境跑完整 agent（多步、含工具呼叫），量：
- task success：最終答案是否含正解字串（outcome-based，與用哪個工具無關）
- note 處理正確：該建筆記有建 / 不該建沒建
- spurious note：不該建卻建了（false action）
- 平均步數
task success 附 bootstrap 95% CI。agent eval 比 retrieval eval 稀有，是面試亮點。
用法（rag/）：python eval_agent.py   （多步 × CPU，數十分鐘）
"""
import agent
from eval_retrieval import norm
from stats import bootstrap_ci

REPS = 2   # temperature=0 下為穩定度 sanity-check，非獨立樣本；有效 n 以「不重複情境數」計

# (需求, 最終答案應含, 是否需要建筆記)
SCENARIOS = [
    ("鴻海 2022 全年 EPS 是多少？", "10.21", False),
    ("鴻海 2022 全年合併營收是多少？", "6.6", False),
    ("鴻海 2022 第四季毛利率是多少？", "5.66", False),
    ("鴻海帳上現金有多少？", "1.06", False),
    ("興櫃股票市場是何時成立的？", "91", False),
    ("鴻海 2022 EPS 多少？順便幫我記一筆追蹤下季毛利率", "10.21", True),
    ("幫我把『鴻海 2022 EPS 是 10.21 元』記成一筆追蹤筆記", "NOTE", True),
    ("查興櫃市場何時成立，並把結論記成一筆筆記", "91", True),
]


def main():
    per_scenario, note_ok, spurious, turns = [], [], [], []
    unstable = 0   # REPS 間結果不一致的情境數（temperature=0 下理應為 0）
    print(f"{'task':>4} {'note':>4} {'步數':>4}  情境 / 用到的工具")
    for req, contains, needs_note in SCENARIOS:
        reps_ok = []
        for _ in range(REPS):
            final, trace = agent.run(req, verbose=False)
            called = [t["tool"] for t in trace]
            ok = 1 if norm(contains) in norm(final) else 0
            created = "create_note" in called
            reps_ok.append(ok)
            note_ok.append(1 if created == needs_note else 0)
            spurious.append(1 if (created and not needs_note) else 0)
            turns.append(len(trace))
            print(f"{('OK' if ok else 'x'):>4} {('OK' if created == needs_note else 'x'):>4} "
                  f"{len(trace):>4}  {req[:22]}  {called}")
        per_scenario.append(sum(reps_ok) / len(reps_ok))
        if min(reps_ok) != max(reps_ok):
            unstable += 1

    def rate(xs):
        return sum(xs) / len(xs)

    n = len(per_scenario)   # 有效樣本 = 不重複情境數（非情境×REPS）
    tlo, thi = bootstrap_ci(per_scenario)
    degen = " (0/1 退化→rule-of-three 下界)" if min(per_scenario) == max(per_scenario) else ""
    print("\n========== 聚合 ==========")
    print(f"task success       : {rate(per_scenario):.3f}  95% CI [{tlo:.3f}, {thi:.3f}]{degen}  (有效 n={n} 情境)")
    print(f"note 處理正確       : {rate(note_ok):.3f}  （該建有建 / 不該建沒建）")
    print(f"spurious note(誤建) : {rate(spurious):.3f}  （false action，越低越好）")
    print(f"平均步數           : {sum(turns) / len(turns):.1f}")
    print(f"REPS={REPS} 跨重複不一致情境：{unstable}（temperature=0 下理應為 0；REPS 僅穩定度 sanity-check，未計入有效 n）")
    print("\n註：有效 n 以不重複情境數計（temperature=0 下 REPS 非獨立樣本，不灌大 n）；全 0/1 時 CI 改報 "
          "rule-of-three 下界而非假的 [1,1]。多步 task success 本就低於 W0 單步，低門檻情境靠難步驟路由雲端救（hook）。")


if __name__ == "__main__":
    main()
