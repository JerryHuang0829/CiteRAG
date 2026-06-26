"""multi-turn golden 評測：量化多輪對話記憶（實體/意圖解析 ＋ 答案命中），含 ON vs OFF ablation。

每段對話逐輪帶累積 history 跑（ON）；對每個 follow-up 句再以空 history 單獨跑一次（OFF），
配對 bootstrap 量 Δ(ON−OFF)＝「記憶買到多少解析力」（沿用 hybrid/reranker 的 ablation 風格）。

resolved（follow-up turn）＝ expect_args 全present ∧ expect_not_args 全absent ∧ (expect_tool 命中，若指定)。
  以 **tool args 字串** 為準，不看 result——避免 DEFAULT_UNIVERSE 等彙總結果造成「假解析」。
answer_ok（全 turn）＝ expect_answer 任一子字串出現在 final（沿用 golden 的 any-match）。

用法（rag/）：
  python eval_multiturn.py validate   只驗結構（不跑 LLM）
  python eval_multiturn.py            跑完整評測（~20 個 agent run）
"""
import json
import sys
from pathlib import Path

import agent
from stats import bootstrap_ci, paired_bootstrap_delta

GOLDEN = Path(__file__).resolve().parent / "golden_multiturn.jsonl"
OUT = Path(__file__).resolve().parent / "multiturn_results.json"


def load() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate() -> list[str]:
    convos = load()
    problems = []
    n_turn = n_fu = 0
    for c in convos:
        if not c.get("turns") or c["turns"][0].get("followup"):
            problems.append(f"{c.get('id')}：首輪不應是 follow-up")
        for i, t in enumerate(c["turns"]):
            n_turn += 1
            n_fu += 1 if t.get("followup") else 0
            for key in ("q", "followup", "expect_tool", "expect_args", "expect_answer"):
                if key not in t:
                    problems.append(f"{c['id']} t{i}：缺欄位 {key}")
    print(f"multi-turn golden：{len(convos)} 段對話 / {n_turn} turns / {n_fu} follow-ups")
    if problems:
        print(f"⚠️ {len(problems)} 個結構問題：")
        for p in problems:
            print("  " + p)
    else:
        print("✅ 結構合法（每段首輪非 follow-up、欄位齊全）。")
    return problems


def _args_str(trace) -> str:
    return " ".join(json.dumps(t["args"], ensure_ascii=False) for t in trace)


def _tools(trace) -> list[str]:
    return [t["tool"] for t in trace]


def resolved(turn: dict, trace: list) -> bool:
    a = _args_str(trace)
    if not all(s in a for s in turn.get("expect_args", [])):
        return False
    if any(s in a for s in turn.get("expect_not_args", [])):
        return False
    et = turn.get("expect_tool")
    if et and et not in _tools(trace):
        return False
    return True


def answer_ok(turn: dict, final: str) -> bool:
    exp = turn.get("expect_answer", [])
    return (not exp) or any(s in final for s in exp)


def run_eval():
    convos = load()
    res_on, res_off, ans_flags, rows = [], [], [], []
    for c in convos:
        history = []
        for i, turn in enumerate(c["turns"]):
            final, trace = agent.run(turn["q"], history, verbose=False)
            a_ok = answer_ok(turn, final)
            ans_flags.append(1 if a_ok else 0)
            rec = {"id": c["id"], "phenom": c["phenom"], "turn": i, "q": turn["q"],
                   "followup": bool(turn["followup"]), "tools": _tools(trace),
                   "args": _args_str(trace)[:140], "answer": final[:100], "answer_ok": a_ok}
            if turn["followup"]:
                r_on = resolved(turn, trace)
                off_final, off_trace = agent.run(turn["q"], [], verbose=False)   # OFF：空 history
                r_off = resolved(turn, off_trace)
                res_on.append(1 if r_on else 0)
                res_off.append(1 if r_off else 0)
                rec.update({"resolved_on": r_on, "resolved_off": r_off,
                            "off_tools": _tools(off_trace), "off_args": _args_str(off_trace)[:140]})
            rows.append(rec)
            history += [{"role": "user", "content": turn["q"]},
                        {"role": "assistant", "content": final}]
            history = history[-6:]
    return res_on, res_off, ans_flags, rows


def main():
    res_on, res_off, ans_flags, rows = run_eval()
    n_fu = len(res_on)
    on_acc = sum(res_on) / n_fu if n_fu else 0.0
    off_acc = sum(res_off) / n_fu if n_fu else 0.0
    on_ci, off_ci = bootstrap_ci(res_on), bootstrap_ci(res_off)
    d, dlo, dhi = paired_bootstrap_delta(res_on, res_off)
    ans_acc = sum(ans_flags) / len(ans_flags) if ans_flags else 0.0
    ans_ci = bootstrap_ci(ans_flags)
    n_conv = len({r["id"] for r in rows})

    print("=" * 70)
    print(f"Multi-turn golden：{n_conv} 段對話 / {len(rows)} turns / {n_fu} follow-ups")
    print("=" * 70)
    print("\n[逐 turn 明細]")
    for r in rows:
        tag = "FU" if r["followup"] else "  "
        line = f"  {r['id']} t{r['turn']} {tag} tools={r['tools']} ans_ok={r['answer_ok']}"
        if r["followup"]:
            line += f" | resolved ON={r['resolved_on']} OFF={r['resolved_off']}"
            if not r["resolved_off"]:
                line += f" (off_tools={r['off_tools']})"
        print(line)
        print(f"       q={r['q']}  →  {r['answer']}")

    print("\n[指標]")
    sig = "顯著" if dlo > 0 else "不顯著"
    print(f"  解析準確率(follow-up, n={n_fu})  ON  : {on_acc:.3f}  95%CI [{on_ci[0]:.3f}, {on_ci[1]:.3f}]")
    print(f"  解析準確率(follow-up, n={n_fu})  OFF : {off_acc:.3f}  95%CI [{off_ci[0]:.3f}, {off_ci[1]:.3f}]")
    print(f"  Δ(ON−OFF) 配對 bootstrap           : {d:+.3f}  95%CI [{dlo:+.3f}, {dhi:+.3f}]  ({sig})")
    print(f"  答案準確率(全 {len(ans_flags)} turns)         : {ans_acc:.3f}  95%CI [{ans_ci[0]:.3f}, {ans_ci[1]:.3f}]")

    summary = {
        "n_convo": n_conv, "n_turn": len(rows), "n_followup": n_fu,
        "resolution_on": on_acc, "resolution_on_ci": on_ci,
        "resolution_off": off_acc, "resolution_off_ci": off_ci,
        "resolution_delta": d, "resolution_delta_ci": [dlo, dhi],
        "answer_acc": ans_acc, "answer_acc_ci": ans_ci, "rows": rows,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已寫入 {OUT.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        sys.exit(1 if validate() else 0)
    main()
