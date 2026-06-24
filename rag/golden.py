"""② golden set 載入 + 驗證（所有 eval 的單一事實來源）。

每筆：{id, question, answerable, gold[], category}。
- answerable 題：gold 為可接受答案子字串（任一命中即算對）。驗證要求『至少一個 gold 出現在語料』
  以確保事實可檢索；像「2002」這種正確但語料未literal 出現的換算答案不要求在語料（只要同題另有
  一個 grounded 的 gold 即可）。
- refuse 題：answerable=false、gold 為空，正解＝拒答（不可編造）。
用法（rag/）：python golden.py   → 驗證 golden set 是否 grounded（CI 可當 gate）。
"""
import json
import sys
from pathlib import Path

import core
from eval_retrieval import norm

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate() -> list[tuple]:
    _, chunks = core.load_index()
    corpus = norm(" ".join(c["text"] for c in chunks))
    items = load_golden()
    problems = []
    cats: dict[str, int] = {}
    n_ans = n_ref = 0
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
        if it["answerable"]:
            n_ans += 1
            if not any(norm(g) in corpus for g in it["gold"]):
                problems.append((it["id"], "無任何 gold 出現在語料（事實不可檢索/gold 錯）", it["gold"]))
        else:
            n_ref += 1
            if it["gold"]:
                problems.append((it["id"], "refuse 題不應有 gold", it["gold"]))

    print(f"golden set: {len(items)} 題（answerable {n_ans} / refuse {n_ref}）；分類 {cats}")
    if problems:
        print(f"\n⚠️ {len(problems)} 個問題：")
        for pid, msg, g in problems:
            print(f"  [{pid}] {msg}: {g}")
    else:
        print("✅ 全部 grounded：每個 answerable 題至少一個 gold 在語料、refuse 題無 gold。")
    return problems


if __name__ == "__main__":
    sys.exit(1 if validate() else 0)
