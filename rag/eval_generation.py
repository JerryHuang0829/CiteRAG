"""W2 generation eval：跑完整 RAG（retrieve+rerank+generate），量答案層三件事：

1. answer correctness（程式硬驗）：答案是否含正解字串。
2. citation 正確率（程式硬驗）：答案標的頁碼是否落在「含正解的 chunk」的頁碼集合。
3. faithfulness（LLM-as-judge，本地 4B）：答案是否被檢索內容支持（有 judge 偏差，已標註）。

前 2 項客觀可重現；第 3 項是 noisy proxy，附 CI 並提醒需人工校準。
用法（rag/）：python eval_generation.py   （會跑 ~12 題 × 生成+評審，CPU 慢，數分鐘）
"""
import json
import re

import core
from eval_retrieval import CASES, gold_indices, norm
from stats import bootstrap_ci

CITE_RE = re.compile(r"p\.?\s*(\d+)", re.IGNORECASE)


def cited_pages(answer: str):
    return sorted({int(m) for m in CITE_RE.findall(answer)})


def judge_faithful(context: str, answer: str):
    prompt = (
        "你是嚴格的評審。判斷「答案」中的事實陳述是否都能被「參考資料」支持（不可加料、不可超出資料）。\n"
        "只輸出 JSON：{\"faithful\": 1 或 0, \"unsupported\": \"未被支持處，沒有則空字串\"}\n\n"
        f"參考資料：\n{context}\n\n答案：\n{answer}"
    )
    try:
        obj = json.loads(core.generate(prompt, as_json=True))
        return 1 if obj.get("faithful") in (1, "1", True) else 0
    except Exception:
        return None   # judge 沒吐出合法 JSON


def main():
    _, chunks = core.load_index()
    ac, cc, faith = [], [], []
    cite_stat = {"correct": 0, "wrong": 0, "uncited": 0}

    print(f"{'ans':>3} {'cite':>7} {'faith':>5}  question")
    for q, gold_subs in CASES:
        gold_idx = gold_indices(chunks, gold_subs)
        gold_pages = {chunks[i]["page"] for i in gold_idx}

        hits = core.retrieve(q)                       # live 設定（含 rerank）
        ctx = "\n\n".join(f"(p.{h['page']}) {h['text']}" for h in hits)
        ans = core.generate(core.build_prompt(q, hits))

        a_ok = 1 if any(norm(s) in norm(ans) for s in gold_subs) else 0
        cp = cited_pages(ans)
        if not cp:
            cstat = "uncited"
        elif any(p in gold_pages for p in cp):
            cstat = "correct"
        else:
            cstat = "wrong"
        c_ok = 1 if cstat == "correct" else 0
        f = judge_faithful(ctx, ans)

        ac.append(a_ok)
        cc.append(c_ok)
        cite_stat[cstat] += 1
        if f is not None:
            faith.append(f)
        print(f"{('OK' if a_ok else 'x'):>3} {cstat:>7} "
              f"{('-' if f is None else ('Y' if f else 'N')):>5}  {q[:26]}")

    def rate(xs):
        return sum(xs) / len(xs) if xs else 0.0

    alo, ahi = bootstrap_ci(ac)
    clo, chi = bootstrap_ci(cc)
    print("\n========== 聚合 ==========")
    print(f"answer correctness : {rate(ac):.3f}  95% CI [{alo:.3f}, {ahi:.3f}]  (n={len(ac)})")
    print(f"citation 正確率     : {rate(cc):.3f}  95% CI [{clo:.3f}, {chi:.3f}]  "
          f"(correct={cite_stat['correct']} / wrong={cite_stat['wrong']} / uncited={cite_stat['uncited']})")
    if faith:
        flo, fhi = bootstrap_ci(faith)
        print(f"faithfulness(4B judge): {rate(faith):.3f}  95% CI [{flo:.3f}, {fhi:.3f}]  (n={len(faith)})")
    print("\n註：answer/citation 為程式硬驗(客觀)；faithfulness 用本地 4B 當 judge＝noisy proxy，"
          "一致性低、需人工抽查校準或改用更強 judge，數字僅供參考。")


if __name__ == "__main__":
    main()
