"""② RAG Triad 生成端評測（golden set）：手刻 RAGAS-style 指標 + 本地 judge。

對 golden set 跑完整 RAG（retrieve→rerank→generate→引用護欄），量：
  answer correctness（程式硬驗）：answerable→答案含 gold；refuse→是否正確拒答
  context recall   （程式硬驗）：gold 是否出現在「檢索到的 context」（檢索層有沒有撈到答案）
  context precision（程式硬驗）：檢索 chunk 中「含 gold」的比例（檢索精準度）
  faithfulness     （本地 judge）：答案事實陳述是否都被 context 支撐（RAG Triad 核心，擋幻覺）
  answer relevancy （本地 judge）：答案是否切題

程式硬驗指標客觀可重現、附 bootstrap CI；faithfulness/relevancy 用本地 qwen3:4b 當 judge＝noisy
proxy（position/verbosity/self-enhancement bias），數字為相對趨勢非絕對真值，需人工抽查或更強 judge 校準。
用法（rag/）：python eval_rag_triad.py [N]   （N=只跑前 N 題，省時/CI 用；省略＝全部）
"""
import json
import sys

import core
from eval_retrieval import norm
from golden import load_golden
from stats import bootstrap_ci

REFUSE = ["查無", "查不到", "無法", "沒有", "未提供", "未提及", "未明確", "未揭露",
          "找不到", "無資料", "無相關", "不到", "無此", "未說明"]


def judge_faithful(context: str, answer: str):
    prompt = ("你是嚴格評審。判斷「答案」中的事實陳述是否『都』能被「參考資料」支持（不可加料、不可超出資料）。\n"
              "只輸出 JSON：{\"faithful\": 1 或 0}\n\n"
              f"參考資料：\n{context}\n\n答案：\n{answer}")
    try:
        return 1 if json.loads(core.generate(prompt, as_json=True)).get("faithful") in (1, "1", True) else 0
    except Exception:
        return None


def judge_relevant(question: str, answer: str):
    prompt = ("你是評審。判斷「答案」是否切題回答了「問題」（即使因資訊不足而拒答，只要針對該問題也算切題）。\n"
              "只輸出 JSON：{\"relevant\": 1 或 0}\n\n"
              f"問題：{question}\n\n答案：{answer}")
    try:
        return 1 if json.loads(core.generate(prompt, as_json=True)).get("relevant") in (1, "1", True) else 0
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return "Y" if x else "N"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    items = load_golden()[:limit]

    correct, c_recall, c_prec, faith, relev = [], [], [], [], []
    by_cat = {}
    print(f"{'id':>4} {'cat':>10} {'corr':>4} {'cRec':>4} {'cPrec':>5} {'fth':>3} {'rel':>3}  question")
    for it in items:
        hits = core.retrieve(it["question"])
        ctx = "\n\n".join(f"(p.{h['page']}) {h['text']}" for h in hits)
        ans, _ = core.verify_citations(core.generate(core.build_prompt(it["question"], hits)),
                                       {h["page"] for h in hits})

        if it["answerable"]:
            gold = it["gold"]
            corr = 1 if any(norm(g) in norm(ans) for g in gold) else 0
            hit_has_gold = [any(norm(g) in norm(h["text"]) for g in gold) for h in hits]
            crec = 1 if any(hit_has_gold) else 0
            cprec = (sum(hit_has_gold) / len(hits)) if hits else 0.0
            c_recall.append(crec); c_prec.append(cprec)
        else:
            corr = 1 if any(w in ans for w in REFUSE) else 0   # refuse 題正解＝拒答
            crec = cprec = None

        f = judge_faithful(ctx, ans)
        rel = judge_relevant(it["question"], ans)
        correct.append(corr)
        if f is not None:
            faith.append(f)
        if rel is not None:
            relev.append(rel)
        by_cat.setdefault(it["category"], []).append(corr)
        print(f"{it['id']:>4} {it['category']:>10} {('OK' if corr else 'x'):>4} "
              f"{_fmt(crec):>4} {_fmt(cprec):>5} {_fmt(f):>3} {_fmt(rel):>3}  {it['question'][:24]}")

    def rate(xs):
        return sum(xs) / len(xs) if xs else 0.0

    clo, chi = bootstrap_ci(correct)
    print("\n========== 聚合（RAG Triad）==========")
    print(f"answer correctness    : {rate(correct):.3f}  95% CI [{clo:.3f}, {chi:.3f}]  (n={len(correct)})")
    print(f"context recall        : {rate(c_recall):.3f}  (answerable n={len(c_recall)}；gold 有無被檢索到)")
    print(f"context precision     : {rate(c_prec):.3f}  (檢索 chunk 含 gold 的比例)")
    print(f"faithfulness (4B judge): {rate(faith):.3f}  (n={len(faith)})  ← noisy proxy")
    print(f"answer relevancy(judge): {rate(relev):.3f}  (n={len(relev)})  ← noisy proxy")
    print("\n— answer correctness 分類 —")
    for cat, xs in sorted(by_cat.items()):
        print(f"  {cat:>10}: {rate(xs):.3f}  (n={len(xs)})")
    print("\n註：context recall/precision、answer correctness＝程式硬驗（客觀可重現）；faithfulness/relevancy "
          "用本地 qwen3:4b 當 judge＝noisy proxy（有 bias），為相對趨勢非絕對真值，需人工抽查校準。"
          "refuse 題 correctness＝是否正確拒答。")


if __name__ == "__main__":
    main()
