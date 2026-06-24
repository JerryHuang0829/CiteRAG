"""W2 retrieval eval：對測試集量 hit@k / recall@k / MRR / nDCG@k + bootstrap 95% CI。

gold chunk 由「答案字串」自動推導（normalize 去逗號/空白後 substring 命中），
不需手工標頁碼、客觀可重現。只評檢索（不呼叫 LLM），跑很快。
用法（rag/）：python eval_retrieval.py
"""
import math
import re

import core
from stats import bootstrap_ci

KS = [3, 5, 10]
MAIN_K = 5

# (問題, [答案字串候選]) — 任一字串(normalize後)出現在某 chunk 即視該 chunk 為 gold
CASES = [
    ("鴻海 2022 全年每股盈餘 EPS 是多少？", ["10.21"]),
    ("鴻海 2022 全年合併營收是多少？", ["6.627", "6,627", "66,272", "6兆"]),
    ("鴻海 2022 第四季 EPS 是多少？", ["2.88"]),
    ("鴻海 2022 第四季毛利率是多少？", ["5.66"]),
    ("鴻海 2022 全年資本支出是多少？", ["979"]),
    ("鴻海法說會的董事長是誰？", ["劉揚偉"]),
    ("鴻海的財務長是誰？", ["黃德才"]),
    ("鴻海帳上現金有多少？", ["1.06兆", "1.06", "1兆0"]),
    ("興櫃股票市場是何時成立的？", ["91/1/2", "91年", "2002"]),
    ("興櫃累計有多少家次登錄？", ["1,326", "1326"]),
    ("興櫃有多少家轉上市櫃？", ["776"]),
    ("興櫃掛牌家數有多少？", ["256"]),
]


def norm(s: str) -> str:
    return re.sub(r"[,\s]", "", s)


def gold_indices(chunks, answers):
    na = [norm(a) for a in answers]
    return {i for i, c in enumerate(chunks) if any(a in norm(c["text"]) for a in na)}


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def metrics_at_k(ret, gold, k):
    topk = ret[:k]
    rels = [1 if i in gold else 0 for i in topk]
    hit = 1 if any(rels) else 0
    recall = sum(rels) / len(gold)
    mrr = 0.0
    for rank, r in enumerate(rels, 1):
        if r:
            mrr = 1.0 / rank
            break
    idcg = dcg([1] * min(len(gold), k))
    ndcg = dcg(rels) / idcg if idcg else 0.0
    return hit, recall, mrr, ndcg


def main():
    index, chunks = core.load_index()
    embedder = core.get_embedder()
    maxk = max(KS)

    valid = []          # (q, gold, ret)
    print("=== 每題（gold 數 / top-{} 命中）===".format(MAIN_K))
    for q, ans in CASES:
        gold = gold_indices(chunks, ans)
        if not gold:
            print(f"  [no gold] {q}  ← 答案字串未命中任何 chunk，檢查")
            continue
        qv = core.embed_query(embedder, q)
        _, idxs = index.search(qv, maxk)
        ret = [int(i) for i in idxs[0] if i >= 0]
        hit5, rec5, _, _ = metrics_at_k(ret, gold, MAIN_K)
        mark = "OK " if hit5 else "MISS"
        print(f"  [{mark}] gold={len(gold)} {q}")
        valid.append((q, gold, ret))

    print(f"\n=== 聚合（n={len(valid)} 題有效）===")
    print(f"{'k':>3} {'hit@k':>7} {'recall@k':>9} {'MRR':>6} {'nDCG@k':>7}")
    hit_at_main = []
    for k in KS:
        hs, rs, ms, ns = [], [], [], []
        for _, gold, ret in valid:
            h, r, mr, nd = metrics_at_k(ret, gold, k)
            hs.append(h); rs.append(r); ms.append(mr); ns.append(nd)
        if k == MAIN_K:
            hit_at_main = hs
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        print(f"{k:>3} {avg(hs):>7.3f} {avg(rs):>9.3f} {avg(ms):>6.3f} {avg(ns):>7.3f}")

    lo, hi = bootstrap_ci(hit_at_main)
    degen = " (0/1 退化→rule-of-three 下界)" if min(hit_at_main) == max(hit_at_main) else ""
    print(f"\nhit@{MAIN_K} = {sum(hit_at_main)/len(hit_at_main):.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]{degen}  (n={len(hit_at_main)})")
    print("→ 這是 baseline（bge-small-zh, 無 rerank）。下一步：reranker / chunk size ablation 對比。")


if __name__ == "__main__":
    main()
