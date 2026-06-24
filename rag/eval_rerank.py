"""W2 reranker ablation（嚴格版）：無 rerank vs bge-reranker-base vs jina-reranker-v2-multilingual。

流程：每題先取 FAISS top-N 候選，各 reranker 重排後取 top-k，量 hit@3/recall@5/MRR/nDCG@5。
嚴格之處：對「ordering 品質」指標(MRR, nDCG@5)做 **paired bootstrap**（同題重抽），
回報 Δ(reranker − none) 的 95% CI 與是否顯著——避免只看點估計。
用法（rag/）：python eval_rerank.py
"""
import core
from eval_retrieval import CASES, gold_indices, metrics_at_k
from stats import paired_bootstrap_delta

N = core.RERANK_CANDIDATES

RERANKERS = [
    ("none", None),
    ("bge-base", "BAAI/bge-reranker-base"),
    ("jina-v2-ml", "jinaai/jina-reranker-v2-base-multilingual"),
]


def main():
    index, chunks = core.load_index()
    embedder = core.get_embedder()

    valid = []   # (q, gold, faiss_cand_idxs)
    for q, ans in CASES:
        gold = gold_indices(chunks, ans)
        if not gold:
            continue
        valid.append((q, gold, core.faiss_topn(index, embedder, q, N)))
    print(f"有效題數 n={len(valid)}（每題 FAISS top-{N} 候選重排）\n")

    results = {}
    for name, model in RERANKERS:
        reranker = core.get_reranker(model) if model else None
        per = {"hit@3": [], "recall@5": [], "mrr": [], "ndcg@5": []}
        for q, gold, cand in valid:
            if reranker is None:
                order = cand
            else:
                order = [cand[i] for i in core.rerank_order(
                    reranker, q, [chunks[i]["text"] for i in cand])]
            h3, _, _, _ = metrics_at_k(order, gold, 3)
            _, r5, mrr, nd5 = metrics_at_k(order, gold, 5)
            per["hit@3"].append(h3)
            per["recall@5"].append(r5)
            per["mrr"].append(mrr)
            per["ndcg@5"].append(nd5)
        results[name] = per

        def avg(xs):
            return sum(xs) / len(xs)
        print(f"[{name:>10}] hit@3 {avg(per['hit@3']):.3f} | "
              f"recall@5 {avg(per['recall@5']):.3f} | "
              f"MRR {avg(per['mrr']):.3f} | nDCG@5 {avg(per['ndcg@5']):.3f}")

    base = results["none"]
    print("\n=== paired bootstrap：Δ(reranker − none)，95% CI ===")
    for name, _ in RERANKERS:
        if name == "none":
            continue
        for metric in ["mrr", "ndcg@5"]:
            pt, lo, hi = paired_bootstrap_delta(results[name][metric], base[metric])
            sig = "顯著" if (lo > 0 or hi < 0) else "不顯著(CI 跨 0)"
            print(f"  {name:>10} {metric:>7}: Δ={pt:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]  {sig}")
    print("\n註：n 小時 CI 會寬；不顯著代表『此測試集上看不出差異』，"
          "是誠實結論而非失敗。擴題後再下定論。")


if __name__ == "__main__":
    main()
