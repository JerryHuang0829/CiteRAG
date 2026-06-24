"""① hybrid 檢索 ablation：dense-only vs hybrid(dense + BM25，RRF 融合)。

量兩個層級（皆 paired bootstrap Δ + 95% CI）：
  (1) 檢索層（無 rerank，top-5）—— hybrid 影響的是「哪些 chunk 被撈上來」，這層才看得到效果。
  (2) 端到端（候選 top-N → reranker → top-5）—— 強 reranker 在小語料會「補回」dense 的漏召，可能洗掉差異。

誠實結論（本 78-chunk 語料）：hybrid 在檢索層提升 recall/hit；但 top-20 候選＝26% 語料 + 強
reranker → 端到端可能打平。hybrid 的端到端 ROI 隨語料變大而上升（候選佔比變小時，gold 可能根本
不在 dense top-N，reranker 救不回不在池子裡的東西）。
用法（rag/）：python eval_hybrid.py
"""
import core
from eval_retrieval import CASES, gold_indices, metrics_at_k
from stats import paired_bootstrap_delta

N = core.RERANK_CANDIDATES


def eval_arm(valid, index, emb, chunks, hybrid, reranker=None):
    per = {"hit@3": [], "recall@5": [], "mrr": [], "ndcg@5": []}
    for q, gold in valid:
        if reranker is None:                       # 檢索層：直接取 top-5
            order = core.candidates(index, emb, chunks, q, 5, hybrid)
        else:                                      # 端到端：候選 top-N → rerank
            cand = core.candidates(index, emb, chunks, q, N, hybrid)
            order = [cand[i] for i in core.rerank_order(
                reranker, q, [chunks[i]["text"] for i in cand])]
        h3, _, _, _ = metrics_at_k(order, gold, 3)
        _, r5, mrr, nd5 = metrics_at_k(order, gold, 5)
        per["hit@3"].append(h3); per["recall@5"].append(r5)
        per["mrr"].append(mrr); per["ndcg@5"].append(nd5)
    return per


def report(title, dense, hybrid):
    avg = lambda xs: sum(xs) / len(xs)
    print(f"\n── {title} ──")
    for name, per in [("dense", dense), ("hybrid", hybrid)]:
        print(f"  [{name:>7}] hit@3 {avg(per['hit@3']):.3f} | recall@5 {avg(per['recall@5']):.3f} | "
              f"MRR {avg(per['mrr']):.3f} | nDCG@5 {avg(per['ndcg@5']):.3f}")
    for metric in ["recall@5", "hit@3", "mrr", "ndcg@5"]:
        pt, lo, hi = paired_bootstrap_delta(hybrid[metric], dense[metric])
        sig = "顯著" if (lo > 0 or hi < 0) else "不顯著(CI 跨 0)"
        print(f"    Δ {metric:>8} = {pt:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]  {sig}")


def main():
    index, chunks = core.load_index()
    emb = core.get_embedder()
    reranker = core._live_reranker()
    valid = [(q, g) for q, g in ((q, gold_indices(chunks, a)) for q, a in CASES) if g]
    print(f"有效題數 n={len(valid)}；語料 {len(chunks)} chunks；候選 N={N}（{N/len(chunks)*100:.0f}% 語料）")

    report("(1) 檢索層（無 rerank，top-5）— hybrid 的效果在這裡",
           eval_arm(valid, index, emb, chunks, False),
           eval_arm(valid, index, emb, chunks, True))
    report("(2) 端到端（候選→rerank→top-5）— 強 reranker 在小語料會洗掉差異",
           eval_arm(valid, index, emb, chunks, False, reranker),
           eval_arm(valid, index, emb, chunks, True, reranker))
    print("\n註：hybrid 在檢索層提升 recall/hit（BM25 補精確詞/數字 silent fail）；端到端因 top-N=26% 語料 + "
          "強 reranker 而可能打平。hybrid 的端到端 ROI 隨語料規模上升——大語料時 gold 可能不在 dense top-N，"
          "reranker 救不回不在候選池的 chunk。此為小語料的誠實結論。")


if __name__ == "__main__":
    main()
