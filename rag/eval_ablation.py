"""檢索元件 ablation：在真實語料量 rerank / hybrid 的貢獻（recall + MRR@k）。

recall 在大語料常都很高（gold 進得了 top-k），差別看 **MRR@k（gold 排多前面）**——rerank 應把 gold 推到更前。
證明每個元件值不值得（研究嚴謹，非「跟 tutorial 加了就加」）。用法（rag/）：python eval_ablation.py
"""
import core
from eval_gate import _rate
from eval_retrieval import norm
from golden import load_golden

CONFIGS = [
    ("dense only", False, False),
    ("+ hybrid(BM25)", True, False),
    ("+ hybrid + rerank", True, True),
]


def _metrics():
    items = [it for it in load_golden() if it["answerable"]]
    recalls, mrrs = [], []
    for it in items:
        hits = core.retrieve(it["question"])
        rel = [1 if any(norm(g) in norm(h["text"]) for g in it["gold"]) else 0 for h in hits]
        recalls.append(1 if any(rel) else 0)
        mrr = 0.0
        for i, r in enumerate(rel):
            if r:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
    return _rate(recalls), _rate(mrrs), len(items)


def main():
    print(f"{'config':20} {'recall':>7} {'MRR@5':>7}")
    for name, hy, rr in CONFIGS:
        core.USE_HYBRID, core.USE_RERANK = hy, rr
        rec, mrr, n = _metrics()
        print(f"{name:20} {rec:>7.3f} {mrr:>7.3f}  (n={n})")
    print("\n註：真實語料 9,579 chunks / golden answerable。recall 高＝gold 都進 top-5；MRR 差異＝排序品質（rerank 貢獻）。")


if __name__ == "__main__":
    main()
