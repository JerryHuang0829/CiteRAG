"""W0-04（可選、較重）embedding 繁中檢索實測：bge-small-zh vs bge-m3，
並用 OpenCC 繁→簡量「腳本錯配」（公開 benchmark 只有簡體，繁中是最大盲區）。

前置：pip install -r requirements_w0.txt
資料：放 w0_corpus.jsonl（{"id","text"}）與 w0_queries.jsonl（{"query","relevant":[id,...]}）；
      不存在則用內建 toy 繁中資料跑通流程（數字無意義，僅驗 pipeline）。
輸出：recall@k 表 + w0_results_embed.json
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

K = 5
MODELS = {"bge-small-zh": "BAAI/bge-small-zh-v1.5", "bge-m3": "BAAI/bge-m3"}

TOY_CORPUS = [
    {"id": "c1", "text": "泵浦更換濾芯步驟：關閉進水閥、洩壓、拆濾殼、換新濾芯、回裝、開閥測漏。"},
    {"id": "c2", "text": "資產負債表反映企業在特定時點的資產、負債與股東權益。"},
    {"id": "c3", "text": "除息是指股票發放現金股利後，股價向下調整的過程。"},
    {"id": "c4", "text": "壓力錶校正應每年一次，並記錄誤差於保養紀錄表。"},
]
TOY_QUERIES = [
    {"query": "濾芯怎麼換", "relevant": ["c1"]},
    {"query": "什麼是除息", "relevant": ["c3"]},
    {"query": "壓力錶多久校正一次", "relevant": ["c4"]},
]


def load(path, fallback):
    if os.path.exists(path):
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print(f"  (找不到 {path}，用內建 toy 資料；數字僅供 pipeline 驗證)")
    return fallback


def recall_at_k(model, corpus, queries, k, conv=None):
    ids = [c["id"] for c in corpus]
    emb = model.encode([c["text"] for c in corpus], normalize_embeddings=True)
    hit = 0
    for q in queries:
        qt = conv.convert(q["query"]) if conv else q["query"]
        qe = model.encode([qt], normalize_embeddings=True)[0]
        topk = [ids[i] for i in np.argsort(-(emb @ qe))[:k]]
        if any(r in topk for r in q["relevant"]):
            hit += 1
    return hit / len(queries)


def main():
    corpus = load("w0_corpus.jsonl", TOY_CORPUS)
    queries = load("w0_queries.jsonl", TOY_QUERIES)
    conv = None
    try:
        from opencc import OpenCC
        conv = OpenCC("t2s")   # 繁→簡
    except Exception:
        print("  (未裝 opencc，跳過繁→簡 A/B)")

    print(f"\n{'model':>14} {'recall@'+str(K)+'(繁中)':>16} {'recall@'+str(K)+'(轉簡)':>16}")
    out = {}
    for name, path in MODELS.items():
        m = SentenceTransformer(path)
        r_orig = recall_at_k(m, corpus, queries, K)
        r_s = recall_at_k(m, corpus, queries, K, conv) if conv else None
        out[name] = {"recall_orig": r_orig, "recall_simplified": r_s}
        s = f"{r_s:.3f}" if r_s is not None else "n/a"
        print(f"{name:>14} {r_orig:>16.3f} {s:>16}")
    with open("w0_results_embed.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n→ 『轉簡 recall 明顯較高』= 腳本錯配嚴重，離線臂改用 bge-m3。")


if __name__ == "__main__":
    main()
