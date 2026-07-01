"""凍結的品質 SLO（eval-as-CI-gate 門檻）。改門檻＝改此檔，git 留痕。

門檻設在「目前實測值 − 緩衝」：高到能抓真實退步、低到不被 judge/小樣本雜訊誤殺。
golden answerable n 小（~25）故門檻寬、CI 信賴區間寬；擴題後可上調。
retrieval 層不需 LLM（只需 index+embedder+reranker）；generation 層需雲端或本機 Ollama。
"""

# 檢索層（走 core.retrieve 真實路徑；不呼叫 LLM）
RETRIEVAL_SLO = {
    "context_recall": 0.90,      # answerable 題：至少一個檢索 chunk 含 gold
    "context_precision": 0.30,   # 檢索 chunk 含 gold 的平均比例（floor）
}

# 生成層（需 LLM；PR / nightly 才跑以省免費額度）
GENERATION_SLO = {
    "answer_correctness": 0.80,  # 答案含 gold 字串（程式硬驗，客觀）
    "faithfulness": 0.55,        # 本地/便宜 judge＝noisy proxy，門檻設保守
}
