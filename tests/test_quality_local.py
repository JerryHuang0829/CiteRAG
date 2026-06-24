"""品質 gate（需 index / reranker；標 @local，雲端 CI 跳過，本地跑：pytest -m local）。

faithfulness 等需 Ollama 的門檻不放這裡（CPU 數十分鐘），由 eval_rag_triad.py 本地產出後人工把關。
這裡只放「不需 Ollama、但需 index+embedder」的檢索層 gate。
"""
import pytest


@pytest.mark.local
def test_golden_set_grounded():
    # 每個 answerable 題至少一個 gold 在語料、refuse 題無 gold
    from golden import validate
    assert validate() == []


@pytest.mark.local
def test_context_recall_baseline():
    # 檢索層 context recall：gold 是否被撈進 hits（不經 Ollama）。低於 baseline 即 fail。
    import core
    from golden import load_golden
    from eval_retrieval import norm

    ok = []
    for it in load_golden():
        if not it["answerable"]:
            continue
        hits = core.retrieve(it["question"])
        ok.append(1 if any(any(norm(g) in norm(h["text"]) for g in it["gold"]) for h in hits) else 0)
    recall = sum(ok) / len(ok)
    assert recall >= 0.70, f"context recall {recall:.3f} < 0.70 baseline（檢索退步）"
