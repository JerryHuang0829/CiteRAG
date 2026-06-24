"""BM25 / RRF / tokenizer 回歸測試（hybrid 檢索的純函式部分，無需 index/Ollama）。"""
import core


def test_tokenizer_keeps_numbers_and_terms():
    t = core.bm25_tokens("鴻海 2022 EPS 10.21 元")
    assert "10.21" in t          # 數字(含小數)整段保留 → 精確比對
    assert "2022" in t
    assert "eps" in t            # 英文小寫化
    assert "鴻海" in t           # 中文 bigram
    assert "元" in t             # 中文單字也保留


def test_bm25_ranks_exact_term_doc_first():
    # 純向量易漏的 silent-fail 情境：精確股號只有 BM25 抓得準
    docs = [core.bm25_tokens(x) for x in
            ["鴻海全年每股盈餘表現", "台積電營收成長", "鴻海 2317 EPS 10.21"]]
    scores = core._BM25(docs).scores(core.bm25_tokens("2317"))
    assert scores.index(max(scores)) == 2


def test_rrf_favors_cross_list_item():
    # idx 1 在兩路排名都靠前 → 融合後最高
    fused = core._rrf([[5, 1, 2], [1, 9, 5]])
    assert max(fused, key=fused.get) == 1
