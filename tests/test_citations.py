"""引用護欄回歸測試（core.verify_citations / pages_in）。擋 citation-shaped 幻覺的核心邏輯。"""
import core


def test_strip_out_of_range_page():
    assert core.verify_citations("EPS 10.21 (p.3)。毛利率 5.66% (p.7)", {3}) == \
        ("EPS 10.21 (p.3)。毛利率 5.66%", [7])


def test_strip_when_no_retrieved_source():
    # lookup_metric-only 答案捏造頁碼 → 全剝
    assert core.verify_citations("純指標答案 (p.3)", set()) == ("純指標答案", [3])


def test_keep_clean_answer_unchanged():
    assert core.verify_citations("沒有引用的答案", {3}) == ("沒有引用的答案", [])


def test_multipage_partial_keeps_valid():
    assert core.verify_citations("A (p.3,4) B", {3}) == ("A (p.3) B", [4])


def test_multipage_strip_both_no_orphan_comma():
    cleaned, stripped = core.verify_citations("A (p.99,100) B", {3})
    assert cleaned == "A B"
    assert sorted(stripped) == [99, 100]


def test_multipage_keep_both_when_all_valid():
    assert core.verify_citations("A (p.3,4) B", {3, 4}) == ("A (p.3,4) B", [])


def test_pages_in_extracts_all_forms():
    assert core.pages_in("(p.3,4) 另見 p.5 與（p.12）") == {3, 4, 5, 12}


# ---- verify_numbers：數值溯源護欄（擋『沒查就編數字』，#6 多輪幻覺）----
def test_numbers_all_grounded_no_flag():
    g = "台積電(2330) 2023 全年營收 2.162 兆元"
    assert core.verify_numbers("台積電 2023 年營收為 2.162 兆元", g) == []


def test_numbers_fabricated_flagged():
    # mt6：問鴻海股價卻沒查、編出 112.50/+23.7（grounded 只有台積電 2390）
    g = "台積電(2330) 2026-06-25 收盤 2390.0 元；近一年 +143.4%"
    bad = core.verify_numbers("鴻海現股價為 112.50 元，近一年漲幅為 +23.7%", g)
    assert "112.50" in bad and "23.7" in bad


def test_numbers_rounding_tolerated():
    # 答案四捨五入 2.16 vs 工具 2.162 → 不誤判
    assert core.verify_numbers("營收約 2.16 兆元", "全年營收 2.162 兆元") == []


def test_numbers_year_excluded():
    # 年份不視為需溯源的值（即使 grounded 沒有也不報）
    assert core.verify_numbers("2024 年的數據", "全年營收 100 億元") == []


def test_numbers_thousand_separator_and_percent():
    g = "全年營收 2225 億元；毛利率 47.84%"
    assert core.verify_numbers("營收 2,225 億元、毛利率 47.84%", g) == []


def test_numbers_partial_mix():
    g = "EPS 32.34 元"
    bad = core.verify_numbers("EPS 32.34 元，本益比約 18.5 倍", g)
    assert bad == ["18.5"]   # 32.34 grounded、18.5 未溯源
