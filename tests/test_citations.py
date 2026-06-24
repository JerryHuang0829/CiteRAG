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
