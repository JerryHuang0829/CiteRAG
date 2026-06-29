"""findata 結構化查詢的離線單元測試（agent 三個數值工具的真值來源）。

monkeypatch 掉所有網路（_fetch/_resolve_year/_load_all_codes），純測運算與分派邏輯；
屬雲端確定性測試（不需 FinMind/Ollama）。
"""
import findata as f


def _rows(eps=(8.0, 8.0, 8.0, 8.34), rev_q=5e11, gp_q=2.5e11, net_q=2e11):
    rows = [{"type": "EPS", "value": v} for v in eps]
    rows += [{"type": "Revenue", "value": rev_q} for _ in range(4)]
    rows += [{"type": "GrossProfit", "value": gp_q} for _ in range(4)]
    rows += [{"type": "IncomeAfterTaxes", "value": net_q} for _ in range(4)]
    return rows


def test_metric_value_eps_revenue_grossmargin():
    rows = _rows()
    assert round(f._metric_value(rows, "EPS"), 2) == 32.34
    assert f._metric_value(rows, "營收") == 2e12                 # Revenue 5e11×4
    assert round(f._metric_value(rows, "毛利率"), 1) == 50.0     # (2.5e11×4)/(5e11×4)×100


def test_grossmargin_vs_grossprofit_dispatch_order():
    # 「毛利率」須在「毛利」子串之前判定（順序敏感的分派）
    rows = _rows()
    assert f._metric_value(rows, "毛利率") == 50.0
    assert f._metric_value(rows, "毛利") == 1e12                 # GrossProfit sum，非百分比


def test_grossmargin_zero_revenue_returns_none():
    assert f._metric_value([{"type": "GrossProfit", "value": 100}], "毛利率") is None


def test_money_boundaries():
    assert f._money(2.162e12) == "2.162 兆元"
    assert f._money(2.225e11) == "2225 億元"
    assert f._money(5000) == "5,000 元"


def test_fmt_metric():
    assert f._fmt_metric("EPS", 32.34) == "32.34 元"
    assert f._fmt_metric("毛利率", 47.84) == "47.84%"
    assert "兆" in f._fmt_metric("營收", 2.162e12)


def test_sum_tolerates_missing_and_none_value():
    rows = [{"type": "EPS", "value": 1.0}, {"type": "EPS"}, {"type": "EPS", "value": None}]
    assert f._sum(rows, "EPS") == 1.0


def test_resolve_code_exact_and_substring():
    assert f._resolve_code("台積電") == "2330"
    assert f._resolve_code("2330") == "2330"
    assert f._resolve_code("台積電的股價") == "2330"            # 子串命中


def test_resolve_code_longest_match_not_prefix(monkeypatch):
    monkeypatch.setattr(f, "_load_all_codes", lambda: {"台塑化": "6505", "南亞科": "2408"})
    # 台塑化 不在內建；子串時取最長匹配（台塑化）而非內建短名台塑(1301)
    assert f._resolve_code("台塑化的EPS") == "6505"


def test_lookup_partial_year_labeled_not_full_year(monkeypatch):
    rows = [{"type": "EPS", "value": 5.0}, {"type": "EPS", "value": 5.0}]   # 僅 2 季
    monkeypatch.setattr(f, "_resolve_code", lambda c: "2330")
    monkeypatch.setattr(f, "_resolve_year", lambda c, y: "2026")
    monkeypatch.setattr(f, "_fetch", lambda code, year: rows)
    out = f.lookup("台積電", "EPS", 2026)
    assert "前2季累計" in out and "全年" not in out


def test_lookup_full_year_labeled_full(monkeypatch):
    monkeypatch.setattr(f, "_resolve_code", lambda c: "2330")
    monkeypatch.setattr(f, "_resolve_year", lambda c, y: "2023")
    monkeypatch.setattr(f, "_fetch", lambda code, year: _rows())            # 四季完整
    assert "全年EPS 32.34 元" in f.lookup("台積電", "EPS", 2023)


def test_compare_threshold_no_match_shows_candidates(monkeypatch):
    monkeypatch.setattr(f, "_resolve_code", lambda c: {"台積電": "2330", "聯電": "2303"}.get(c, "0000"))
    monkeypatch.setattr(f, "_resolve_year", lambda c, y: "2023")
    monkeypatch.setattr(f, "_fetch", lambda code, year: _rows())            # EPS 32.34
    out = f.compare("EPS", "台積電,聯電", 2023, ">100")                      # 無人符合
    assert "沒有公司符合" in out and "候選" in out and "32.34" in out        # 不再是空字串範圍
