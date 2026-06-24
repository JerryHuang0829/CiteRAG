"""統計原語回歸測試（純 stdlib，秒級，雲端 CI 可跑）。"""
import stats


def test_rule_of_three():
    assert abs(stats.rule_of_three(8) - 0.625) < 1e-9
    assert abs(stats.rule_of_three(10) - 0.70) < 1e-9
    assert stats.rule_of_three(0) == 0.0


def test_percentile_matches_linear_interpolation():
    # 與 numpy.percentile 預設 method="linear" 一致的已知值
    s = [1, 2, 3, 4]
    assert abs(stats._percentile(s, 2.5) - 1.075) < 1e-9
    assert abs(stats._percentile(s, 50) - 2.5) < 1e-9
    assert abs(stats._percentile(s, 97.5) - 3.925) < 1e-9


def test_bootstrap_degenerate_all_one():
    # 全 1 → percentile bootstrap 必塌成單點 → 改報 rule-of-three（M1）
    assert stats.bootstrap_ci([1] * 12) == (0.75, 1.0)


def test_bootstrap_degenerate_all_zero():
    assert stats.bootstrap_ci([0] * 12) == (0.0, 0.25)


def test_bootstrap_empty():
    assert stats.bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_reproducible():
    xs = [1, 0, 1, 1, 0, 1, 1, 0]
    assert stats.bootstrap_ci(xs) == stats.bootstrap_ci(xs)   # 固定 seed → 可重現


def test_paired_bootstrap_empty_guard():
    assert stats.paired_bootstrap_delta([], []) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_point():
    pt, lo, hi = stats.paired_bootstrap_delta([1, 1, 1, 0, 1], [0, 0, 1, 0, 0])
    assert abs(pt - 0.6) < 1e-9
    assert lo <= pt <= hi
