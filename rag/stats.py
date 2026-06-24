"""eval 共用統計原語：bootstrap CI / paired bootstrap delta / rule-of-three。

集中一處避免各 eval 各寫一份漂移（DRY）。純 stdlib（無 numpy），故 w0/ 的零依賴
benchmark 也能共用。CI 一律用線性插值 percentile（等同 numpy.percentile 預設），
比 int() 截斷取單一 order statistic 更準、上下尾對稱。bootstrap 帶固定 seed → 可重現。
"""
import random


def rule_of_three(n: int) -> float:
    # 0 失敗（或 0 成功）時，比率的 95% 單尾界 ≈ 3/n（Hanley & Lippman-Hand）。
    # 全成功 → 真實成功率下界 ≈ 1 - 3/n；全失敗 → 上界 ≈ 3/n。
    return max(0.0, 1.0 - 3.0 / n) if n > 0 else 0.0


def _percentile(sorted_vals, q: float) -> float:
    # 線性插值百分位（等同 numpy.percentile 預設 method="linear"）
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return float(sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo]))
    return float(sorted_vals[lo])


def bootstrap_ci(samples, n_boot: int = 2000, seed: int = 0):
    """樣本均值的 95% percentile bootstrap CI。

    退化處理（M1）：樣本全同值（含全 0 / 全 1，如 hit@k 全命中）時 percentile bootstrap
    必塌成單點、不具統計意義；改回報 rule-of-three 單尾界（全 1 → [1-3/n, 1.0]，
    全 0 → [0.0, 3/n]），避免印出假的 [1,1]/[0,0]。
    """
    if not samples:
        return (0.0, 0.0)
    s = [float(x) for x in samples]
    m = len(s)
    lo_v, hi_v = min(s), max(s)
    if lo_v == hi_v:
        if hi_v == 1.0:
            return (rule_of_three(m), 1.0)
        if lo_v == 0.0:
            return (0.0, 1.0 - rule_of_three(m))
        return (lo_v, hi_v)   # 全為同一非 0/1 常數
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(s) for _ in range(m)) / m for _ in range(n_boot))
    return (_percentile(means, 2.5), _percentile(means, 97.5))


def paired_bootstrap_delta(a, b, n_boot: int = 5000, seed: int = 0):
    """配對 bootstrap：同單位（同題）重抽，回 (Δ點估計, lo, hi)。Δ = mean(a) - mean(b)。

    配對控制了題目難度的共變，比 unpaired 更有檢定力。CI 跨 0 = 此測試集看不出差異。
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b:
        return (0.0, 0.0, 0.0)
    m = len(a)
    point = sum(a) / m - sum(b) / m
    rng = random.Random(seed)
    ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        ds.append(sum(a[i] for i in idx) / m - sum(b[i] for i in idx) / m)
    ds.sort()
    return (point, _percentile(ds, 2.5), _percentile(ds, 97.5))
