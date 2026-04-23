import numpy as np
import pytest

from churn.economics import Economics, best_k, operating_point, profit_curve, segment_cutoffs, sensitivity

ECON = Economics(contact_cost=10, discount_cost=100, retention_rate=0.3, customer_value=800)


def test_unit_economics():
    assert ECON.gain_per_churner == pytest.approx(0.3 * 700 - 10)
    assert ECON.loss_per_non_churner == 110
    p = ECON.break_even_probability()
    assert p * ECON.gain_per_churner == pytest.approx((1 - p) * ECON.loss_per_non_churner)


def test_profit_curve_by_hand():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    profit = profit_curve(y, score, ECON)
    g, loss = ECON.gain_per_churner, ECON.loss_per_non_churner
    np.testing.assert_allclose(profit, [0, g, g - loss, 2 * g - loss, 2 * g - 2 * loss])


def test_operating_point_respects_budget():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = (rng.random(1000) < p).astype(int)
    free = operating_point(y, p, ECON, budget_fraction=1.0)
    capped = operating_point(y, p, ECON, budget_fraction=0.1)
    assert capped["k"] <= 100 and capped["budget_binding"]
    assert free["k"] >= capped["k"] and free["profit"] >= capped["profit"]
    # при калиброванных вероятностях оптимальный порог близок к точке безубыточности
    assert abs(free["threshold"] - ECON.break_even_probability()) < 0.1
    assert best_k(profit_curve(y, p, ECON), 0) == 0


def test_sensitivity_moves_k_in_expected_direction():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < p).astype(int)
    table = sensitivity(y, p, ECON, factors=(0.5, 2.0)).set_index(["parameter", "factor"])
    base = table.loc[("base", 1.0), "optimal_k_fraction"]
    assert table.loc[("customer_value", 2.0), "optimal_k_fraction"] >= base  # ценнее клиент — шире охват
    assert table.loc[("discount_cost", 2.0), "optimal_k_fraction"] <= base  # дороже скидка — уже охват


def test_segment_cutoffs_upper_bound():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 600)
    y = (rng.random(600) < p).astype(int)
    seg = rng.choice(["a", "b", "c"], 600)
    res = segment_cutoffs(y, p, seg, ECON)
    # отдельные отсечки подбираются на тех же данных и не могут дать меньше глобальной
    assert res["segment_cutoffs_profit"] >= res["global_profit"] - 1e-9
    assert sum(s["k_in_global_top"] for s in res["segments"].values()) == res["global_k"]
