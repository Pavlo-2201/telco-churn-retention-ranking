"""Статистика: bootstrap-интервалы и корректированный t-тест для повторной CV."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def bootstrap(
    statistic: Callable[[np.ndarray], dict[str, float]],
    n_rows: int,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Перцентильные интервалы: statistic(indices) -> {имя: значение} считается на
    n_resamples выборках строк с возвращением. NaN (например, AUC в выборке без
    ушедших) пропускаются."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        idx = rng.integers(0, n_rows, n_rows)
        for key, value in statistic(idx).items():
            draws.setdefault(key, []).append(value)
    alpha = (1 - confidence) / 2
    out = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=float)
        if np.isnan(values).all():
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha)))
    return out


def with_ci(point: dict[str, float], ci: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {key: {"value": value, "ci_low": ci[key][0], "ci_high": ci[key][1]} for key, value in point.items()}


def corrected_ttest(differences, n_train: int, n_test: int, confidence: float = 0.95) -> dict[str, float]:
    """Корректированный t-тест Nadeau & Bengio (2003) для разностей метрик на
    повторной k-fold CV (в форме Bouckaert & Frank, 2004).

    Фолды пересекаются по обучающим данным, поэтому разности зависимы и обычный
    t-тест занижает дисперсию. Поправка: var * (1/J + n_test/n_train), J = k*r.
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]  # фолды, где метрика не определена (например, AUC без ушедших)
    j = len(d)
    if j < 2:
        nan = float("nan")
        return {"mean_diff": nan, "ci_low": nan, "ci_high": nan, "t": nan, "p_value": nan, "n_folds": j}
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = np.sqrt((1 / j + n_test / n_train) * var)
    if se == 0:
        t_stat, p_value = (np.inf if mean else 0.0), (0.0 if mean else 1.0)
    else:
        t_stat = mean / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df=j - 1))
    half = float(stats.t.ppf(0.5 + confidence / 2, df=j - 1) * se)
    return {
        "mean_diff": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "t": float(t_stat),
        "p_value": float(p_value),
        "n_folds": j,
    }
