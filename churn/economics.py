"""Экономика кампании удержания: прибыль от контакта с верхними K клиентами списка.

Все денежные параметры — допущения (configs/train.json, раздел business), а не данные
оператора. Модель прибыли для одного контакта:

- ушедший клиент с вероятностью retention_rate принимает скидку и остаётся:
  выигрыш customer_value - discount_cost; иначе уходит, скидку не использует;
- оставшийся клиент остался бы и так, но скидку берёт: потеря discount_cost;
- каждый контакт стоит contact_cost.

Прибыль(K) = TP·r·(V − d) − FP·d − K·c.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from churn.metrics import ranking_order

PARAMETERS = ("contact_cost", "discount_cost", "retention_rate", "customer_value")


@dataclass(frozen=True)
class Economics:
    contact_cost: float
    discount_cost: float
    retention_rate: float
    customer_value: float

    @classmethod
    def from_config(cls, business: dict) -> Economics:
        return cls(**{name: float(business[name]) for name in PARAMETERS})

    @property
    def gain_per_churner(self) -> float:
        return self.retention_rate * (self.customer_value - self.discount_cost) - self.contact_cost

    @property
    def loss_per_non_churner(self) -> float:
        return self.discount_cost + self.contact_cost

    def break_even_probability(self) -> float:
        """Вероятность ухода, при которой контакт окупается в среднем: p·gain = (1−p)·loss."""
        return self.loss_per_non_churner / (self.gain_per_churner + self.loss_per_non_churner)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def profit_curve(y_true, score, econ: Economics) -> np.ndarray:
    """Прибыль при контакте с верхними K клиентами, K = 0..n (элемент K)."""
    y_sorted = np.asarray(y_true)[ranking_order(score)]
    tp = np.concatenate([[0], np.cumsum(y_sorted)])
    fp = np.arange(len(y_sorted) + 1) - tp
    return tp * econ.gain_per_churner - fp * econ.loss_per_non_churner


def best_k(profit: np.ndarray, max_k: int | None = None) -> int:
    """K с максимальной прибылью (не больше max_k). При равенстве — меньший K."""
    limit = len(profit) - 1 if max_k is None else min(max_k, len(profit) - 1)
    return int(np.argmax(profit[: limit + 1]))


def operating_point(y_true, score, econ: Economics, budget_fraction: float) -> dict:
    """Выбранная точка: K с максимальной прибылью в пределах бюджета и порог оценки на этом K."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    profit = profit_curve(y_true, score, econ)
    max_k = int(round(n * budget_fraction))
    k = best_k(profit, max_k)
    k_free = best_k(profit)
    ordered = score[ranking_order(score)]
    threshold = float(ordered[k - 1]) if k > 0 else float(ordered[0]) + 1e-9
    y_top = np.asarray(y_true)[ranking_order(score)[:k]]
    return {
        "k": k,
        "k_fraction": k / n,
        "threshold": threshold,
        "profit": float(profit[k]),
        "profit_per_customer_in_base": float(profit[k] / n),
        "precision": float(y_top.mean()) if k else float("nan"),
        "recall": float(y_top.sum() / max(np.asarray(y_true).sum(), 1)),
        "budget_fraction": budget_fraction,
        "budget_binding": k_free > max_k,
        "unconstrained_k_fraction": k_free / n,
        "unconstrained_profit": float(profit[k_free]),
    }


def sensitivity(y_true, score, econ: Economics, factors=(0.5, 2.0), budget_fraction: float = 1.0) -> pd.DataFrame:
    """Как меняется оптимальный K, если сдвинуть одно допущение в factor раз."""
    rows = []
    n = len(score)
    for name in ("base", *PARAMETERS):
        for factor in ((1.0,) if name == "base" else factors):
            changed = econ if name == "base" else replace(econ, **{name: getattr(econ, name) * factor})
            if changed.retention_rate > 1:
                changed = replace(changed, retention_rate=1.0)
            profit = profit_curve(y_true, score, changed)
            k_free = best_k(profit)
            k_budget = best_k(profit, int(round(n * budget_fraction)))
            rows.append({
                "parameter": name,
                "factor": factor,
                "value": None if name == "base" else getattr(changed, name),
                "break_even_probability": changed.break_even_probability(),
                "optimal_k_fraction": k_free / n,
                "optimal_profit": float(profit[k_free]),
                "k_fraction_within_budget": k_budget / n,
                "profit_within_budget": float(profit[k_budget]),
            })
    return pd.DataFrame(rows)


def segment_cutoffs(y_true, score, segments, econ: Economics) -> dict:
    """Диагностика «один глобальный порог или свои пороги по сегментам»: прибыль
    глобального top-K против суммы оптимальных отсечек внутри каждого сегмента.
    Отсечки по сегментам подбираются на тех же данных, поэтому их прибыль —
    оптимистичная верхняя оценка."""
    y_true, score, segments = np.asarray(y_true), np.asarray(score, dtype=float), np.asarray(segments)
    global_profit = profit_curve(y_true, score, econ)
    k_global = best_k(global_profit)
    top = np.zeros(len(score), dtype=bool)
    top[ranking_order(score)[:k_global]] = True
    per_segment = {}
    total = 0.0
    for value in np.unique(segments):
        mask = segments == value
        profit = profit_curve(y_true[mask], score[mask], econ)
        k = best_k(profit)
        total += profit[k]
        per_segment[str(value)] = {
            "n": int(mask.sum()),
            "own_optimal_k": k,
            "own_optimal_profit": float(profit[k]),
            "k_in_global_top": int(top[mask].sum()),
            "profit_in_global_top": float(profit[int(top[mask].sum())]) if mask.any() else 0.0,
        }
    return {
        "global_k": k_global,
        "global_profit": float(global_profit[k_global]),
        "segment_cutoffs_profit": float(total),
        "segments": per_segment,
    }
