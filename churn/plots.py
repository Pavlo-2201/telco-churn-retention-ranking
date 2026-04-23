"""Графики отчёта (matplotlib, PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.economics import Economics, profit_curve
from churn.metrics import cumulative_gains

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, TEXT, MUTED = "#8a8984", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gains_and_profit(y, score, econ: Economics, point: dict, sensitivity: pd.DataFrame, path: Path,
                          fractions=(0.1, 0.2), title_suffix="OOF, train") -> None:
    """Кривая cumulative gains, кривая прибыли с выбранной точкой и чувствительность K*."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    contacted, captured = cumulative_gains(y, score)
    ax = axes[0]
    ax.plot(contacted * 100, captured * 100, color=BLUE, label="модель")
    ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--", label="случайный порядок")
    for f in fractions:
        i = int(round(f * len(y)))
        ax.plot(contacted[i] * 100, captured[i] * 100, "o", color=BLUE, ms=8, mec="white", mew=2)
        ax.annotate(f"{f:.0%} базы → {captured[i]:.0%} ушедших", (contacted[i] * 100, captured[i] * 100),
                    xytext=(8, -14), textcoords="offset points", color=TEXT, fontsize=9)
    ax.set(title=f"Cumulative gains ({title_suffix})", xlabel="охвачено клиентов, % (по убыванию риска)",
           ylabel="найдено ушедших, %", xlim=(0, 100), ylim=(0, 102))
    ax.legend(loc="lower right")

    ax = axes[1]
    profit = profit_curve(y, score, econ)
    k = np.arange(len(profit)) / len(y) * 100
    ax.plot(k, profit / 1000, color=BLUE)
    ax.axvline(point["budget_fraction"] * 100, color=GREY, lw=1, ls="--")
    ax.annotate("бюджет", (point["budget_fraction"] * 100, ax.get_ylim()[0]), xytext=(4, 6),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.plot(point["k_fraction"] * 100, point["profit"] / 1000, "o", color=ORANGE, ms=9, mec="white", mew=2)
    ax.annotate(f"K = {point['k_fraction']:.0%}, прибыль {point['profit'] / 1000:.0f} тыс. у.е.",
                (point["k_fraction"] * 100, point["profit"] / 1000), xytext=(8, 6), textcoords="offset points",
                color=TEXT, fontsize=9)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(title=f"Ожидаемая прибыль кампании ({title_suffix}; допущения)",
           xlabel="контакт с верхними K% списка", ylabel="тыс. у.е.", xlim=(0, 100))

    ax = axes[2]
    sens = sensitivity[sensitivity["parameter"] != "base"]
    base = sensitivity[sensitivity["parameter"] == "base"].iloc[0]
    labels, lows, highs = [], [], []
    for name, part in sens.groupby("parameter", sort=False):
        labels.append(name)
        lows.append(part.loc[part["factor"] < 1, "optimal_k_fraction"].iloc[0] * 100)
        highs.append(part.loc[part["factor"] > 1, "optimal_k_fraction"].iloc[0] * 100)
    pos = np.arange(len(labels))
    for p, lo, hi in zip(pos, lows, highs):
        ax.plot([lo, hi], [p, p], color="#d0cfca", lw=2, zorder=1)
    ax.scatter(lows, pos, color=AQUA, s=60, zorder=2, label="допущение × 0.5", edgecolor="white", linewidth=1.5)
    ax.scatter(highs, pos, color=ORANGE, s=60, zorder=2, label="допущение × 2", edgecolor="white", linewidth=1.5)
    ax.axvline(base["optimal_k_fraction"] * 100, color=BLUE, lw=1.5, ls="--", label="базовые допущения")
    ax.set_yticks(pos, labels)
    ax.set(title="Оптимальный K при сдвиге одного допущения", xlabel="оптимальный K, % базы (без бюджета)",
           xlim=(0, 100))
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_calibration(calibration: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="идеальная калибровка")
    for key, color, label in (("raw", BLUE, "без калибратора"), ("calibrated", ORANGE, "sigmoid-калибратор")):
        curve = calibration[key]["reliability_curve"]
        ax.plot([c["mean_predicted"] for c in curve], [c["observed_rate"] for c in curve], "-o", color=color,
                ms=5, label=f"{label}: Brier {calibration[key]['brier']:.4f}")
    ax.set(title="Reliability-кривая (OOF, train)", xlabel="средняя предсказанная вероятность в корзине",
           ylabel="доля ушедших в корзине", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def plot_segments(table: pd.DataFrame, path: Path, fractions=(0.1, 0.2)) -> None:
    """Lift внутри сегмента и ROC-AUC внутри сегмента с 95% CI."""
    metrics = [f"lift@{round(f * 100)}" for f in fractions] + ["roc_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.6))
    segments = table["segment"].tolist()
    pos = np.arange(len(segments))
    for ax, metric in zip(axes, metrics):
        value = table[metric].to_numpy()
        err = np.vstack([value - table[f"{metric}_ci_low"].to_numpy(), table[f"{metric}_ci_high"].to_numpy() - value])
        ax.errorbar(value, pos, xerr=err, fmt="o", color=BLUE, ecolor="#9fbfe8", elinewidth=3, ms=8, mec="white",
                    mew=2)
        for p, v in zip(pos, value):
            ax.annotate(f"{v:.2f}", (v, p), xytext=(0, 9), textcoords="offset points", ha="center", color=TEXT,
                        fontsize=9)
        ref = 0.5 if metric == "roc_auc" else 1.0
        ax.axvline(ref, color=GREY, lw=1, ls="--")
        ax.set_yticks(pos, segments)
        ax.set_ylim(-0.6, len(segments) - 0.4)
        ax.invert_yaxis()
        title = "ROC-AUC внутри сегмента" if metric == "roc_auc" else f"{metric} внутри сегмента"
        ax.set(title=f"{title} (OOF, 95% CI)")
    _save(fig, path)


def plot_test_gains(y_test, proba_test, y_oof, proba_oof, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for y, p, color, label in ((y_oof, proba_oof, BLUE, "OOF, train"), (y_test, proba_test, ORANGE, "test")):
        contacted, captured = cumulative_gains(y, p)
        ax.plot(contacted * 100, captured * 100, color=color, label=label)
    ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--", label="случайный порядок")
    ax.set(title="Cumulative gains: OOF и test", xlabel="охвачено клиентов, %", ylabel="найдено ушедших, %",
           xlim=(0, 100), ylim=(0, 102))
    ax.legend(loc="lower right")
    _save(fig, path)
