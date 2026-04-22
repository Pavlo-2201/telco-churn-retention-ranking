"""Синтетический набор в схеме Telco Customer Churn для тестов и CI без сети.

Значения правдоподобны по форме (те же столбцы, категории, пустой TotalCharges
у клиентов с tenure 0), уход зависит от контракта, стажа, интернета и способа
оплаты. Для выводов о реальных клиентах непригоден.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYNTHETIC_PATH = Path("data/synthetic/telco_synthetic.csv")


def make_synthetic_raw(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def pick(values, p=None):
        return rng.choice(values, size=n_rows, p=p)

    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.22, 0.23])
    tenure = np.where(
        contract == "Month-to-month", rng.integers(0, 40, n_rows), rng.integers(0, 73, n_rows)
    ).astype(int)
    tenure[:4] = 0  # гарантируем клиентов без начислений
    internet = pick(["DSL", "Fiber optic", "No"], [0.35, 0.45, 0.2])
    phone = pick(["Yes", "No"], [0.9, 0.1])

    def addon():
        return np.where(internet == "No", "No internet service", pick(["Yes", "No"]))

    payment = pick(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
    )
    monthly = np.round(
        np.where(internet == "Fiber optic", 85, np.where(internet == "DSL", 55, 22)) + rng.normal(0, 8, n_rows), 2
    ).clip(18.25, 118.75)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n_rows), 2)
    total_str = np.where(tenure == 0, " ", total.astype(str))

    logit = (
        -1.0
        + 1.4 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
        - 0.04 * tenure + 0.8 * (internet == "Fiber optic") + 0.6 * (payment == "Electronic check")
    )
    churn = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    # в каждом сегменте контракта должны быть оба класса
    for value in ("Month-to-month", "One year", "Two year"):
        idx = np.flatnonzero(contract == value)
        churn[idx[:2]] = True
        churn[idx[2:4]] = False

    return pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n_rows)],
        "gender": pick(["Female", "Male"]),
        "SeniorCitizen": pick([0, 1], [0.84, 0.16]).astype("int64"),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure.astype("int64"),
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", pick(["Yes", "No"])),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": addon(),
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"], [0.6, 0.4]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly.astype("float64"),
        "TotalCharges": total_str,
        "Churn": np.where(churn, "Yes", "No"),
    })


def write_synthetic(path: Path = DEFAULT_SYNTHETIC_PATH, n_rows: int = 300, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_raw(n_rows, seed).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Синтетические данные: {write_synthetic(args.output, args.rows, args.seed)}")


if __name__ == "__main__":
    main()
