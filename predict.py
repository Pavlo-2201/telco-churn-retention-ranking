"""Оценка риска ухода для клиентов и отбор в список для удержания."""

import argparse
from pathlib import Path

import pandas as pd

from churn.inference import ChurnModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_model.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = ChurnModel.load(args.model)
    records = pd.read_csv(args.input, dtype={"TotalCharges": "object"})
    proba = model.predict_proba(records)

    result = pd.DataFrame(index=records.index)
    if "customerID" in records.columns:
        result["customerID"] = records["customerID"]
    result["churn_probability"] = proba.round(6)
    result["in_retention_list"] = (proba >= model.threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"Модель {model.model_name}, порог {model.threshold:.3f}: {len(result)} клиентов, "
        f"в список для удержания попали {int(result['in_retention_list'].sum())}. Сохранено в {args.output}"
    )


if __name__ == "__main__":
    main()
