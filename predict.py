"""Оценка риска ухода для клиентов и отбор в список для удержания."""

import argparse
from pathlib import Path

import pandas as pd

from churn.inference import ChurnModel, select_for_retention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_model.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget", type=int, default=None,
        help="сколько клиентов можно включить в список (K); без него в список попадают все выше порога",
    )
    args = parser.parse_args()
    if args.budget is not None and args.budget < 0:
        parser.error("--budget должен быть неотрицательным")

    model = ChurnModel.load(args.model)
    records = pd.read_csv(args.input, dtype={"TotalCharges": "object"})
    proba = model.predict_proba(records)

    result = pd.DataFrame(index=records.index)
    if "customerID" in records.columns:
        result["customerID"] = records["customerID"]
    result["churn_probability"] = proba.round(6)
    result["in_retention_list"] = select_for_retention(proba, model.threshold, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    above = int((proba >= model.threshold).sum())
    selected = int(result["in_retention_list"].sum())
    print(
        f"Модель {model.model_name}, порог {model.threshold:.3f}: {len(result)} клиентов, выше порога {above}, "
        f"в список для удержания попали {selected}"
        + (f" (бюджет {args.budget})" if args.budget is not None else "")
        + f". Сохранено в {args.output}"
    )
    fraction = model.budget_fraction
    if args.budget is None and fraction is not None and above > len(result) * fraction:
        print(
            f"Внимание: выше порога {above / len(result):.1%} клиентов, а при обучении бюджет был {fraction:.0%} базы. "
            "Чтобы ограничить список, задайте --budget."
        )


if __name__ == "__main__":
    main()
