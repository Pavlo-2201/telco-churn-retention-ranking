"""Проверяет схему и печатает распределение цели."""

import argparse
import json
from pathlib import Path

from churn.data import (
    DATA_SHA256,
    DATA_URL,
    DEFAULT_DATA_PATH,
    download_data,
    load_raw,
    target_distribution,
    validate_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/data_summary.json"))
    args = parser.parse_args()

    path = download_data(args.path)
    df = load_raw(path)
    validate_schema(df)
    distribution = target_distribution(df)

    summary = {
        "source_url": DATA_URL,
        "sha256": DATA_SHA256,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_distribution": distribution,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Строк: {len(df)}, столбцов: {df.shape[1]}. Схема в порядке.")
    for label, stats in distribution.items():
        print(f"Churn={label}: {stats['count']} ({stats['share']:.1%})")
    print(f"Сводка сохранена в {args.output}")


if __name__ == "__main__":
    main()
