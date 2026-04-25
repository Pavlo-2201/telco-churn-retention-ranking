"""Сравнение моделей без обучения финальной модели: подбор параметров и
повторная CV с корректированным t-тестом (та же процедура, что в train.py).
Настройки — из configs/train.json; test не используется."""

import argparse
from pathlib import Path

from churn.config import DEFAULT_CONFIG_PATH, load_config
from churn.selection import comparison_table, run_model_selection
from churn.training import comparison_markdown, dev_and_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None, help="куда записать markdown-таблицу")
    args = parser.parse_args()
    config = load_config(args.config)
    (X, y, _), _ = dev_and_test(config)
    selection = run_model_selection(config, X, y)
    table = comparison_table(selection, tuple(config["ranking"]["top_fractions"]))
    markdown = comparison_markdown(table, selection["comparison"])
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
