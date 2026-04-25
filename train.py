"""Обучение модели оттока от данных до инференс-артефакта.

Без --evaluate-test test-выборка не используется: выбор модели, калибровка и
точка работы определяются на кросс-валидации внутри train.
"""

import argparse
from pathlib import Path

from churn.config import DEFAULT_CONFIG_PATH, load_config
from churn.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="один раз оценить финальную модель на test (обращение пишется в results/test_access_log.json)",
    )
    args = parser.parse_args()
    run_training(load_config(args.config), evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
