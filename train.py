"""Обучение модели оттока от данных до инференс-артефакта."""

import argparse
from pathlib import Path

from churn.config import load_config
from churn.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
