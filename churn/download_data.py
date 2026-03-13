"""Скачивает датасет."""

import argparse
from pathlib import Path

from churn.data import DATA_URL, DEFAULT_DATA_PATH, download_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--force", action="store_true", help="скачать заново")
    args = parser.parse_args()
    path = download_data(args.path, force=args.force)
    print(f"Данные: {path} (источник {DATA_URL}), контрольная сумма совпадает")


if __name__ == "__main__":
    main()
