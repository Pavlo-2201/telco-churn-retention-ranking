# tabular-ml-pipeline: прогноз оттока клиентов

Бинарная классификация оттока (`Churn`) на датасете IBM Telco Customer Churn.

## Данные

- Источник: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d),
  файл `data/Telco-Customer-Churn.csv`, коммит `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
- Лицензия репозитория: Apache-2.0. Это официальный пример IBM, источник доступен (проверено 2026-01-12).
- SHA-256 файла: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- 7043 клиента, 21 столбец, доля оттока 26.5%.

Данные в Git не хранятся, их скачивает скрипт.

## Установка и запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m churn.download_data   # скачивание и проверка SHA-256
python -m churn.check_data      # проверка схемы, распределение цели -> results/data_summary.json
pytest -q
```
