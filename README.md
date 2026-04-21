# tabular-ml-pipeline: прогноз оттока клиентов

Бинарная классификация оттока клиентов телеком-оператора (`Churn`) на датасете
IBM Telco Customer Churn. Полный цикл: воспроизводимая загрузка данных, EDA, базовые модели,
RandomForest и CatBoost, кросс-валидация с подбором параметров, выбор порога, однократная оценка
на test, интерпретация и анализ ошибок, обучение и прогноз из командной строки.

## Задача

По данным о клиенте (стаж, контракт, подключённые услуги, способ оплаты, платежи) предсказать,
уйдёт ли он. Положительный класс: `Churn = Yes`. Модель выдаёт вероятность и решение по порогу,
выбранному по максимуму F1 на out-of-fold предсказаниях.

## Данные

- Источник: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d),
  файл `data/Telco-Customer-Churn.csv`, коммит `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
  Это официальный пример IBM; источник доступен, замена на аналог не понадобилась.
- Лицензия репозитория: Apache-2.0.
- SHA-256 файла: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- 7043 клиента, 19 признаков + `customerID` + цель, доля оттока 26.5%.
- Данные в Git не хранятся: `python -m churn.download_data` скачивает файл и сверяет SHA-256.

## Структура

```
churn/                  пакет с кодом
  data.py               загрузка, проверка схемы, очистка, список признаков
  splits.py             фиксированное разбиение train/val/test
  features.py           предобработка для каждого типа модели
  models.py             фабрика моделей (Pipeline = предобработка + модель)
  tuning.py             кросс-валидация и GridSearchCV
  threshold.py          выбор порога
  metrics.py            метрики
  training.py           полный цикл обучения для train.py
  inference.py          инференс-артефакт ChurnModel
  config.py             чтение конфигурации
  check_data.py, compare_models.py, analysis.py   CLI отдельных этапов
configs/train.json      конфигурация полного обучения
configs/smoke.json      маленькая конфигурация для быстрой проверки
train.py, predict.py    обучение и прогноз
notebooks/01_eda.ipynb  EDA (только исследование)
results/                метрики, параметры, предсказания (JSON/CSV)
reports/figures/        графики
examples/               пример входа для predict.py
tests/                  pytest
```

## Установка

Проверено на Python 3.13, CPU (4 ядра), GPU не нужен.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

```bash
python -m churn.download_data            # скачать данные и проверить SHA-256
python train.py                          # полный цикл обучения (~4 мин на 4 CPU)
python predict.py --input examples/new_customers.csv --output predictions.csv
pytest -q                                # тесты (~30 с)
```

`train.py` делает всё без notebook: проверяет данные, берёт разбиение из `results/splits.csv`
(создаёт его, если файла нет), подбирает параметры всех моделей из `configs/train.json` на 5-fold CV,
выбирает лучшую по PR-AUC, подбирает порог по OOF-предсказаниям, обучает модель на train+val,
один раз оценивает её на test и сохраняет:

- `artifacts/churn_model.joblib`: инференс-артефакт `ChurnModel` (Pipeline с предобработкой,
  порог, список признаков, параметры, метрики, версии библиотек, конфигурация). В Git не
  хранится, воспроизводится командой `python train.py`;
- `results/cv_search.json`, `results/cv_candidates.csv`: результаты подбора;
- `results/final_evaluation.json`: порог и метрики OOF и test;
- `results/model_comparison.csv` и `.md`: таблица сравнения моделей;
- `results/oof_predictions.csv`, `results/test_predictions.csv`, `results/train_config.json`;
- `reports/figures/final_curves.png`: ROC и PR кривые на test, F1 от порога.

`python train.py --config configs/smoke.json` проходит тот же путь на маленьких сетках за несколько
секунд и пишет результаты в `smoke_run/` (не в Git).

`predict.py` принимает CSV в формате исходного датасета (`customerID` и `Churn` необязательны,
`TotalCharges` может быть пустым у новых клиентов) и пишет `customerID`, `churn_probability`,
`churn_prediction`.

Отдельные этапы, которые не входят в `train.py`:

```bash
python -m churn.check_data               # схема и распределение цели -> results/data_summary.json
python -m churn.compare_models --ablation   # модели на одной validation-выборке -> results/val_comparison.json
python -m churn.analysis                 # важность, ошибки, утечка -> results/analysis_summary.json (после train.py)
```

EDA: `notebooks/01_eda.ipynb` (выполнен, выводы сохранены). Перезапуск без Jupyter-интерфейса:

```bash
python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('notebooks/01_eda.ipynb', 4); NotebookClient(nb, record_timing=False, resources={'metadata': {'path': 'notebooks'}}).execute(); nbformat.write(nb, 'notebooks/01_eda.ipynb')"
```

## Результаты

Все числа получены реальным запуском `python train.py` (seed 42) на CPU (4 ядра).

Сравнение моделей (`results/model_comparison.md`), 5-fold CV на train+val, среднее ± стандартное
отклонение по fold:

| Модель | CV PR-AUC | CV ROC-AUC | Лучшие параметры |
|---|---|---|---|
| Dummy (prior) | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 | – |
| LogisticRegression | 0.6612 ± 0.0176 | 0.8459 ± 0.0083 | C=1.0, class_weight=None |
| RandomForest | 0.6670 ± 0.0214 | 0.8470 ± 0.0095 | max_depth=8, min_samples_leaf=5, max_features=sqrt, 300 деревьев |
| **CatBoost** (выбрана) | **0.6738 ± 0.0236** | **0.8502 ± 0.0089** | depth=4, learning_rate=0.03, l2_leaf_reg=10, 500 итераций |

Порог: 0.35 (по максимуму F1 на OOF-предсказаниях train+val). Финальная модель на test
(1409 клиентов, 374 ушедших):

| ROC-AUC | PR-AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| 0.8447 | 0.6661 | 0.6300 | 0.5604 | 0.7193 | 824 | 211 | 105 | 269 |

Модель находит 72% уходящих клиентов; из отмеченных моделью уходит 56%. OOF F1 при выбранном
пороге 0.640 против 0.605 при пороге 0.5.

- Метрики test близки к CV, переобучения на выборе параметров не видно.
- Отрыв CatBoost от логистической регрессии и леса меньше разброса между fold; на одних и тех же
  fold он лучше LogisticRegression во всех пяти и лучше RandomForest в четырёх из пяти.
- Остальные модели на test намеренно не оценивались («не измерено»): test использован один раз,
  для выбранной модели.

## Ограничения

- Порог выбран по F1, то есть цены FP и FN считаются равными. Для реальной кампании удержания
  порог нужно выбирать по стоимости удержания и потерь от ухода.
- Сетки параметров ограничены; лучшие параметры CatBoost на краю сетки.
- Подбор параметров и выбор модели идут на одних и тех же fold (без nested CV), поэтому средние
  CV-метрики лучшей модели слегка оптимистичны; test-оценка этим не затронута.
