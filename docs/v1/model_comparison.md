| Модель | CV PR-AUC | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Параметры |
|---|---|---|---|---|---|---|
| dummy | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 | не измерено | не измерено | не измерено | `{}` |
| logreg | 0.6612 ± 0.0176 | 0.8459 ± 0.0083 | не измерено | не измерено | не измерено | `{"C": 1.0, "class_weight": null}` |
| random_forest | 0.6670 ± 0.0214 | 0.8470 ± 0.0095 | не измерено | не измерено | не измерено | `{"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 300}` |
| catboost (выбрана) | 0.6738 ± 0.0236 | 0.8502 ± 0.0089 | 0.8447 | 0.6661 | 0.63 | `{"depth": 4, "iterations": 500, "l2_leaf_reg": 10, "learning_rate": 0.03}` |
