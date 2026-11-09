# E2: XGBoost optimization experiments

These are member-owned XGBoost validation studies. Each uses the shared chronological folds, five seeds, target keys, and official scorer. The [member overview](../README.md) explains each comparison and its output directory.

| File | Use |
|---|---|
| [`train_e2.py`](train_e2.py) | Compare a fixed four-configuration XGBoost parameter grid on F0. |
| [`features_e2.py`](features_e2.py) | Build three extra stock-history features and preprocess them per training fold. |
| [`train_e2_features.py`](train_e2_features.py) | Compare the selected grid setting on F0 versus those added features. |
| [`train_e2_weighted.py`](train_e2_weighted.py) | Compare ordinary direction training with return-magnitude training weights. |
| [`train_e2_return_regression.py`](train_e2_return_regression.py) | Compare the direction classifier with direct return regression. The `--sampled` option tests row and column sampling. |

Run from the repository root using `uv run python -m members.zhenci.e2.<script_name>`, for example:

```text
uv run python -m members.zhenci.e2.train_e2
```

These scripts write validation results under their existing `results/` paths. They do not refit deployment models or submit files.
