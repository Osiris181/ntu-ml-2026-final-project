# 鄭振慈

Personal experiment workspace. The chosen primary approach is **XGBoost**. The first E1 runner is prepared, but no model has been trained and no results have been produced yet.

[`xgboost_e1.py`](xgboost_e1.py) runs one **fixed, untuned** XGBoost configuration over the three expanding validation folds. It loads the shared [`F0 dataset`](../../data/processed/README.md), fits a fresh [baseline preprocessor](../../shared/preprocessing.py) on each training fold, predicts every target-stock validation row, and uses the [official shared scorer](../../shared/evaluation.py). It never uses the Jan–May 2026 holdout for selection, trains a deployment model, or creates a submission.

From the repository root, with `uv` installed:

```text
uv run python -m members.鄭振慈.xgboost_e1
```

`uv run` installs the dependencies declared in `pyproject.toml` and creates a lockfile if needed. The script writes `fold_metrics.csv`, `validation_predictions.csv`, and `summary.json` under `results/zheng_xgboost_e1_initial_v1/`. Those generated files are gitignored. If you run a second experiment, pass a new `--output-dir` instead of overwriting the first. The configuration, threshold, package versions, and per-fold scores are recorded in `summary.json`. See the [team guide](../../context/README.md) for the shared E1 inputs and outputs.
