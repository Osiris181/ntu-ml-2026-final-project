# 鄭振慈

Personal experiment workspace. The chosen primary approach is **XGBoost**.

[`train_e1.py`](train_e1.py) runs one **fixed, untuned** XGBoost configuration for seeds `42`, `64`, `128`, `256`, and `512` over the three expanding validation folds. It loads the shared [`F0 dataset`](../../data/processed/README.md), fits a fresh [baseline preprocessor](../../shared/preprocessing.py) on each training fold, predicts every target-stock validation row for each seed, and uses the [official shared scorer](../../shared/evaluation.py). It does not use the Jan–May 2026 holdout for selection, train a deployment model, or create a submission.

From the repository root, with `uv` installed:

```text
uv run python -m members.鄭振慈.train_e1
```

You can also run `uv run python members/鄭振慈/train_e1.py` from the repository root. The script resolves the shared package and its default data and results paths from the repository location, so direct file execution works too.

`uv run` installs the dependencies declared in `pyproject.toml` and creates a lockfile if needed. The script writes `fold_metrics.csv` (15 seed/fold rows), `validation_predictions.csv` (each prediction tagged with its seed), and `summary.json` (per-seed CV, five-seed mean and standard deviation, configuration, threshold, and package versions) under `results/zheng_xgboost_e1_five_seeds_v1/`. Those generated files are gitignored. The script refuses to overwrite an existing output directory; pass a different `--output-dir` for another run. See the [team guide](../../context/README.md) for the shared E1 inputs and outputs.

The XGBoost configuration uses all rows and columns (`subsample=1`, `colsample_bytree=1`), so seed changes may produce identical models and scores. Row and column sampling are model-specific choices for a separately recorded configuration.
