# E6: Chronological early stopping for XGBoost

E5 found that removing individual or related F0 features did not beat the existing E3 control. Its full F0 model had average **in-sample training AUC around 0.83** and **future-year validation AUC around 0.52**. E6 tests whether choosing the boosting-round count from an earlier year reduces that gap and improves the official score. This is a member-owned validation study, not a change to shared F0 or the team's E1 baseline.

## Run

From `final_project`:

```text
uv run python -m members.zhenci.e6.train_e6
```

The defaults use the original `data/processed/f0.csv`, saved full-sample E1 probabilities, the fixed E3 TAIEX rule, and output directory `results/zhenci_xgboost_e6_nested_stop_v1/`. The script refuses to overwrite an existing output directory. Optional flags allow different paths if needed.

## How each outer fold works

| Outer score year | Inner fitting years | Inner validation year | Final refit years |
|---|---|---|---|
| 2023 | 2020–2021 | 2022 | 2020–2022 |
| 2024 | 2020–2022 | 2023 | 2020–2023 |
| 2025 | 2020–2023 | 2024 | 2020–2024 |

For each seed and each of two fixed model settings, the script fits on the inner training years and uses **target-stock inner validation log loss** to choose a boosting-round count. It allows at most 300 rounds and stops after 20 rounds without inner improvement. It then fits a new model on that outer fold's full training history using the selected round count. The outer validation year is used only to calculate the experiment's official score; it never chooses that fold's round count.

XGBoost's native early-stopping call returns the last trained booster, so E6 reads its zero-based `best_iteration`, adds one, and **refits** on the full outer training window with exactly that many rounds. See the [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train).

The two tested settings are:

1. The original E1 tree settings with an inner-selected round count.
2. Shallower, more regularized trees (`max_depth=3`, `eta=0.1`, `min_child_weight=20`, `lambda=5`, `alpha=0.1`) with an inner-selected round count.

Both retain all 21 logical F0 fields, the shared fold-fitted preprocessor, all 100 training stocks, the 50 target evaluation stocks, and seeds `42, 64, 128, 256, 512`. Each model is evaluated under both `0.5` and the **unchanged** E3 TAIEX rule. The two saved E1/E3 controls are scored on the same outer keys and checked against their archived results. The model settings and stopping policy are fixed before running E6.

## Results and next decision

| Output | Contents |
|---|---|
| `summary.json` | All arm scores, selected arm, difference from E3, exact inner/outer policy, input hashes |
| `fold_metrics.csv` | Official score, accuracy, predicted-up rate, and training/validation AUC by outer year and seed |
| `selected_rounds.csv` | Inner validation year, selected rounds, best inner log loss, and fit times |
| `selected_validation_predictions.csv` | Probabilities and binary predictions for the selected arm |

The selection criterion is the same equal-year, equal-seed mean official score used throughout this member's experiments. The incumbent to beat is E3's `0.00064050` CV score. Training AUC, inner log loss, and the 2026 submission result are **diagnostics, not selection criteria**. Five identical seed results are possible because these settings use no row or column sampling.

This runner uses no 2026 holdout labels or later market files, and it does not create a deployment artifact or submission. Even a higher three-fold CV score would remain exploratory after several member experiments; inspect each year's result before deciding on a refit or a new score-aligned model.
