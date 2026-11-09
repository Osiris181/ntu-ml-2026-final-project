# E5: F0 feature-ablation experiment

This is a **validation experiment** for the XGBoost member track. It asks whether removing selected F0 inputs improves the official score compared with the existing E3 XGBoost model and fixed TAIEX decision rule. Running the script trains fold models and writes results; it does not save a deployment model.

## Run

From the `final_project` directory, run the script with its default inputs and output path:

```text
uv run python -m members.zhenci.e5.train_e5_ablation
```

The default output is `results/zhenci_xgboost_e5_f0_ablation_v1/`. The script refuses to overwrite that directory; use `--output-dir PATH` for a later run. Other optional flags are `--data-dir`, `--f0-path`, `--e1-summary`, and `--e3-summary`. Use the **original** `data/processed/f0.csv`, not the updated submission feature file.

## What the script compares

It starts with the same full-sample E1 XGBoost configuration, 100-stock training universe, three expanding validation folds, five required seeds, fold-fitted shared F0 preprocessor, and 50-stock evaluation keys. It checks that its unchanged F0 control reproduces the archived E1 and E3 fold scores. Its baseline to beat is **E3 with the fixed prior-TAIEX rule**, whose saved mean CV official score is about `0.00064050`.

The choices below were fixed before this experiment. They come from an earlier read-only audit of feature drift, correlation, and model importance. Removing a feature changes only the columns passed to XGBoost; the shared F0 file and its lag rules stay unchanged. The shared preprocessor is still fitted on each training fold and applied to that fold's validation year.

| Model input | Removed F0 fields | Reason to test |
|---|---|---|
| F0 control | None | Reproduce the existing benchmark |
| No relative amount | `relative_amount_20d` | Nearly identical rank order to relative volume in historical data |
| No RSI | `rsi_normalized` | Strongly correlated with SMA distance |
| No SMA distance | `sma_distance` | Paired alternative to removing RSI |
| No Nasdaq return | `nasdaq_return` | Strongly correlated with S&P 500 return |
| No S&P 500 return | `sp500_return` | Paired alternative to removing Nasdaq return |
| No ATR | `atr_normalized` | Large 2026 distribution shift |
| No volatility group | `price_range_1d`, `atr_normalized`, `bb_width` | Test the related shifted volatility inputs together |

Each model is evaluated with both the normal `0.5` cutoff and the **unchanged E3 TAIEX rule**: threshold `0.525` after a negative prior TAIEX return, `0.475` after a nonnegative one, and `0.5` if TAIEX is missing. The rule always reads the original lagged F0 `taiex_return`, even when a different input field is removed from the model. The script does not search for new thresholds.

## How to read the results

The runner writes four files:

| File | Contents |
|---|---|
| `summary.json` | Every arm's year, seed, and mean CV official scores; the selected arm; its difference from E3; and the exact removed fields |
| `fold_metrics.csv` | Official score, accuracy, predicted-up rate, training target-stock score, and training/validation AUC for each fold and seed |
| `logical_feature_importance.csv` | Mean normalized XGBoost split gain across the 15 fits for each model input; removed fields are marked explicitly |
| `selected_validation_predictions.csv` | Validation probabilities and binary predictions for the selected arm on all fold/seed target keys |

Select by the team's equal-weight mean of the three yearly official scores for each seed, then the mean over five seeds. The script reports `ablation_beats_e3` only when the selected arm strictly exceeds the existing E3 control. The in-sample training metrics and gain scores are **diagnostics**, never selection criteria. A low gain score alone does not prove a feature is useless; correlated fields can share split credit.

This run uses only original historical data through 2025 for training and validation. It does not use the Jan–May 2026 independent holdout, later market files, or Phase 1 outcomes to choose a feature set. The feature candidates were proposed after a diagnostic look at 2026 input distributions, so any CV winner remains exploratory. A CV improvement is not a guarantee of a better submission score. No model artifact or submission is created. After the run, inspect yearly scores and the training/validation gap before deciding whether to refit a candidate or design another model.
