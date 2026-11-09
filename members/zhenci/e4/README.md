# E4: F2 intraday and overnight history

**Status: implementation prepared; validation has not been run as part of this change.** The F2 feature builder, fold-fitted preprocessor, paired validation runner, and synthetic checks are available. No E4 model artifact or submission file is provided. This is a member-owned XGBoost study; the shared F0 definition and team protocol remain unchanged.

## Files and run order

Run these from the repository root when you are ready. Both commands have default input and output paths, so no flags are required:

```text
uv run python -m members.zhenci.e4.features_e4
uv run python -m members.zhenci.e4.train_e4
```

[`features_e4.py`](features_e4.py) reads the unchanged historical market data and shared `data/processed/f0.csv`, then writes `data/processed/f2.csv`. It also combines the original and later market history as feature inputs, uses the shared F0 builder on that combined history, checks the official template keys, and writes `data/processed/f2_with_updates.csv`. Updated labels are ignored. Build historical `data/processed/f0.csv` first if it is absent. The updated F2 file is an input check for possible future inference; it is not used in the E4 validation comparison.

[`preprocessing_e4.py`](preprocessing_e4.py) applies the shared F0 preprocessing and fits median imputation plus standard scaling for the five added numeric fields on each training fold. [`train_e4.py`](train_e4.py) reads **historical** F2, checks its 21 F0 fields against the shared F0 file, and reads the saved full-sample E1 validation probabilities, configuration, and fixed E3 threshold rule. It compares the four prespecified arms below on the same keys; only E4 needs new XGBoost fits (one fit per fold and seed, reused for both E4 thresholds). By default it writes `fold_metrics.csv` for all four arms, `validation_predictions.csv` for E4 probabilities and binary outputs, `prediction_differences.csv` for paired label changes, `model_fit_runtime.csv`, and `summary.json` under `results/zhenci_xgboost_e4_f2_v1/`. It refuses to overwrite an existing results directory.

The synthetic checks are in [`test_zhenci_e4_features.py`](../../../tests/test_zhenci_e4_features.py) and [`test_zhenci_e4_runner.py`](../../../tests/test_zhenci_e4_runner.py). Result rows use the team's `experiment=E2` category for member optimization; **E4** names this local study.

## Question

Does separating recent **intraday** stock returns from recent **overnight** returns improve the official score beyond the existing F0 XGBoost model and its selected TAIEX threshold rule? Call the proposed feature set **F2**: the 21 shared F0 fields plus the five E4 fields below, for 26 logical fields total. E2's separate F1 additions are not included in F2.

E1 already includes the previous session's intraday return and five-session close-to-close momentum. E2 tested one previous overnight gap, ten-session close momentum, and five-session close volatility. E4 tests the *history of the two return components separately*. This is motivated by [Lou, Polk, and Skouras (2019)](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf), which reports different continuation and reversal patterns for overnight and intraday returns. Its US-market, longer-horizon findings are a hypothesis source, not evidence that E4 will work for this Taiwan next-session task.

## Five F2 numeric features

For each stock and completed trading session `s`, using adjusted prices:

```text
intraday[s]  = close[s] / open[s] - 1
overnight[s] = open[s] / close[s-1] - 1
```

For a prediction on trading session `t`, use only completed **per-stock** sessions. A window `t-n ... t-1` contains exactly `n` sessions. Require a full window; undefined ratios or insufficient history become missing, and the stock/date row remains present.

| Field | Definition |
|---|---|
| `e4_intraday_mean_5` | Mean of `intraday[s]` for `s=t-5 ... t-1` |
| `e4_intraday_mean_20` | Mean of `intraday[s]` for `s=t-20 ... t-1` |
| `e4_overnight_mean_5` | Mean of `overnight[s]` for `s=t-5 ... t-1` |
| `e4_overnight_mean_20` | Mean of `overnight[s]` for `s=t-20 ... t-1` |
| `e4_overnight_minus_intraday_20` | `e4_overnight_mean_20 - e4_intraday_mean_20` |

The most recent numerator/denominator used by these formulas is from `t-1`; no target-day open, close, label, or completed trading value enters E4. Compute all five fields from stock history **before** slicing chronological folds, then join them to shared F0 on exact `(stock_id, date)` keys. The date-only source still lacks verified publication timestamps, so the team's conservative `t-1` availability rule remains an assumption. These new fields use raw adjusted OHLC, not undocumented supplied indicator formulas.

## Prespecified comparison

Use the original **full-sample** E1 XGBoost configuration (`subsample=1.0`, `colsample_bytree=1.0`, 100 rounds) from `results/zheng_xgboost_e1_five_seeds_v1/summary.json`. The current `members/zhenci/e1/train_e1.py` defaults to a *different sampled* configuration; do not substitute it for this control. Keep the pooled 100-stock training universe, 50 target-stock validation keys, F0 fields, and the agreed three expanding folds. Train E4 for seeds `42, 64, 128, 256, 512` and report all 15 fold/seed scores. The five new numeric fields receive training-fold-only median imputation and standard scaling; shared F0 preprocessing and categorical handling stay the same.

Compare these four **predeclared** prediction arms on identical validation keys:

1. Full-sample E1 F0 probabilities with threshold `0.5` (existing control).
2. The same E1 probabilities with the fixed E3 TAIEX rule: prior `taiex_return < 0` uses `0.525`, nonnegative uses `0.475`, missing uses `0.5`.
3. E4 F2 probabilities with threshold `0.5`.
4. The same E4 F2 probabilities with that **unchanged** E3 TAIEX rule. This tests whether the already selected rule also helps the new model; it is not retuned for E4.

Reuse existing E1 validation probabilities only after checking their configuration and exact fold/seed/stock/date coverage. Select by the locked official score: an equal mean of the 2023, 2024, and 2025 target-stock scores for each seed, then the mean of the five seed scores. Report year scores, seed variability, and how many predictions differ. Identical seed results from a deterministic configuration are reported as identical, not treated as independent confirmation. Do **not** use the Jan–May 2026 holdout or Phase 1 submission feedback to choose features, parameters, or thresholds.

## Implementation scope and deployment gate

The member-owned builder, paired validation runner, and synthetic checks cover `t-1`/`t-2` boundaries, full rolling windows, missing history, key completeness, and agreement between historical and updated builds. Neither the builder nor the runner changes `shared/features.py` or the four-member F0 dataset.

If an E4 arm wins validation, it still needs a matching refitted model, fitted preprocessing, frozen feature definitions and prediction rule, plus a checked exporter that builds the five fields from updated historical stock data. `shared.updated_features` already loads updated market files as feature inputs without using their labels, but the E4 builder must explicitly support those inputs and every official template key before a Phase 1 file is generated. The team's agreed Phase 2 workflow currently specifies F0; using E4 there would require a separate team decision and reproducible live E4 inputs.

Another broad hyperparameter search is deferred. E2's fixed grid and E3's Optuna search did not beat the original E1 configuration, so E4 first tests whether these new input signals add value while holding XGBoost settings fixed. If this fails, a separate future experiment could test a two-part expected-return model, but it is not part of this predeclared E4 comparison.
