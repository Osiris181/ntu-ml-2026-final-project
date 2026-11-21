# Selected XGBoost model: static E3

E3 uses the original full-sample E1 probability model with a decision threshold conditioned on the latest eligible prior TAIEX return. The deployed fit is static and ends on **2025-12-31**. January–May 2026 remains reserved for the team’s already-inspected holdout. The selected recipe is separate from the best documented personal Phase 1 score, which was user-reported for E16-A.

## Inputs, fitting and decisions

| Component | Frozen specification |
| --- | --- |
| Universe | Metadata `Top100@Y` for training and `Top50@Y` for validation and submissions |
| Training | Available stock sessions from 2020-01-01 through 2025-12-31, 145,847 rows, uniform weights, seed 42 for the saved refit |
| Target | `1` when adjusted close exceeds adjusted open, `0` otherwise, including ties |
| Historical comparison | Expanding training ending 2022, 2023 and 2024, each evaluated on the full following year. Original E1 used seeds 42, 64, 128, 256, and 512. The unsampled recipe reproduced the same historical score across those seeds. |
| Trees | `gbtree`, `hist`, `binary:logistic`, log-loss metric, 100 rounds, depthwise growth, maximum depth 6, learning rate 0.30, `min_child_weight=1`, `reg_lambda=1`, `reg_alpha=0`, `gamma=0`, `max_leaves=0`, `subsample=1`, `colsample_bytree=1`, `nthread=4` |
| Preprocessing | Replace nonfinite numeric inputs with missing, fit medians on training rows, impute, then standardize with the training mean and population SD after imputation. Zero SD uses scale 1. Map missing categories to `Unknown` and one-hot encode using the frozen vocabulary and order. Unseen categories produce all-zero columns for that field. The saved fit has 124 encoded columns. |
| Decision | Predict 1 only when the probability is **strictly greater** than 0.525 after a negative eligible TAIEX move, 0.475 after a nonnegative move, or 0.5 when the value is missing. The decision reads the original, untransformed F0 TAIEX value. |
| Official score | Mean of `(2 × prediction − 1) × (adjusted_close − adjusted_open) / adjusted_open` across eligible stock-date pairs |

F0 contains the following columns in order. [Shared feature source](../../../../shared/features.py), [shared preprocessing](../../../../shared/preprocessing.py).

```text
intraday_return_1d, daily_return_1d, momentum_5d, price_range_1d,
relative_volume_20d, relative_amount_20d, trade_size_ratio_20d,
foreign_ownership, foreign_net_buy_ratio, trust_net_buy_ratio,
rsi_normalized, sma_distance, macd_normalized, atr_normalized, bb_width,
vix_normalized, nasdaq_return, sp500_return, taiex_return, stock_id, day_of_week
```

Stock price, activity, and supplied technical values use the last completed stock session (`t−1`). Institutional values and their denominators use `t−2`. Each global input uses its own latest observation strictly before the prediction date, up to five calendar days old. The team uses these conservative date rules as its pre-09:00 Asia/Taipei information policy. Actual publication times, historical revisions, and the settings of supplied technical indicators remain unverified.

## Artifacts and inference

| Purpose | Repository path |
| --- | --- |
| Frozen booster | `results/zhenci_e1_full_pre_holdout_artifact/model.json` |
| Fitted preprocessing | `results/zhenci_e1_full_pre_holdout_artifact/preprocessor.json` |
| Training recipe, feature order and hashes | `results/zhenci_e1_full_pre_holdout_artifact/manifest.json` |
| Selected decision rule | `results/zhenci_xgboost_e3_market_regime_v2/summary.json` |
| Historical Phase 1 exporter | `members/zhenci/xgboost/e3/predict_e3_market_regime.py` |
| One-day Phase 2 wrapper | `members/zhenci/xgboost/e3/export_phase2.py` |

The artifact directory retains its E1 name because E3 uses the same fitted probability model. The exporter checks model, preprocessing, and shared-definition hashes. The historical E3 decision score averages **0.00064050** across the three full validation years, versus **0.00058145** for the fixed E1 decision on the same probabilities. [Archived E3 threshold comparison](../../../../results/zhenci_xgboost_e3_market_regime_v2/summary.json).

From `final_project`, use XGBoost **3.4.1**, the version recorded by the frozen artifact. With the original dataset and supplied November 7 update, the following commands reproduce the historical Phase 1 procedure when writing to fresh paths.

```powershell
uv run python -m shared.updated_features --update-dir updated_data_7_Nov/data --template sample_submission.csv --output data/processed/f0_phase1_replay.csv
uv run --with xgboost==3.4.1 python -m members.zhenci.xgboost.e3.predict_e3_market_regime --f0-path data/processed/f0_phase1_replay.csv --template sample_submission.csv --output submissions/zhenci_e3_phase1_replay.csv
```

For a prospective Phase 2 session, use the official one-date template, the actual prior completed trading date, and a cumulative feature update available before the session cutoff. The wrapper validates the source dates and hashes before applying the same frozen preprocessor, booster and TAIEX rule. Replace the placeholders below with actual inputs.

```powershell
uv run --with xgboost==3.4.1 python -m members.zhenci.xgboost.e3.export_phase2 --template path/to/session_template.csv --update-dir path/to/cumulative_update --previous-session YYYY-MM-DD --output path/to/new_e3_predictions.csv
```

The wrapper and dated November inputs may be local workspace files. Transfer the actual frozen artifacts, rule summary, exporter, data sources and template if reproducing on another machine. Fitting is unnecessary for inference. If the original frozen files are unavailable, the existing `members.zhenci.xgboost.e1.save_e1` module records the original seed-42 refit procedure and supports a separate artifact destination. Verify any reconstruction against the archive before treating it as the same model.

The synthesis checked the frozen artifact hashes and replayed both archived files in memory, with exact binary agreement on **4,200 Phase 1 calls** and **50 November 18 calls**. The latter includes nine calls of 1 and has no verified competition score in the repository. The replay establishes reproducible inference on those saved inputs. A timely prospective record additionally requires a before-deadline timestamp and independently available outcomes. [Phase 1 archive](../../../../submissions/zhenci_e3_taiex_phase1.csv), [November 18 archive](../../../../submissions/zhenci_e3_taiex_2026-11-18.csv).
