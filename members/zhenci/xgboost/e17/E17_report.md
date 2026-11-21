# E17 date-dependent weighted binary cross-entropy

**Retain E3 as the historical reference.** Partial balancing lowered the official score in all three validation years. Full balancing improved 2024 and lost in 2023 and 2025. The paired intervals for both equal-year differences include zero. Four exploratory Phase 1 files cover both balancing strengths and both fixed decision rules.

## Design and controls

The absolute label remains `adjusted close > adjusted open`. In each fold, the weight for a stock on date `t` is `(1 − λ) + λ × N_t / (2 × N_class)`, with weights of 1 on single-class dates. The mixed-class formula gives each class half the date’s total weight at λ = 1. The partial rule applies half that adjustment. The total weight stays equal to the number of eligible rows on each date. A check with constructed data verified the unweighted, partial, full, and single-class cases. The largest absolute error in the saved daily totals was `5.7e-14`.

All arms keep the 21 F0 fields, train-only shared preprocessing, full 100-stock training universe, absolute labels, original full-sample tree settings, 100 boosting rounds, and seed 42. The model uses `XGBClassifier.fit(..., sample_weight=...)` with `binary:logistic`. E17 uses a small classifier subclass to keep the same `DMatrix` construction as archived E3, since the standard histogram wrapper may choose `QuantileDMatrix`. The unweighted classifier reproduced the archived probabilities within `1e-7`, the E1 and E3 calls exactly, and their official scores to `1e-12` on all folds. [XGBoost classifier API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier.fit).

The primary rule uses the original E3 thresholds of 0.525 after a negative eligible TAIEX return, 0.475 after a nonnegative one, and 0.5 when missing. The fixed 0.5 rule reuses the very same model probabilities. E16-A uses archived probabilities of beating its daily peer median. All seven configurations share the original 11,950, 12,099, and 12,140 evaluation keys in 2023, 2024, and 2025 respectively. The score is unweighted for every configuration. Phase 1 and 2026 outcomes played no role in configuration selection.

## Official scores

| Arm | Decision rule | 2023 | 2024 | 2025 | Equal-year mean |
| --- | --- | ---: | ---: | ---: | ---: |
| E3 | TAIEX | 0.00061150 | 0.00039108 | 0.00091892 | **0.00064050** |
| E17-A | TAIEX | 0.00030340 | 0.00019451 | 0.00059623 | 0.00036471 |
| E17-B | TAIEX | 0.00034937 | 0.00061029 | 0.00041982 | 0.00045983 |
| E1 | Fixed 0.5 | 0.00063396 | 0.00021084 | 0.00089955 | 0.00058145 |
| E17-A-050 | Fixed 0.5 | 0.00032291 | 0.00007570 | 0.00048691 | 0.00029518 |
| E17-B-050 | Fixed 0.5 | 0.00029938 | 0.00045348 | 0.00055917 | 0.00043734 |
| E16-A | Fixed 0.5 | 0.00030170 | 0.00062076 | 0.00067828 | 0.00053358 |

| Period | E17-A minus E3 [95% interval] | E17-B minus E3 [95% interval] |
| --- | ---: | ---: |
| 2023 | −0.00030810 [−0.00080982, +0.00017948] | −0.00026213 [−0.00086485, +0.00028057] |
| 2024 | −0.00019658 [−0.00069309, +0.00026233] | +0.00021921 [−0.00036181, +0.00078698] |
| 2025 | −0.00032269 [−0.00084919, +0.00021487] | −0.00049910 [−0.00121470, +0.00023131] |
| Equal-year | −0.00027579 [−0.00057085, +0.00000697] | −0.00018067 [−0.00055188, +0.00018396] |

Under the fixed 0.5 rule, both weighted models trailed E1 and E16-A on the equal-year mean. Full balancing beat E1 in 2024 only and trailed E16-A in each year. The bootstrap used 3,000 paired circular draws of five consecutive trading days within each fold, with all stocks kept together on each sampled date. Yearly scores used sampled row counts, and the yearly draws were averaged equally. All primary yearly intervals and all secondary equal-year intervals include zero. These are exploratory comparisons on reused validation years.

## Predictions, probability error, and daily behavior

| Year | Arm | Direction accuracy | Up calls | Unweighted BCE | Brier | Mean daily Spearman | Daily score SD | Negative days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | E3 | 52.08% | 47.60% | 0.72192 | 0.26235 | 0.0470 | 0.00391 | 43.51% |
| 2023 | E17-A | 51.43% | 50.08% | 0.71529 | 0.25966 | 0.0469 | 0.00345 | 48.54% |
| 2023 | E17-B | 51.45% | 53.82% | 0.71098 | 0.25797 | 0.0465 | 0.00264 | 41.84% |
| 2024 | E3 | 52.65% | 34.90% | 0.72181 | 0.26219 | 0.0348 | 0.00502 | 46.69% |
| 2024 | E17-A | 51.06% | 41.29% | 0.71465 | 0.25962 | 0.0217 | 0.00394 | 47.93% |
| 2024 | E17-B | 51.33% | 49.91% | 0.70692 | 0.25615 | 0.0362 | 0.00339 | 41.74% |
| 2025 | E3 | 53.88% | 33.51% | 0.70752 | 0.25568 | 0.0434 | 0.00578 | 46.09% |
| 2025 | E17-A | 52.74% | 37.86% | 0.70200 | 0.25369 | 0.0343 | 0.00409 | 43.62% |
| 2025 | E17-B | 50.69% | 51.28% | 0.70884 | 0.25702 | 0.0365 | 0.00343 | 46.09% |

For example, in 2025 the 10th, 50th and 90th percentiles move from `0.306 / 0.444 / 0.610` for E3 to `0.356 / 0.467 / 0.596` for partial balancing and `0.375 / 0.500 / 0.622` for full balancing. The weighted models issued more up calls and had lower daily score variability across all three years. Full balancing had a better mean daily Spearman only in 2024. The [fold metrics](outputs/fold_metrics.csv) include all probability summaries, daily means and quantiles, and the metrics for the fixed 0.5 comparison. E16-A’s BCE and Brier entries are blank because it predicts a different event. For E16-A only, the saved score column carries relative outperformance probability, as indicated by `probability_target`.

Partial balancing lowered unweighted BCE and Brier in each year while lowering the official score. Those probability losses assess direction forecasts and do not by themselves establish calibration. The experiment provides no separate evidence about overfitting or about the cause of differences across market regimes.

## Weight diagnostics

| Fold | λ | Min / median / p95 / p99 / max | Effective sample size | Weight > 2 | Weight > 5 | Weight > 10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2023 | 0.5 | 0.755 / 0.931 / 1.426 / 1.971 / 13 | 68,187 (92.8%) | 666 | 16 | 2 |
| 2023 | 1.0 | 0.510 / 0.862 / 1.852 / 2.941 / 25 | 56,098 (76.4%) | 2,640 | 136 | 16 |
| 2024 | 0.5 | 0.755 / 0.939 / 1.393 / 1.889 / 13 | 91,041 (93.5%) | 693 | 18 | 4 |
| 2024 | 1.0 | 0.510 / 0.877 / 1.786 / 2.778 / 25 | 76,196 (78.3%) | 3,126 | 147 | 18 |
| 2025 | 0.5 | 0.755 / 0.939 / 1.393 / 1.889 / 13 | 114,064 (93.8%) | 805 | 23 | 4 |
| 2025 | 1.0 | 0.510 / 0.877 / 1.786 / 2.778 / 25 | 96,257 (79.2%) | 3,860 | 166 | 18 |

At λ = 0, all weight quantiles equal 1 and effective sample sizes equal the training counts of 73,465, 97,364, and 121,560. There were zero single-class training dates. The aggregate up share rose from 44.39–44.67% unweighted to 47.19–47.33% under partial balancing and 50% under full balancing. The median absolute daily deviation from 50% was 13–14 percentage points before weighting, 6.5–7 points with partial balancing, and zero with full balancing. The corresponding 95th percentiles were 35–37, 17.5–18.5, and zero percentage points. Fit times ranged from 1.19–1.77 seconds for the weighted models on this run, with weight calculation under 0.02 seconds per fold.

Weights above 2 covered roughly 0.66–0.91% of partially balanced rows and 3.18–3.59% of fully balanced rows. These rows came from the minority class on strongly one-sided dates, and roughly three quarters had positive returns. Their median absolute intraday returns were about 0.65–0.69%, versus 0.88–0.92% for other rows. Their average absolute-return percentile within the same day was also lower. The weights amplified unusual **direction relative to peers** even when the absolute price move was small. This check does not establish that those observations were noise or that they caused the score losses.

## Phase 1 files and reproduction

The two weighted models were refitted on the same 145,847 eligible rows from 100 stocks through 2025-12-31. Each frozen model supplied both decision rules. All four files were checked against the official template, including 50 stocks, 84 dates, and 4,200 binary cells. They are ready for manual upload.

| Model | TAIEX rule | Fixed 0.5 rule |
| --- | --- | --- |
| Partial | [CSV](../../../../submissions/zhenci_e17_a_taiex_phase1.csv), 1,364 up | [CSV](../../../../submissions/zhenci_e17_a_050_phase1.csv), 1,266 up |
| Full | [CSV](../../../../submissions/zhenci_e17_b_taiex_phase1.csv), 1,950 up | [CSV](../../../../submissions/zhenci_e17_b_050_phase1.csv), 1,855 up |

The comparison files are [E3](../../../../submissions/zhenci_e3_taiex_phase1.csv), [E1](../../../../submissions/zhenci_e1_phase1.csv), and [E16-A](../../../../submissions/zhenci_e16_a_exploratory_phase1.csv).

From `final_project`, use distinct output paths for reproduction. The run used XGBoost 3.4.1 and scikit-learn 1.9.1.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e17.train_e17 --output-dir members/zhenci/xgboost/e17/reproduction
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e17.export_e17_phase1 --cv-dir members/zhenci/xgboost/e17/reproduction --submission-dir submissions/e17_reproduction --artifact results/zhenci_e17_reproduction_artifact
```

Saved outputs include [out-of-sample probabilities](outputs/validation_predictions.csv), [daily scores](outputs/daily_metrics.csv), [paired intervals](outputs/paired_uncertainty.csv), [daily class and weight totals](outputs/daily_weight_diagnostics.csv), [aggregate weight statistics](outputs/weight_summary.csv), [high-weight stock rows](outputs/extreme_weight_rows.csv), [runtime](outputs/runtime.csv), and [run provenance](outputs/summary.json).
