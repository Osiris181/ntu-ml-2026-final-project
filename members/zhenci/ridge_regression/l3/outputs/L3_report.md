# L3 — Ridge regression for next-session intraday return

**L3-B qualifies for one exploratory Phase 1 submission.** Keep L2-B as the Phase 1 reference while checking the Ridge candidate.

Against L2-B, L3-A changed the equal-year official score by **-0.00000360** and won
1 of three years. L3-B changed it by **+0.00001499** and won 2 years.
The reused 2023–2025 folds provide exploratory evidence.

## Design

The conditional expected official contribution for binary action a is
(2a − 1) E[r | X]. Predicting up when E[r | X] is positive maximizes that expectation.
Under unrestricted squared-error minimization the population optimum is E[r | X].
Ridge approximates it with a regularized linear function. Linear restrictions, shrinkage,
finite samples, and distribution shift can separate the fitted model from that optimum.

The experiment fits one Ridge per fold with alpha 1, an intercept, solver auto, and uniform
observation weights. The target is the adjusted open-to-close decimal return, including flats.
Preprocessing is fitted without return weights on each fold's original 100 training stocks.
Each state exactly matches the saved L1 and L2 state. Every comparison uses identical
50-stock evaluation keys and the shared scorer.

L3-A predicts up strictly above zero. L3-B uses +delta after a negative eligible TAIEX
return, −delta after a nonnegative one, and zero if missing. Delta is 0.05 times the
training fold's mean absolute return. The earlier weighted classification identity gives
2p_w(X) − 1 = E[r | X] / E[|r| | X]. The 0.475/0.525 offsets correspond to ±0.05
on its left side. Substituting an unconditional training mean for the conditional expected
magnitude is an approximation. L3-B is an empirical extension. Comparisons between
market-conditioned Ridge and logistic rules also change the output scale and cutoff.

## Official score and stability

| arm | 2023 | 2024 | 2025 | official_score | worst_four_month_score | worst_block |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.0006340 | 0.0002108 | 0.0008996 | 0.0005815 | 0.0001246 | 2024-B3 |
| E3 | 0.0006115 | 0.0003911 | 0.0009189 | 0.0006405 | 0.0002867 | 2024-B3 |
| L1-A | 0.0000547 | 0.0003951 | 0.0009787 | 0.0004762 | -0.0002793 | 2023-B3 |
| L1-B | -0.0000165 | 0.0006993 | 0.0007591 | 0.0004806 | -0.0001612 | 2023-B3 |
| L2-A | -0.0000259 | 0.0007991 | 0.0008642 | 0.0005458 | -0.0004985 | 2023-B3 |
| L2-B | 0.0001067 | 0.0009374 | 0.0006036 | 0.0005492 | -0.0001176 | 2023-B3 |
| L3-A | 0.0000478 | 0.0008058 | 0.0007832 | 0.0005456 | -0.0002600 | 2023-B3 |
| L3-B | 0.0001677 | 0.0009607 | 0.0005642 | 0.0005642 | 0.0000138 | 2023-B2 |

| arm | accuracy | direction_auc | mean_daily_spearman | predicted_up_rate | actual_up_rate | market_direction | stock_selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.5281501 | 0.5203427 | 0.0417554 | 0.3697531 | 0.4463689 | 0.0003647 | 0.0002167 |
| E3 | 0.5287080 | 0.5203427 | 0.0417554 | 0.3866761 | 0.4463689 | 0.0004121 | 0.0002284 |
| L1-A | 0.5442129 | 0.5234573 | 0.0442931 | 0.1802182 | 0.4463689 | 0.0002983 | 0.0001779 |
| L1-B | 0.5397739 | 0.5234573 | 0.0442931 | 0.2417845 | 0.4463689 | 0.0003065 | 0.0001741 |
| L2-A | 0.5013641 | 0.5182112 | 0.0610190 | 0.5970666 | 0.4463689 | 0.0001470 | 0.0003988 |
| L2-B | 0.5000873 | 0.5182112 | 0.0610190 | 0.6069371 | 0.4463689 | 0.0001930 | 0.0003562 |
| L3-A | 0.5014290 | 0.5124713 | 0.0529348 | 0.5913122 | 0.4463689 | 0.0001612 | 0.0003844 |
| L3-B | 0.5004431 | 0.5124713 | 0.0529348 | 0.5890088 | 0.4463689 | 0.0002129 | 0.0003513 |

Spearman here ranks the continuous outputs within each trading date. Market direction is
the daily mean signed position times the daily mean realized return. Stock selection is the
remaining daily official score. Period aggregates weight dates by their observation counts.
See [daily](daily_scores.csv), [monthly](monthly_scores.csv), [four-month](four_month_scores.csv)
and [stock-date scores and decisions](validation_predictions.csv).

| arm | control | period | delta | low_95 | high_95 |
| --- | --- | --- | --- | --- | --- |
| L3-A | L1-A | equal_year | 0.0000694 | -0.0003039 | 0.0004426 |
| L3-A | L1-B | equal_year | 0.0000650 | -0.0002554 | 0.0003908 |
| L3-A | L2-A | equal_year | -0.0000002 | -0.0001282 | 0.0001326 |
| L3-A | L2-B | equal_year | -0.0000036 | -0.0001835 | 0.0001798 |
| L3-A | E1 | equal_year | -0.0000358 | -0.0005076 | 0.0004172 |
| L3-A | E3 | equal_year | -0.0000949 | -0.0005279 | 0.0003174 |
| L3-B | L1-A | equal_year | 0.0000880 | -0.0003050 | 0.0004689 |
| L3-B | L1-B | equal_year | 0.0000836 | -0.0002335 | 0.0004016 |
| L3-B | L2-A | equal_year | 0.0000184 | -0.0001655 | 0.0001994 |
| L3-B | L2-B | equal_year | 0.0000150 | -0.0001136 | 0.0001507 |
| L3-B | E1 | equal_year | -0.0000173 | -0.0004928 | 0.0004739 |
| L3-B | E3 | equal_year | -0.0000763 | -0.0005034 | 0.0003579 |
| L3-B | L3-A | equal_year | 0.0000186 | -0.0001354 | 0.0001663 |

The [paired intervals](paired_uncertainty.csv) use 3,000 circular bootstrap samples with
five-trading-day blocks drawn within each year, retaining every stock from a sampled day.
The [historical screen](phase1_screen.csv) used the following rule before Phase 1 verification.
Compare each arm with L2-B on the same historical keys. Require a positive equal-year official-score difference, wins in at least two years, and a worst four-month score at least as high as L2-B's. Prefer the higher equal-year score among eligible arms, with ties going to L3-A. Use paired intervals to describe uncertainty.

## Numerical return estimation

| fold | split | rows | rmse_pct | mae_pct | pearson | spearman | r2_vs_zero | r2_vs_training_mean | predicted_mean_pct | predicted_sd_pct | realized_mean_pct | realized_sd_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | train_top100 | 73465 | 2.0456871 | 1.3893715 | 0.1250308 | 0.1234487 | 0.0166027 | 0.0156327 | -0.0647575 | 0.2575657 | -0.0647575 | 2.0618668 |
| 2023 | train_top50 | 36735 | 1.8872723 | 1.2537650 | 0.1178283 | 0.1126881 | 0.0139612 | 0.0139931 | -0.0315529 | 0.2502985 | -0.0314764 | 1.9003255 |
| 2023 | validation | 11950 | 1.6356457 | 1.0740049 | 0.0068584 | 0.0276439 | -0.0171397 | -0.0152945 | 0.0404248 | 0.2207356 | 0.0045290 | 1.6217998 |
| 2024 | train_top100 | 97364 | 1.9926714 | 1.3463515 | 0.1046985 | 0.1066393 | 0.0117614 | 0.0109618 | -0.0569946 | 0.2096488 | -0.0569946 | 2.0036836 |
| 2024 | train_top50 | 48685 | 1.8271693 | 1.2066564 | 0.0984266 | 0.0958723 | 0.0096849 | 0.0098811 | -0.0226888 | 0.2034474 | -0.0226387 | 1.8359425 |
| 2024 | validation | 12099 | 1.8286923 | 1.2741357 | 0.0669734 | 0.0559146 | 0.0015195 | 0.0013394 | 0.0435768 | 0.1938494 | -0.0337950 | 1.8297712 |
| 2025 | train_top100 | 121560 | 1.9935315 | 1.3570360 | 0.1037834 | 0.1015954 | 0.0117187 | 0.0107710 | -0.0620679 | 0.2079540 | -0.0620679 | 2.0043552 |
| 2025 | train_top50 | 60784 | 1.8264881 | 1.2196416 | 0.0956779 | 0.0867065 | 0.0091365 | 0.0093621 | -0.0248935 | 0.2014655 | -0.0248593 | 1.8347212 |
| 2025 | validation | 12140 | 1.7970571 | 1.2471170 | 0.1191676 | 0.0569042 | 0.0144212 | 0.0139104 | -0.0107922 | 0.2255567 | -0.0447093 | 1.8096046 |

Columns ending in pct use percentage return units. The full [regression table](regression_metrics.csv)
uses decimal returns, with MSE in squared decimal return units and quantiles for both actual
and predicted returns. Baseline R² is 1 − model MSE / baseline MSE, against either an all-zero
prediction or the mean from the corresponding Top100 training fold. Training rows summarize
in-sample fit. The [metric gaps](regression_gaps.csv) compare validation with each training universe.
Validation R² exceeded zero in 2 of three years against the
zero prediction and in 2 years against the
training-mean prediction. Direction AUC uses the continuous predicted returns as scores.

| fold | solver | fit_seconds | inference_seconds | mean_abs_training_return | delta | training_rows |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | cholesky | 0.5591184 | 0.0755324 | 0.0139683 | 0.0006984 | 73465 |
| 2024 | cholesky | 0.5872311 | 0.0690184 | 0.0135150 | 0.0006758 | 97364 |
| 2025 | cholesky | 0.6906087 | 0.0696363 | 0.0136143 | 0.0006807 | 121560 |

Fit time includes preprocessing and Ridge fitting. Inference time includes preprocessing
transformation and predicting all validation returns.

## Magnitude and residuals

The quintiles rank absolute predicted return within each fold without splitting exact ties.
The highest quintile earned a higher fixed-rule score than the lowest in 3
of three years. The [quintile table](prediction_magnitude_quintiles.csv) records mean predicted and
realized returns, accuracy, absolute realized return, and both within-group score and group
contribution to the fold score.

| fold | residual_p01 | residual_p50 | residual_p99 | top_realized_1pct_squared_error_share | top_realized_5pct_squared_error_share | top_realized_10pct_squared_error_share |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | -0.0463225 | -0.0007792 | 0.0523911 | 0.2452851 | 0.5425522 | 0.6901559 |
| 2024 | -0.0505669 | -0.0012027 | 0.0546061 | 0.1878036 | 0.4640897 | 0.6237398 |
| 2025 | -0.0453317 | -0.0010632 | 0.0578419 | 0.2017270 | 0.4762068 | 0.6327090 |

The [residual diagnostics](residual_diagnostics.csv) use realized minus predicted return.
Tail error shares quantify squared-loss concentration among large realized returns.

## Coefficients

| feature | 2023 | 2024 | 2025 | mean_abs | sign_flip |
| --- | --- | --- | --- | --- | --- |
| relative_amount_20d | 0.0048119 | 0.0032367 | 0.0021203 | 0.0033897 | False |
| relative_volume_20d | -0.0046150 | -0.0031154 | -0.0021401 | 0.0032902 | False |
| nasdaq_return | 0.0026785 | 0.0019043 | 0.0015820 | 0.0020549 | False |
| sp500_return | -0.0018617 | -0.0011385 | -0.0010203 | 0.0013402 | False |
| sma_distance | -0.0015377 | -0.0010610 | -0.0008972 | 0.0011653 | False |
| intraday_return_1d | -0.0013547 | -0.0009594 | -0.0011524 | 0.0011555 | False |
| momentum_5d | -0.0008560 | -0.0008497 | -0.0009865 | 0.0008974 | False |

| fold | L3_return_coef |
| --- | --- |
| 2023 | -0.0006474 |
| 2024 | -0.0005678 |
| 2025 | -0.0006200 |

For every validation row, the fitted intercept plus the encoded coefficient products recovered
the predicted return. The intercept represents the return at an all-zero encoded vector.
Actual average predictions are in the regression table. The [coefficient file](coefficients.csv)
compares signs and signed coefficient shares, normalized by each model's sum of absolute
encoded coefficients. This makes relative weights comparable without equating Ridge return
units with logistic log-odds. See the [full stability table](coefficient_stability.csv) and
[numerical, stock-ID and weekday contributions](feature_family_contributions.csv).
Coefficients describe conditional associations.

## TAIEX decisions

| fold | mean_abs_training_return | delta | changed | changed_fraction | down_to_up | up_to_down | official_score_delta | market_direction_delta | stock_selection_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 0.0139683 | 0.0006984 | 1337 | 0.1118828 | 647 | 690 | 0.0001199 | 0.0001053 | 0.0000145 |
| 2024 | 0.0135150 | 0.0006758 | 1419 | 0.1172824 | 679 | 740 | 0.0001549 | 0.0001235 | 0.0000314 |
| 2025 | 0.0136143 | 0.0006807 | 1471 | 0.1211697 | 746 | 725 | -0.0002191 | -0.0000738 | -0.0001452 |

The [regime-specific breakdown](threshold_sensitivity.csv) includes up rates, threshold distances,
direction switches, and their incremental signed returns. Summed switch contributions
reproduced the official score difference, including its market-direction and stock-selection parts.

## Earlier Phase 1 score verification

| arm | archived_cv_score | reported_phase1_score | locally_replayed_phase1_score | rounded_match | submission |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.0005815 | -0.0003179 | -0.0003179 | True | zhenci_e1_phase1.csv |
| E3 | 0.0006405 | -0.0000619 | -0.0000619 | True | zhenci_e3_taiex_phase1.csv |
| L1-A | 0.0004762 | -0.0001190 | -0.0001195 | True | zhenci_l1_a_exploratory_phase1.csv |
| L1-B | 0.0004806 | 0.0003040 | 0.0003037 | True | zhenci_l1_b_exploratory_phase1.csv |
| L2-A | 0.0005458 | — | — | — | — |
| L2-B | 0.0005492 | 0.0006590 | 0.0006593 | True | zhenci_l2_b_exploratory_phase1.csv |

All claimed historical means matched the repository archives. The local replay checked the
saved files' hashes and scored each existing submission on the exact 4,200 keys with archived
adjusted intraday returns. Competition-server receipts were unavailable. This verification
ran after the historical selection and covered earlier submissions only.
January–May 2026 and Phase 1 outcomes did not enter fitting or selection.

## Reproduction

Run from the repository root with a fresh output directory.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.ridge_regression.l3.train_l3 --output-dir members/zhenci/ridge_regression/l3/reproduction
```

[Historical manifest](summary.json) · [Phase 1 audit provenance](phase1_reference_provenance.json)
