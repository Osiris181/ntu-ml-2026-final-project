# E15 cross-sectional rank normalization

**Retain the original E3 representation.** The original 21 F0 fields scored highest in each of the three historical validation years. Appending ranks came close in 2025 but offered no observed gain in the equal-year mean. Replacing the stock fields with ranks reduced the score more substantially. These repeatedly used folds provide exploratory evidence for this XGBoost setup. Generalization to a new period remains untested.

## Design

Gu, Kelly, and Xiu rank stock characteristics within each period. Kottas compares raw, ranked, and combined technical predictors and measures ordering with daily Spearman correlation. E15 applies those ideas to daily F0 inputs and measures the resulting binary decisions with the competition scorer. [Gu, Kelly, and Xiu (2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276), [Kottas (2026)](https://www.mdpi.com/2227-7099/14/9/393).

| Arm | Representation | Logical fields |
| --- | --- | ---: |
| E3 | Original F0 | 21 |
| E15-A | Replace the 15 stock-specific numerical fields with daily ranks. Keep the four global fields and two categories. | 21 |
| E15-B | Keep F0 and append the 15 daily ranks. | 36 |

For each feature and prediction date, the runner uses the available eligible F0 values from the fixed metadata `Top100@Y` universe. It computes ascending average ranks and maps them by `(rank - 1) / (N - 1)`, where `N` counts nonmissing eligible values for that feature and date. A singleton gets `0.5`, a constant group maps to `0.5` by average ranks, and an entirely missing group stays missing. Rankings are calculated before subsetting to the target stocks. The same function accepts a full eligible Top100 snapshot for inference and reports its coverage. It rejects a date with no non-target peer rows. The shared F0 timing policy supplies the lagged inputs and actual publication timestamps remain unverified.

Both rank representations use the shared median imputation and scaling policy on rank columns, fitted on the corresponding training fold. The runner checks exact equality of transformed globals and categories across arms. E15-B also reuses the original transformed stock fields. Each of the nine runs trains the archived XGBoost configuration with all 100 training stocks, seed 42, `binary:logistic`, and 100 rounds. All predictions use the original eligible TAIEX return and the E3 cutoff of `0.525` when negative, `0.475` otherwise, or `0.5` when missing. The runner verifies the retrained E3 probabilities within `1e-7` of the archived CSV, its binary decisions exactly, and its official scores to `1e-12` in all three folds. Evaluation keeps the same 11,950, 12,099, and 12,140 target-stock rows for 2023, 2024, and 2025 respectively, including the incomplete target-stock dates. Selection evidence comes from 2023–2025 only.

## Official scores and direction metrics

| Year | Arm | Official score | Difference vs. E3 | Accuracy | Predicted up | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | E3 | 0.00061150 | 0 | 52.08% | 47.60% | 0.262355 |
| 2023 | E15-A | 0.00028999 | −0.00032151 | 50.60% | 47.64% | 0.266531 |
| 2023 | E15-B | 0.00055691 | −0.00005458 | 51.71% | 47.34% | 0.264767 |
| 2024 | E3 | 0.00039108 | 0 | 52.65% | 34.90% | 0.262194 |
| 2024 | E15-A | 0.00015081 | −0.00024027 | 51.47% | 35.96% | 0.264723 |
| 2024 | E15-B | 0.00033467 | −0.00005641 | 51.95% | 35.38% | 0.262308 |
| 2025 | E3 | 0.00091892 | 0 | 53.88% | 33.51% | 0.255677 |
| 2025 | E15-A | 0.00067011 | −0.00024880 | 53.28% | 33.35% | 0.256985 |
| 2025 | E15-B | 0.00091267 | −0.00000625 | 53.58% | 32.98% | 0.254921 |
| Equal-year mean | E3 | **0.00064050** | 0 | | | |
| Equal-year mean | E15-A | 0.00037030 | −0.00027020 | | | |
| Equal-year mean | E15-B | 0.00060142 | −0.00003908 | | | |

E15-A also had worse accuracy, Brier score, and daily rank correlation in each year. E15-B improved Brier score in 2025 and daily rank correlation in 2023. Its official score fell in both cases. The up-call rates stayed close to E3 across all three representations.

## Probability and daily score distributions

| Year | Arm | Up probability p10 / median / p90 | Mean daily Spearman | Positive days | Daily official score p10 / median / p90 |
| --- | --- | --- | ---: | ---: | --- |
| 2023 | E3 | .313 / .485 / .676 | .0470 | 56.49% | −.003931 / .000575 / .005731 |
| 2023 | E15-A | .304 / .486 / .691 | .0404 | 53.14% | −.004468 / .000159 / .005284 |
| 2023 | E15-B | .307 / .484 / .677 | .0526 | 55.65% | −.003454 / .000388 / .004738 |
| 2024 | E3 | .285 / .442 / .622 | .0348 | 53.31% | −.005047 / .000414 / .005616 |
| 2024 | E15-A | .287 / .449 / .619 | .0318 | 56.20% | −.005914 / .000458 / .005298 |
| 2024 | E15-B | .293 / .447 / .612 | .0312 | 54.96% | −.005093 / .000478 / .005413 |
| 2025 | E3 | .306 / .444 / .610 | .0435 | 53.91% | −.004811 / .000521 / .006988 |
| 2025 | E15-A | .300 / .439 / .607 | .0322 | 51.85% | −.006117 / .000253 / .007431 |
| 2025 | E15-B | .308 / .444 / .598 | .0364 | 51.03% | −.004869 / .000375 / .007233 |

Spearman is calculated within each day between the raw predicted up probability and the adjusted open-to-close return, then averaged across days with defined correlations. Brier measures squared probability error against binary direction. The official score measures signed returns after applying the fixed cutoff. E15-A had more positive days in 2024 and a worse lower tail in every year.

## Paired uncertainty

| Period | E15-A − E3, 95% interval | E15-B − E3, 95% interval |
| --- | --- | --- |
| 2023 | [−0.00071463, +0.00007840] | [−0.00046349, +0.00033312] |
| 2024 | [−0.00072739, +0.00021430] | [−0.00047262, +0.00038081] |
| 2025 | [−0.00071249, +0.00022852] | [−0.00045561, +0.00043570] |
| Equal-year mean | [−0.00053481, −0.00001368] | [−0.00028303, +0.00020548] |

Intervals use 3,000 paired circular bootstrap draws with five trading days per block, separately within each year. A sampled day carries all of its stock rows, and each yearly score uses the sampled row counts so incomplete days contribute the same way as in the official scorer. Yearly draws are averaged with equal weights. The intervals describe uncertainty conditional on the fitted models and historical folds. They do not account for the repeated use of these years across earlier experiments.

## Missingness and reference coverage

| Validation year | Trading days | Days with 100 reference rows | Minimum reference rows | Minimum eligible values on any date and ranked feature |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 239 | 238 | 99 | 99 |
| 2024 | 242 | 239 | 98 | 98 |
| 2025 | 243 | 231 | 98 | 98 |

All present validation target rows have values for all 15 rank fields. In the 2020 warm-up, 81 date-feature groups have no eligible value. Across the three expanding training folds, average missingness across the 15 ranked fields is 0.74%, 0.55%, and 0.44% before imputation. The worst single rank field in each training fold has 2.72%, 2.05%, and 1.65% missing respectively. Edge checks exercised tied, constant, singleton, and wholly missing groups and rejection of a target-only snapshot. The saved predictions verify identical fold, stock, and date keys for all three arms.

## Reproduce

From `final_project`, choose a fresh output folder.

```powershell
uv run python -m members.zhenci.xgboost.e15.train_e15 --output-dir members/zhenci/xgboost/e15/reproduction
```

The [runner](train_e15.py) writes [fold metrics](outputs/fold_metrics.csv), [paired predictions](outputs/validation_predictions.csv), [daily metrics](outputs/daily_metrics.csv), [paired intervals](outputs/paired_uncertainty.csv), [reference coverage](outputs/reference_coverage.csv), [missingness](outputs/missingness.csv), and a [run summary](outputs/summary.json) with input hashes, versions, model parameters, and the reference IDs. The completed run used XGBoost 3.4.1.
