# E18 expanded technical features

**Retain E3.** The combined set of 79 additions won in 2024 and 2025, but finished slightly behind E3 on the equal-year official score. Its worst four-month block was materially weaker. No standalone family improved the three-year mean, and no arm met the frozen Phase 1 criterion. The results support retaining the combined arm as an exploratory comparison for a fresh period. They provide little reason to keep expanding the catalogue on these same folds.

## Design and data checks

The [feasibility audit](feasibility_audit.md) covers the original and updated schemas, input coverage, F0, and overlaps with E2, E3 and E4. The [frozen formula catalogue](feature_catalogue.json) records the implementation status and raw columns for all 80 proposed additions. One exact F0 duplicate, `relative_amount_20d`, is excluded from the additions. Kottas describes a broader 196-feature catalogue. The supplement attachment was unavailable for inspection, so E18 uses the predefined feature families and independently specified formulas. [Kottas (2026), §3.3](https://www.mdpi.com/2227-7099/14/9/393#sec3dot3).

Each stock window ends at its latest eligible completed session. Market aggregates use the latest completed supplied-market date and the available members of the same Top100 universe. Relative stock features use peers from the stock’s own source date. The [feature provenance](outputs/feature_provenance.csv) preserves both latest source dates and session indices, and the [stock-session lookup](outputs/stock_source_sessions.csv) makes windows traceable. Market session indices and valid contributor counts are in the [coverage table](outputs/market_coverage.csv). Reference coverage was 98–100 supplied stocks per observed session over 2020–2025. All saved, nonmissing source dates were earlier than their target dates. Tests passed for target-day changes, appended future observations, missing stock sessions, invalid denominators, warm-up, and checks of exact momentum and compounded returns.

E18 independently fits shared median imputation, standardization, and categorical encoding within each training fold. It applies the same numerical policy to additions. All arms keep the existing `XGBClassifier` matrix path, full sample, uniform weights, binary logistic objective, seed 42, 100 rounds, and fixed prior-TAIEX thresholds. The expanding windows train on all 100 stocks and keep the original target-stock keys, including dates with fewer than 50 covered targets. The retrained E3 probabilities matched the archive to `1e-7`, binary decisions matched exactly, and scores matched to `1e-12` in all years. No 2026 outcome entered historical evaluation or selection. Shared F0 retains its previously documented publication-time and source-indicator assumptions.

## Official score by year

| Arm | Added features | Logical / encoded | 2023 | 2024 | 2025 | Equal-year mean | Difference vs E3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3 | Original F0 | 21 / 124 | 0.00061150 | 0.00039108 | 0.00091892 | **0.00064050** | 0 |
| E18-A | Intraday structure and return history | 42 / 145 | 0.00031141 | 0.00042878 | 0.00071560 | 0.00048526 | −0.00015524 |
| E18-B | Trend, volatility and downside risk | 52 / 155 | 0.00044953 | 0.00048147 | 0.00054493 | 0.00049198 | −0.00014852 |
| E18-C | Volume and liquidity | 32 / 135 | 0.00027706 | 0.00030835 | 0.00078399 | 0.00045647 | −0.00018403 |
| E18-D | Market context and peer-relative measures | 37 / 140 | 0.00033773 | 0.00030590 | 0.00087037 | 0.00050466 | −0.00013583 |
| E18-E | All four families | 100 / 203 | 0.00044898 | 0.00044769 | 0.00101257 | 0.00063642 | −0.00000408 |

A and B each improve 2024 alone. C and D trail E3 in every year. D is the strongest standalone family by mean score, and the complete set comes considerably closer than any single family. This comparison does not identify which interactions, if any, explain the combined score.

| Arm | Equal-year paired 95% interval for the official-score difference vs E3 |
| --- | --- |
| E18-A | [−0.00047406, +0.00013858] |
| E18-B | [−0.00046142, +0.00014330] |
| E18-C | [−0.00045208, +0.00007671] |
| E18-D | [−0.00048110, +0.00022840] |
| E18-E | [−0.00035515, +0.00033194] |

The [paired intervals](outputs/paired_uncertainty.csv) use 3,000 circular bootstrap draws of five consecutive trading days per block, paired across methods and sampled separately within each year. All target rows from a sampled day move together and yearly scores are normalized by sampled row counts before averaging years equally. These are intervals for the historical comparison conditional on the fitted models. The same folds have been inspected across earlier experiments.

## Temporal stability and contributions

| Arm | Negative months out of 36 | Monthly score SD | Four-month score SD | Worst four-month block | Worst score |
| --- | ---: | ---: | ---: | --- | ---: |
| E3 | 10 | 0.000948 | 0.000431 | Sep–Dec 2024 | 0.000287 |
| E18-A | 13 | 0.000932 | 0.000365 | Sep–Dec 2024 | 0.000118 |
| E18-B | 13 | 0.000968 | 0.000318 | May–Aug 2025 | −0.000030 |
| E18-C | 13 | 0.000920 | 0.000529 | Jan–Apr 2024 | −0.000089 |
| E18-D | 10 | 0.000933 | 0.000443 | May–Aug 2023 | −0.000030 |
| E18-E | 10 | 0.000950 | 0.000486 | Sep–Dec 2024 | 0.000027 |

| Fixed block | E3 | E18-E |
| --- | ---: | ---: |
| Jan–Apr 2023 | 0.000498 | 0.000518 |
| May–Aug 2023 | 0.000887 | 0.000508 |
| Sep–Dec 2023 | 0.000427 | 0.000330 |
| Jan–Apr 2024 | 0.000498 | 0.000818 |
| May–Aug 2024 | 0.000396 | 0.000524 |
| Sep–Dec 2024 | 0.000287 | 0.000027 |
| Jan–Apr 2025 | 0.001659 | 0.001785 |
| May–Aug 2025 | 0.000356 | 0.000774 |
| Sep–Dec 2025 | 0.000826 | 0.000557 |

The combined arm beats E3 in 18 of 36 months and five of the nine blocks. Its daily score standard deviation is modestly lower than E3’s in each year, while its monthly standard deviation is almost identical and its four-month standard deviation is higher. Its late-2024 block gives up about 0.000260 relative to E3. All six arms’ individual months, fixed blocks, daily distributions, and contributions are available in the [monthly](outputs/monthly_scores.csv), [four-month](outputs/four_month_scores.csv), and [daily](outputs/daily_scores.csv) tables.

For each date the market-direction contribution is the mean signed call `(2 × prediction − 1)` multiplied by the mean realized return. The stock-selection contribution is the covariance of signed calls and realized returns within that date, using a population denominator. The two sum to the daily official score. Aggregation uses the original daily observation counts. For E18-E, the higher mean stock-selection contribution (about **0.000286** vs **0.000228**) was offset by lower market-direction contribution (about **0.000350** vs **0.000412**). This decomposition describes the evaluated predictions and does not identify the causal effect of adding a particular field.

## Probability and feature diagnostics

| Year | Arm | Accuracy | Predicted up | BCE | Brier | Median probability | Mean daily Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | E3 | 52.08% | 47.60% | 0.72192 | 0.26235 | 0.48471 | 0.0470 |
| 2023 | E18-E | 51.33% | 43.70% | 0.73293 | 0.26693 | 0.46837 | 0.0573 |
| 2024 | E3 | 52.65% | 34.90% | 0.72181 | 0.26219 | 0.44238 | 0.0348 |
| 2024 | E18-E | 52.03% | 36.82% | 0.72912 | 0.26512 | 0.44546 | 0.0354 |
| 2025 | E3 | 53.88% | 33.51% | 0.70752 | 0.25568 | 0.44418 | 0.0434 |
| 2025 | E18-E | 53.61% | 36.98% | 0.71362 | 0.25805 | 0.44909 | 0.0425 |

The combined model’s unweighted BCE and Brier are worse in all three years. Its mean daily Spearman is higher in two years. In 2025, it improves the return-weighted score despite lower direction accuracy and worse probability error. [Full fold metrics](outputs/fold_metrics.csv) include every arm’s probability mean, standard deviation, 10th, 50th and 90th percentiles, and daily score distributions.

The 79 additions comprise **21 A, 31 B, 11 C and 16 D**. No numerical field was constant across an entire fold split, wholly missing, or infinite in the historical panel. Some market fields are intentionally constant across stocks within each date. Warm-up accounts for most training missingness. In the first training window, the maximum per-feature missing rate was 8.30%, driven by long windows. Validation missingness among the additions was confined to Family A candle ratios, whose daily range can be zero. No evaluation row was removed. [Missingness and finite-value checks](outputs/feature_diagnostics.csv).

The training folds contain 15, 18 and 16 numeric feature pairs with absolute Pearson correlation of at least 0.95. In the latest training fold, the five-session volume and amount ratios correlated at 0.998, and the supplied normalized ATR and new 20-session ATR at 0.988. In the complete arm, the mean fraction of total XGBoost split gain was 27.9% F0, 12.3% A, 15.8% B, 7.9% C and 36.1% D. Family D’s gain share describes fitted-tree use and coexists with a lower standalone score. [High correlations](outputs/high_correlations.csv), [encoded feature gains](outputs/feature_importance.csv), [family gains](outputs/family_importance.csv).

## Reproduce and use the saved artifacts

From `final_project`. The completed run used XGBoost 3.4.1 and scikit-learn 1.9.1.

```powershell
uv run --with scikit-learn==1.9.1 python -m unittest members.zhenci.xgboost.e18.test_features_e18 -v
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e18.train_e18 --output-dir members/zhenci/xgboost/e18/reproduction
```

The experiment fitted 18 fold models on identical stock-date keys. The [model directory](outputs/models) contains one saved booster and preprocessing state per year and arm. Reloading the 2025 combined arm reproduced all 12,140 saved probabilities to `1e-7`. The complete arm took 2.69–4.06 seconds for each model fit, and historical feature construction took 5.26 seconds on this run. See [runtime](outputs/runtime.csv), [saved validation probabilities and calls](outputs/validation_predictions.csv), [stability and selection](outputs/stability.csv), and [versions and input hashes](outputs/summary.json).

The predefined Phase 1 criteria were a higher equal-year mean than E3, wins in at least two years, and a worst four-month score at least as high as E3’s. E18-E met the year-count criterion and missed the other two. The historical comparison selected no candidate. The existing [E3 submission file](../../../../submissions/zhenci_e3_taiex_phase1.csv) remains available.

At the user’s subsequent request, an exploratory E18-E submission was prepared with the frozen feature catalogue and E3 TAIEX rule. The final refit used 145,847 eligible training rows from 100 stocks through December 2025. The CSV preserves the official template order and contains 4,200 binary predictions, including 1,316 up calls. It differs from E3 on 1,405 calls. [Phase 1 CSV](../../../../submissions/zhenci_e18_e_exploratory_phase1.csv), [exporter](export_e18_phase1.py).

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e18.export_e18_phase1 --output submissions/zhenci_e18_e_reproduction_phase1.csv
```
