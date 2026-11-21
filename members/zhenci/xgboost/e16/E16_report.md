# E16 relative-performance binary classification

**Retain original E3.** With the same daily ranking rule, the relative target raised the equal-year official score from **0.00036589 to 0.00048966**, with gains in 2024 and 2025 and a loss in 2023. The paired 95% interval for that gain crosses zero. Both relative-target decision rules scored below original E3 on the three-year mean.

## Design and checks

Wolff and Echterling trained classifiers to predict stock returns above a weekly cross-sectional median. E16 tests a daily intraday version on the existing Taiwanese stock universe and uses the competition’s signed-return score for the resulting binary calls. [Stock picking with machine learning](https://tuprints.ulb.tu-darmstadt.de/27203/1/FOR_FOR3021.pdf).

For each training fold, each stock’s adjusted open-to-close return is compared with the median return of the available eligible training stocks on that date. A return strictly above the median receives label 1. Ties receive 0. As in the existing return-modeling and shared scoring checks, prices must be finite and adjusted opens positive. Invalid outcomes fail validation explicitly. Absent stock-date rows contribute nothing to the median. Every existing eligible training row passed the checks and was retained. Evaluation labels use the same strict comparison against the **eligible target-stock median** for each validation date. They are calculated after inference and used only for evaluation.

E16 reused the 21 original F0 fields and the shared training-fitted preprocessor, the 100-stock training universe, and the archived full-sample XGBoost recipe with seed 42, 100 rounds, and `binary:logistic`. The expanding training windows end in 2022, 2023, and 2024. One relative model was fitted per fold. E16-A applies `score > 0.5` and E16-B ranks those same scores. The relative models estimate the probability of beating the training cross-sectional median. Both absolute-target arms reuse the same archived E3 upward probabilities. Original E3 uses the frozen prior-TAIEX cutoffs of 0.525 for negative, 0.475 for nonnegative, and 0.5 for missing.

For the ranked arms, sort by descending model score and then ascending stock ID. Assign 1 to the first `floor(N/2)` of the `N` eligible targets. This gives 25/25 on full dates and 24/25 on the eleven 49-stock dates. E14’s archived E3 and E3-Rank calls match exactly on its 713 complete dates. E16 extends that comparison to all 724 dates and all 36,189 rows through the balanced rounding rule. The four arms used identical keys, with 11,950, 12,099, and 12,140 rows per validation year. Original E3 reproduced the archived official scores to `1e-12`. Tests with constructed inputs checked median ties, tied ranking scores on an odd-sized date, and AUC tie handling. Phase 1 and 2026 outcomes were outside this comparison.

## Official scores and classification metrics

| Year | Arm | Official score | Direction accuracy | Calls of 1 | Relative accuracy | Relative AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | E3 | 0.00061150 | 52.08% | 47.60% | 51.46% | 0.5204 |
| 2023 | E3-Rank | 0.00031047 | 51.76% | 50.00% | 51.82% | 0.5204 |
| 2023 | E16-A | 0.00030170 | 51.24% | 52.28% | 51.72% | 0.5272 |
| 2023 | E16-B | 0.00025125 | 51.51% | 50.00% | 52.25% | 0.5272 |
| 2024 | E3 | 0.00039108 | 52.65% | 34.90% | 51.50% | 0.5133 |
| 2024 | E3-Rank | 0.00032177 | 50.82% | 50.00% | 50.91% | 0.5133 |
| 2024 | E16-A | 0.00062076 | 52.14% | 48.17% | 51.78% | 0.5220 |
| 2024 | E16-B | 0.00056946 | 51.43% | 50.00% | 51.57% | 0.5220 |
| 2025 | E3 | 0.00091892 | 53.88% | 33.51% | 51.07% | 0.5119 |
| 2025 | E3-Rank | 0.00046542 | 51.40% | 49.96% | 52.02% | 0.5119 |
| 2025 | E16-A | 0.00067828 | 51.98% | 46.45% | 53.06% | 0.5318 |
| 2025 | E16-B | 0.00064825 | 51.33% | 49.96% | 52.91% | 0.5318 |
| Equal-year mean | E3 | **0.00064050** | | | | |
| Equal-year mean | E3-Rank | 0.00036589 | | | | |
| Equal-year mean | E16-A | 0.00053358 | | | | |
| Equal-year mean | E16-B | 0.00048966 | | | | |

Relative accuracy compares the final binary call with the evaluation-only above-median label. Relative ROC AUC uses the raw model score and awards half credit for score ties. It is identical within each model’s two decision rules. E16 improved both pooled relative AUC and mean within-day relative AUC in all three years. Daily Spearman against the size of realized returns dropped in 2023 and rose in 2024 and 2025. The rank correlation and relative AUC measure different aspects of ordering.

## Daily scores and uncertainty

| Year | Arm | Mean daily Spearman | Positive days | Daily score p10 / median / p90 | Daily score SD |
| --- | --- | ---: | ---: | --- | ---: |
| 2023 | E3 | 0.0470 | 56.49% | −0.003931 / 0.000575 / 0.005731 | 0.003910 |
| 2023 | E3-Rank | 0.0470 | 56.49% | −0.002616 / 0.000368 / 0.003384 | 0.002594 |
| 2023 | E16-A | 0.0428 | 55.23% | −0.002949 / 0.000257 / 0.003536 | 0.002929 |
| 2023 | E16-B | 0.0428 | 50.63% | −0.003118 / 0.000103 / 0.003816 | 0.002998 |
| 2024 | E3 | 0.0348 | 53.31% | −0.005047 / 0.000414 / 0.005616 | 0.005023 |
| 2024 | E3-Rank | 0.0348 | 53.72% | −0.003078 / 0.000149 / 0.003943 | 0.002859 |
| 2024 | E16-A | 0.0468 | 57.02% | −0.003895 / 0.000719 / 0.005036 | 0.003486 |
| 2024 | E16-B | 0.0468 | 54.13% | −0.003416 / 0.000329 / 0.004696 | 0.003306 |
| 2025 | E3 | 0.0434 | 53.91% | −0.004811 / 0.000521 / 0.006988 | 0.005784 |
| 2025 | E3-Rank | 0.0434 | 56.38% | −0.003353 / 0.000623 / 0.004067 | 0.002888 |
| 2025 | E16-A | 0.0561 | 56.38% | −0.003910 / 0.000591 / 0.004921 | 0.003875 |
| 2025 | E16-B | 0.0561 | 58.02% | −0.003484 / 0.000616 / 0.004475 | 0.003285 |

| Period | E16-B minus E3-Rank | Paired 95% interval |
| --- | ---: | --- |
| 2023 | −0.00005922 | [−0.00042802, +0.00032460] |
| 2024 | +0.00024769 | [−0.00014038, +0.00066651] |
| 2025 | +0.00018283 | [−0.00017059, +0.00054588] |
| Equal-year | +0.00012377 | [−0.00008609, +0.00034880] |

Relative to original E3, the equal-year differences were **−0.00010692** for E16-A, with interval [−0.00044381, +0.00022426], and **−0.00015084** for E16-B, with interval [−0.00049396, +0.00019519]. The paired circular bootstrap uses 3,000 draws with five trading days per block within each year, seeded at 42. Each sampled trading day carries its full set of stock rows and its original row count. The three yearly estimates are weighted equally. These intervals describe the reused validation years conditional on the fitted models.

E16-A's call rate varied less across the three folds than E3’s. Both E16 arms reduced daily score variability and improved the lower tail relative to original E3, while also reducing the upper tail in every year. With ranking held fixed, E16-B had **higher** daily score variability and a weaker 10th percentile than E3-Rank in each fold. The primary paired comparison therefore gives mixed evidence on stability.

## Training label distribution and runtime

| Fold | Training dates | Rows used | Absent stock-dates | Absolute 1s | Relative 1s | Median ties | Daily rate SD before / after | Fit seconds |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |
| 2023 | 735 | 73,465 | 35 | 32,814 (44.67%) | 36,173 (49.24%) | 1,041 | 18.67 / 2.33 pp | 0.99 |
| 2024 | 974 | 97,364 | 36 | 43,467 (44.64%) | 47,900 (49.20%) | 1,529 | 17.81 / 2.29 pp | 1.30 |
| 2025 | 1,216 | 121,560 | 40 | 53,959 (44.39%) | 59,797 (49.19%) | 1,935 | 17.49 / 2.27 pp | 1.39 |

Across the expanding training panels, the positive daily share ranged from 2% to 95% for absolute direction and 32% to 50% for relative performance. The median ties explain why the labels can be far from balanced on some days. The folds overlap, so training counts are presented separately. Model preparation took 0.97–2.19 s per fold and inference took roughly 8–10 ms on this run. The runtime table reports the exact values and 100% coverage of eligible training keys.

Relabeling reduced variation in daily class balance and improved discrimination of above-median stocks. The experiment changes the label definition and class balance together, so it cannot isolate which property drove the changes in predictions. It also leaves the roles of market regime dependence and overfitting unresolved. The validation results support studying relative scores as an ordering signal while retaining E3 for the current return-weighted decision task. No competition candidate was prepared.

## Reproduce

From `final_project`, choose a fresh output folder.

```powershell
uv run python -m members.zhenci.xgboost.e16.train_e16 --output-dir members/zhenci/xgboost/e16/reproduction
```

The [runner](train_e16.py) writes [fold metrics](outputs/fold_metrics.csv), [out-of-sample probabilities and calls](outputs/validation_predictions.csv), [daily scores and Spearman](outputs/daily_metrics.csv), [paired intervals](outputs/paired_uncertainty.csv), [daily training class counts](outputs/daily_training_labels.csv), [aggregate class counts](outputs/training_label_summary.csv), [runtime and coverage](outputs/runtime.csv), and [input hashes and versions](outputs/summary.json). The archived E1/E3 inputs are required. The E14 archive, when present, is cross-checked on complete dates. The completed run used XGBoost 3.4.1.
