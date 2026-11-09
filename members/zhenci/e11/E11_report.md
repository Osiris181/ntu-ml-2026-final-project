# E11: conditional expected-return decisions

**Decision: retain E3.** The fixed E11 configuration scored **0.00063915** across the three equally weighted validation years, versus **0.00064050** for E3. E11 helped in 2023 and 2025 but hurt in 2024. It reduced missed rising stocks, but the added incorrect-up decisions largely canceled that benefit. No later-period outcomes were used to select or revise E11.

## Design and reproducibility

The [runner](train_e11.py) loads the original market and F0 files, then restricts all fitting and evaluation to the existing 2023, 2024, and 2025 expanding-window folds. The original market files also contain the later January-May 2026 holdout; those rows are never used to fit, score, or select E11. Each fold trains on the 100-stock history ending before that validation year and evaluates the same 50 target stocks as E1/E3. It reuses the archived full-sample E1 **out-of-sample** `P(up)` probabilities; the five archived E1 seeds are identical, so seed 42 represents that fixed control. E3 applies its unchanged prior-TAIEX threshold to the same probabilities. The previous-direction candidate is `F0 intraday_return_1d > 0`; it is not a verified definition of the competition's “Previous day” baseline.

E11 fits three additional XGBoost components **within each training fold**, all with the original 21 F0 features, fold-fitted shared median/standard-scale/one-hot preprocessing, and the original E1 tree settings (100 rounds, depth 6, learning rate .3, full rows and columns, seed 42):

1. `q_flat = P(return = 0 | return <= 0, F0)`, trained on non-up training rows.
2. `mu_gain = E[positive return | return > 0, F0]`, trained on up training rows.
3. `mu_loss = E[absolute negative return | return < 0, F0]`, trained on down training rows.

The decision value is `p_up * mu_gain - (1 - p_up) * (1 - q_flat) * mu_loss`; predict **1 only if positive**, otherwise **0**. Thus `P(flat) = (1 - p_up) * q_flat`, `P(down) = (1 - p_up) * (1 - q_flat)`, and flat returns have zero payoff. The [unit checks](../../../tests/test_zhenci_e11.py) cover this arithmetic and the zero-decision boundary. E11 also compares (a) the same archived direction probabilities at `0.5`, exactly E1, and (b) a constant-payoff ablation using only each fold's training average flat rate and conditional gain/loss. These ablations separate the payoff decision from changing the direction model, and conditional payoff models from constant historical payoffs.

Training-only positive and negative magnitudes are each capped at their own 99.5th percentile, approximately 8.2%-8.7% across folds; about 0.5% of each conditional training sample is capped. Unconstrained regression predictions are clipped to `[0, training cap]`. **Validation returns are never capped for scoring.** Missing F0 numbers receive medians fitted on the current training fold only; missing stock/date feature rows, invalid prices, or missing archived probabilities cause an error. The script checks that archived E1 and E3 official scores reproduce exactly on the evaluation keys. No validation-year label fits any component or payoff statistic.

From `final_project`, reproduce into a new directory (the runner refuses to overwrite existing results):

```text
uv run python -m unittest tests.test_zhenci_e11
uv run python -m members.zhenci.e11.train_e11 --output-dir results/zhenci_xgboost_e11_reproduction
```

The compact [result snapshot](outputs/) includes the [summary](outputs/summary.json), [fold metrics](outputs/fold_metrics.csv), and [component diagnostics](outputs/component_validation.csv) at full precision. The runner also saves complete row-level predictions in its output directory; the default local copy is `results/zhenci_xgboost_e11_expected_return_v1/validation_predictions.csv`. Generated `results/` files are ignored by Git, so rerun the command above to recreate those predictions. This is a validation-only study: no deployment model or Phase 1/Phase 2 submission was produced.

## Same-row results

Each entry below is **official score / accuracy / predicted-up fraction**. “Direction only” is exactly the E1 row because E11 holds `P(up)` fixed. Scores use the shared official return-weighted scorer; the mean gives each year equal weight.

| Arm | 2023 | 2024 | 2025 | Equal-year mean score |
|---|---:|---:|---:|---:|
| E1 = E11 direction only | .000634 / 51.92% / 45.67% | .000211 / 52.65% / 33.24% | .000900 / 53.88% / 32.01% | .00058145 |
| **E3 reference** | **.000611 / 52.08% / 47.60%** | **.000391 / 52.65% / 34.90%** | **.000919 / 53.88% / 33.51%** | **.00064050** |
| Previous-direction candidate | .000174 / 51.46% / 45.35% | -.000008 / 50.79% / 45.04% | -.000268 / 51.03% / 43.54% | -.00003400 |
| E11 constant payoffs | .000605 / 51.42% / 54.22% | .000242 / 51.67% / 42.52% | .000977 / 52.85% / 42.25% | .00060793 |
| **E11 conditional payoffs** | **.000668 / 50.13% / 67.63%** | **.000322 / 50.71% / 49.51%** | **.000927 / 51.58% / 51.39%** | **.00063915** |

Relative to E3, E11 changes the score by **+0.00005660** in 2023, **-0.00006917** in 2024, and **+0.00000853** in 2025; the equal-year difference is **-0.00000135**. The conditional payoff arm exceeds its own direction-only and constant-payoff ablations on the three-year mean, but neither ablation establishes an improvement over E3. The conditional arm is much more bullish than E3 and loses roughly two accuracy points in every year. Its direction and payoff models do not show a stable score gain across years.

## Daily variation and paired uncertainty

| Year | E3 negative days | E11 negative days | E3 daily median | E11 daily median | E3 / E11 daily 10th percentile |
|---|---:|---:|---:|---:|---:|
| 2023 | 104 / 239 | 100 / 239 | .000575 | .000677 | -.003931 / -.003646 |
| 2024 | 113 / 242 | 108 / 242 | .000414 | .000421 | -.005047 / -.004411 |
| 2025 | 112 / 243 | 105 / 243 | .000521 | .000552 | -.004811 / -.004965 |

E11 has fewer negative-score days in each year, yet loses to E3 on the **row-weighted official mean** in 2024. Daily medians and negative-day counts therefore do not settle the competition objective. Full [daily scores](outputs/daily_scores.csv) and [distributions](outputs/daily_distributions.csv) include every arm and daily quantiles.

Paired uncertainty resamples **five consecutive trading days** in circular blocks, keeping the same dates and all stocks together for E3 and E11. The 3,000-draw seed-42 percentile intervals for E11 minus E3 are:

| Year | Observed difference | 95% block-bootstrap interval |
|---|---:|---:|
| 2023 | +.00005660 | [-.000393, +.000471] |
| 2024 | -.00006917 | [-.000455, +.000320] |
| 2025 | +.00000853 | [-.000393, +.000422] |
| Equal-year mean | **-.00000135** | **[-.000236, +.000228]** |

These wide intervals include zero. They account for dependence among stocks on a date and some short-range dependence across dates, but cannot establish generalization under regime change. E3 was previously selected using these historical folds, and E11 was designed after E10 inspected later periods. The comparison is exploratory, not a fresh independent test.

## Did E11 fix large incorrect-down losses?

| Year | E3 incorrect-down rows / score contribution | E11 incorrect-down rows / contribution | E3 incorrect-up contribution | E11 incorrect-up contribution |
|---|---:|---:|---:|---:|
| 2023 | 2,733 / -.002815 | 1,653 / -.001897 | -.002172 | -.003062 |
| 2024 | 3,476 / -.004062 | 2,709 / -.003318 | -.002073 | -.002852 |
| 2025 | 3,405 / -.003893 | 2,459 / -.002942 | -.001861 | -.002808 |

E11 does reduce the exact missed-up category identified in E10, by **1,080, 767, and 946 rows**. But it also makes more incorrect-up calls. Against E3 it flips 2,832, 2,452, and 2,657 stock-date predictions, mostly down to up; the [paired-effect table](outputs/paired_effects.csv) shows that gains and harms almost cancel. In 2024, the added incorrect-up loss more than offsets the saved incorrect-down loss. [Full score decomposition](outputs/score_decomposition.csv) includes correct-up and correct-down contributions as well.

The component diagnostics also warn against treating E11's expected-return estimates as calibrated forecasts. For 2023, average estimated return is **+.1302%** versus realized **+.0045%** on validation rows; E11 predicts up for 67.6% of stocks. Flat probability is modeled explicitly and its mean is close to observed flat frequency (9.52% estimated vs 8.97% observed in 2023), but that alone does not calibrate expected return. The three-component product can compound errors from probability and magnitude estimates.

## Conclusion

The focused E11 test supports the E10 mechanism in a limited sense: explicitly considering payoff magnitudes moves many decisions away from E3's bearish rule and reduces missed rising stocks. It **does not consistently improve the official score**, because more incorrect-up decisions and their return sizes offset the benefit. Retain E3 as the current reference. Do not use Phase 1, the inspected January-May holdout, or October to choose a revised E11 cap, tree setting, or decision cutoff. This experiment does not resolve the shared F0 publication-timing and supplied-indicator reproducibility assumptions documented in E10.
