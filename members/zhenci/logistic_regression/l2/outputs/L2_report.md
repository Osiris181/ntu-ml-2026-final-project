# L2 — Return-weighted Logistic Regression

**L2-B qualifies for one exploratory Phase 1 evaluation.**

Return weighting changed the equal-year official score by **+0.00006963** under the fixed
threshold and **+0.00006860** under the TAIEX rule. The two arms beat their matched
L1 controls in 1 and 2 of the three years, respectively.
These reused 2023–2025 folds provide exploratory evidence.

## Controlled change and mathematical target

The archived L1 models and manifests confirm L2-regularized logistic regression with C=1,
the lbfgs solver, 1,000 maximum iterations, tolerance 1e-4, an intercept, and seed 42.
The installed scikit-learn API represents pure L2 with l1_ratio=0. L2 adds only sample weights.
The shared, unweighted F0 preprocessing matched each archived L1 fold state exactly.
The same 100-stock training windows, 50-stock evaluation keys, labels, and official scorer were used.
E1, E3 and E17-B were read from archived predictions or reconstructed with their saved rule.

The [earlier E2 XGBoost weighting implementation](../../../xgboost/e2/train_e2_weighted.py)
used the same decimal adjusted intraday return magnitude divided by the training-fold mean.
L2 reuses that formula, keeps every flat row with zero weight, and applies no clipping or
class balancing.

The conditional score of a binary action `a` is `(2a − 1) E[r | X]`. For any `X` with positive
expected return magnitude, the unrestricted weighted-BCE optimum is
`p_w(X) = E[|r|y | X] / E[|r| | X]`. Because `|r|(2y − 1) = r`, it follows that
`2p_w(X) − 1 = E[r | X] / E[|r| | X]`. A strict cutoff of 0.5 therefore implements the sign
decision at that population optimum. The fitted regularized linear model is a restricted
approximation. Its output describes the return-weighted target. L2-B's thresholds of
0.525, 0.475, and 0.5 for negative, nonnegative, and missing eligible TAIEX returns are an
empirical comparison.

## Historical evaluation

| arm | 2023 | 2024 | 2025 | official_score | worst_four_month_score | worst_block |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.0006340 | 0.0002108 | 0.0008996 | 0.0005815 | 0.0001246 | 2024-B3 |
| E17-B | 0.0003494 | 0.0006103 | 0.0004198 | 0.0004598 | 0.0001480 | 2025-B3 |
| E3 | 0.0006115 | 0.0003911 | 0.0009189 | 0.0006405 | 0.0002867 | 2024-B3 |
| L1-A | 0.0000547 | 0.0003951 | 0.0009787 | 0.0004762 | -0.0002793 | 2023-B3 |
| L1-B | -0.0000165 | 0.0006993 | 0.0007591 | 0.0004806 | -0.0001612 | 2023-B3 |
| L2-A | -0.0000259 | 0.0007991 | 0.0008642 | 0.0005458 | -0.0004985 | 2023-B3 |
| L2-B | 0.0001067 | 0.0009374 | 0.0006036 | 0.0005492 | -0.0001176 | 2023-B3 |

| arm | accuracy | direction_auc | mean_daily_spearman | predicted_up_rate | actual_up_rate | market_direction | stock_selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.5281501 | 0.5203427 | 0.0417554 | 0.3697531 | 0.4463689 | 0.0003647 | 0.0002167 |
| E17-B | 0.5115815 | 0.5215945 | 0.0397615 | 0.5167142 | 0.4463689 | 0.0001101 | 0.0003498 |
| E3 | 0.5287080 | 0.5203427 | 0.0417554 | 0.3866761 | 0.4463689 | 0.0004121 | 0.0002284 |
| L1-A | 0.5442129 | 0.5234573 | 0.0442931 | 0.1802182 | 0.4463689 | 0.0002983 | 0.0001779 |
| L1-B | 0.5397739 | 0.5234573 | 0.0442931 | 0.2417845 | 0.4463689 | 0.0003065 | 0.0001741 |
| L2-A | 0.5013641 | 0.5182112 | 0.0610190 | 0.5970666 | 0.4463689 | 0.0001470 | 0.0003988 |
| L2-B | 0.5000873 | 0.5182112 | 0.0610190 | 0.6069371 | 0.4463689 | 0.0001930 | 0.0003562 |

Market direction uses the daily mean signed position times the daily mean realized return.
Stock selection is the residual. Period aggregates weight trading days by their observation counts.
See [daily](daily_scores.csv), [monthly](monthly_scores.csv) and [four-month scores](four_month_scores.csv).
The [validation prediction file](validation_predictions.csv) distinguishes weighted scores from
ordinary direction probabilities.

| arm | control | period | delta | low_95 | high_95 |
| --- | --- | --- | --- | --- | --- |
| L2-A | L1-A | 2023 | -0.0000806 | -0.0006539 | 0.0004822 |
| L2-A | L1-A | 2024 | 0.0004040 | -0.0003402 | 0.0011999 |
| L2-A | L1-A | 2025 | -0.0001144 | -0.0008113 | 0.0005556 |
| L2-A | L1-A | equal_year | 0.0000696 | -0.0003204 | 0.0004593 |
| L2-B | L1-B | 2023 | 0.0001232 | -0.0003188 | 0.0006060 |
| L2-B | L1-B | 2024 | 0.0002381 | -0.0003657 | 0.0008869 |
| L2-B | L1-B | 2025 | -0.0001555 | -0.0006916 | 0.0003908 |
| L2-B | L1-B | equal_year | 0.0000686 | -0.0002429 | 0.0003886 |
| L2-B | L2-A | 2023 | 0.0001326 | -0.0001040 | 0.0003803 |
| L2-B | L2-A | 2024 | 0.0001383 | -0.0002305 | 0.0004967 |
| L2-B | L2-A | 2025 | -0.0002606 | -0.0005526 | 0.0000436 |
| L2-B | L2-A | equal_year | 0.0000034 | -0.0001673 | 0.0001771 |

Intervals use 3,000 paired circular bootstrap samples of five-trading-day blocks within years.
The sampling unit keeps every stock from a trading day together. See the
[full paired comparisons](paired_uncertainty.csv) and the [candidate screen](phase1_screen.csv).

## Weights and loss

| fold | rows | normalizer | zero_fraction | p99 | max | ess_fraction | top_1pct_weight_share | top_10pct_weight_share | weighted_up_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 73465 | 0.0139683 | 0.0734772 | 5.2554228 | 13.9579634 | 0.4584961 | 0.0630409 | 0.3570998 | 0.4768198 |
| 2024 | 97364 | 0.0135150 | 0.0751818 | 5.3279243 | 14.4260676 | 0.4545940 | 0.0642262 | 0.3592844 | 0.4789144 |
| 2025 | 121560 | 0.0136143 | 0.0742843 | 5.2465837 | 14.3208637 | 0.4609195 | 0.0632271 | 0.3553023 | 0.4772049 |

The largest-return 1% received 6.3% to
6.4% of training weight. The top 10% received
35.5% to 35.9%.
The effective sample size was 45.5% to
46.1% of the row count. This measures concentration of the loss weights.
Whether the largest returns caused the performance differences remains open.
The [weight table](sample_weights.csv) includes means, standard deviations, quantiles,
zero counts, label totals, and absolute ESS.

| fold | split | L1 | L2 | delta |
| --- | --- | --- | --- | --- |
| 2023 | train | 0.6853445 | 0.6788154 | -0.0065292 |
| 2023 | validation | 0.7008781 | 0.7015514 | 0.0006734 |
| 2024 | train | 0.6887389 | 0.6829242 | -0.0058148 |
| 2024 | validation | 0.6973209 | 0.6914378 | -0.0058831 |
| 2025 | train | 0.6892395 | 0.6831585 | -0.0060810 |
| 2025 | validation | 0.6871474 | 0.6858086 | -0.0013388 |

Both models use the same absolute-return diagnostic weights, scaled by each training fold's
mean absolute return on both splits. Validation weights are applied only when calculating loss.
See [loss and score distributions](loss_and_score_distributions.csv) for all validation arms,
score quantiles, and ordinary BCE and Brier comparisons. The official score governs the recommendation.

| fold | fit_seconds | inference_seconds | optimizer_iterations | converged |
| --- | --- | --- | --- | --- |
| 2023 | 1.1545630 | 0.0756505 | 75 | True |
| 2024 | 1.6162246 | 0.0688187 | 79 | True |
| 2025 | 1.8723508 | 0.0679530 | 69 | True |

## Coefficients and prediction contributions

| fold | L1 | L2 | delta |
| --- | --- | --- | --- |
| 2023 | -0.1803953 | -0.0284957 | 0.1518996 |
| 2024 | -0.1798134 | -0.0240298 | 0.1557836 |
| 2025 | -0.1879381 | -0.0313186 | 0.1566196 |

| feature | mean_L1 | mean_L2 | mean_absolute_change |
| --- | --- | --- | --- |
| nasdaq_return | 0.0826264 | 0.3225049 | 0.2398785 |
| sp500_return | -0.0226186 | -0.2238682 | 0.2012496 |
| relative_amount_20d | 0.4024043 | 0.2475139 | 0.1548905 |
| relative_volume_20d | -0.3790910 | -0.2362870 | 0.1428039 |
| taiex_return | 0.0088380 | 0.0562473 | 0.0474093 |
| foreign_ownership | -0.0173615 | -0.0588728 | 0.0415113 |

| model | family | mean_contribution | mean_abs_contribution |
| --- | --- | --- | --- |
| L1 | day_of_week | -0.0358326 | 0.0538441 |
| L1 | numeric | -0.0048448 | 0.0850603 |
| L1 | stock_id | 0.0465231 | 0.1414676 |
| L2 | day_of_week | -0.0058099 | 0.0541620 |
| L2 | numeric | 0.0339386 | 0.1475733 |
| L2 | stock_id | 0.0709264 | 0.1893815 |

The contribution audit multiplies every encoded feature by its fitted coefficient and verifies the
intercept plus the sum against the model decision function and score for every validation row.
Family rows are equal-year means of the signed and absolute validation log-odds contributions.
The [coefficient table](coefficients.csv) includes signs, magnitudes and matched changes.
See [cross-fold stability](coefficient_stability.csv) and [individual feature contributions](feature_contributions.csv).
The estimates describe conditional associations.

## Threshold sensitivity and Phase 1

| period | model | changed | down_to_up | up_to_down | fraction_in_0475_to_0525 | official_score_delta | market_direction_delta | stock_selection_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | L1 | 1730 | 1212 | 518 | 0.2792469 | -0.0000712 | 0.0000191 | -0.0000903 |
| 2023 | L2 | 1380 | 758 | 622 | 0.2400000 | 0.0001326 | 0.0000987 | 0.0000339 |
| 2024 | L1 | 2062 | 1357 | 705 | 0.3299446 | 0.0003042 | 0.0002504 | 0.0000538 |
| 2024 | L2 | 1455 | 775 | 680 | 0.2589470 | 0.0001383 | 0.0001019 | 0.0000364 |
| 2025 | L1 | 1675 | 1279 | 396 | 0.2747941 | -0.0002196 | -0.0002446 | 0.0000250 |
| 2025 | L2 | 1572 | 849 | 723 | 0.2783361 | -0.0002606 | -0.0000624 | -0.0001982 |

The [full sensitivity table](threshold_sensitivity.csv) includes yearly and regime-specific counts,
quantiles and up rates. Summed changes from switched predictions recover the official-score
differences, including the separate market-direction and stock-selection changes.

In the saved L1 Phase 1 predictions, **28.5%** of scores
fell between 0.475 and 0.525. The TAIEX rule changed **596** calls
(14.2%), including 398 down-to-up and
198 up-to-down. The user-reported aggregate score increased by
**+0.000423**. Concentration
near the threshold gave the rule room to change calls. Identifying which switched predictions
earned the gain would require row-level scoring evidence. This diagnostic used
[saved probabilities and calls](phase1_l1_threshold_sensitivity.csv) and
[reported aggregate scores](phase1_reported_scores.csv).

The screen was set before examining L2 results. Positive equal-year difference against the matched L1 rule, at least two winning years, and no deterioration in the worst four-month score. Prefer the higher official mean among eligible arms, with a tie going to L2-A. Report paired uncertainty to qualify the strength of exploratory evidence.
January–May 2026, Phase 1 and October outcomes remained outside model fit and selection.

## Reproduction

From the repository root, pass a fresh output directory.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.logistic_regression.l2.train_l2 --output-dir members/zhenci/logistic_regression/l2/reproduction
```

The [run manifest](summary.json) records the frozen configuration and input and model digests.
