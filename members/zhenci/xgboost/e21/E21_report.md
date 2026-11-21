# E21 temporal adaptation through a later training cutoff

**Retain the static E3 model.** Extending training through May helped the 2024 June–September period and hurt the 2023 and 2025 periods. The equal-year mean fell from **0.00064179 to 0.00044955**. The update improved within-day stock ranking in all three years and improved the worst four-month score, while losing market-direction contribution in all three years. These reused historical periods provide exploratory evidence and do not support team-wide adoption.

## Data and controlled design

Only the training cutoff differs. The static model uses data through December of the previous year. The update adds the following January–May observations, fits a new shared preprocessor on the expanded panel, and trains one new model from scratch. Both arms start in 2020, use the original absolute-direction label, keep all 100 training stocks, and apply the fixed E3 XGBoost recipe. This includes the unchanged 21 F0 fields, 100 rounds, seed 42, full sampling, uniform weight, binary logistic objective, and TAIEX decision cutoffs. The comparison keeps the complete set of eligible Top50 June–September observations, including incomplete cross-sections. Earlier E3 recency experiments changed the weight of observations within a cutoff. E21 keeps weights uniform and changes the available history.

| Evaluation period | Static cutoff / rows | Updated cutoff / rows | Added rows / dates | Evaluation rows / dates | Updated fit time |
| --- | --- | --- | --- | --- | ---: |
| Jun–Sep 2023 | 2022-12-31 / 73,465 | 2023-05-31 / 82,765 | 9,300 / 93 | 4,150 / 83 | 2.03 s |
| Jun–Sep 2024 | 2023-12-31 / 97,364 | 2024-05-31 / 107,163 | 9,799 / 98 | 4,099 / 82 | 1.84 s |
| Jun–Sep 2025 | 2024-12-31 / 121,560 | 2025-05-31 / 131,060 | 9,500 / 95 | 4,292 / 86 | 1.97 s |

The archived static probabilities reproduce the original full-year E3 scores before subsetting. The static training panel is an exact stock-date and label prefix of the updated panel. Added rows fall after the static cutoff and both training windows end before evaluation starts. Both arms were verified to use identical evaluation keys. All three saved updated boosters reproduce their probabilities when reloaded. [Training counts and timings](outputs/training_counts_runtime.csv), [paired predictions](outputs/validation_predictions.csv), [fold artifacts](outputs/models).

The [eligibility audit](eligibility_audit.md) verified 9,500 January–May 2026 original rows across the full Top100, with valid labels and complete F0 key coverage. A rebuild after excluding later observations matched the archived F0 slice. At the user’s direction, the team holdout remains reserved, so those labels were used only for the data checks.

## Official scores and paired uncertainty

| Period | Static E3 | Updated E3 | Updated − static | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Jun–Sep 2023 | 0.00115755 | 0.00056820 | −0.00058935 | [−0.00188234, +0.00054029] |
| Jun–Sep 2024 | 0.00023324 | 0.00040828 | +0.00017504 | [−0.00051113, +0.00085686] |
| Jun–Sep 2025 | 0.00053458 | 0.00037216 | −0.00016242 | [−0.00087375, +0.00056110] |
| Equal-year mean | **0.00064179** | **0.00044955** | **−0.00019224** | **[−0.00073481, +0.00030096]** |
| Worst period | 0.00023324 | 0.00037216 | | |

The paired circular bootstrap uses 3,000 draws, five trading days per block, and seed 42. All observations on a sampled day move together. Yearly scores use sampled observation counts and the three yearly estimates are weighted equally. Every interval includes zero. The worst-period result improved, but the update provided no consistent mean gain. [Paired uncertainty](outputs/paired_uncertainty.csv).

## Direction probabilities and within-day ranking

| Year | Model | BCE | Brier | ROC AUC | Direction accuracy | Predicted up | Actual up | Mean daily Spearman | Probability p10 / median / p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | Static | 0.71497 | 0.25856 | 0.5560 | 53.73% | 50.65% | 44.53% | 0.0485 | 0.311 / 0.499 / 0.714 |
| 2023 | Updated | 0.72707 | 0.26400 | 0.5379 | 52.02% | 49.61% | 44.53% | 0.0629 | 0.295 / 0.493 / 0.707 |
| 2024 | Static | 0.72639 | 0.26410 | 0.5145 | 51.45% | 36.35% | 46.55% | 0.0264 | 0.278 / 0.440 / 0.625 |
| 2024 | Updated | 0.72105 | 0.26236 | 0.5065 | 52.55% | 31.74% | 46.55% | 0.0286 | 0.290 / 0.429 / 0.597 |
| 2025 | Static | 0.70030 | 0.25294 | 0.5333 | 53.96% | 34.25% | 44.83% | 0.0495 | 0.319 / 0.451 / 0.596 |
| 2025 | Updated | 0.70371 | 0.25442 | 0.5183 | 53.49% | 27.77% | 44.83% | 0.0552 | 0.316 / 0.432 / 0.568 |

The updated arm makes fewer up calls in every period. BCE, Brier and accuracy improve only in 2024, the year when the official score also improves. Pooled direction AUC falls every year even as mean daily Spearman rises. AUC measures discrimination of the binary direction label across the period. Daily Spearman measures cross-sectional ordering by realized return magnitude. Neither is the signed-return score of the thresholded calls. [Complete period metrics and probability distributions](outputs/period_metrics.csv).

## Stability and score decomposition

| Period | Daily score median static / updated | Daily score SD static / updated | Negative days static / updated | Market-direction contribution static / updated | Stock-selection contribution static / updated |
| --- | --- | --- | --- | --- | --- |
| 2023 | 0.001279 / 0.000140 | 0.004512 / 0.004472 | 39.76% / 48.19% | 0.001008 / 0.000459 | 0.000149 / 0.000109 |
| 2024 | 0.000034 / −0.000104 | 0.005506 / 0.006009 | 50.00% / 51.22% | 0.000195 / 0.000128 | 0.000039 / 0.000280 |
| 2025 | −0.000037 / 0.000021 | 0.004722 / 0.004418 | 50.00% / 50.00% | 0.000362 / −0.000022 | 0.000173 / 0.000394 |

The market-direction term is each day’s average signed prediction multiplied by the mean realized return. The stock-selection term is the corresponding within-day covariance. The terms sum to the official score after weighting dates by observation count. In 2024, better stock selection more than offsets weaker market-direction contribution. In 2025, the selection gain is smaller than the directional loss, and in 2023 both components decline. This decomposition accounts for the score differences without attributing them to a specific market regime or causal mechanism.

| Year | June static / updated | July static / updated | August static / updated | September static / updated |
| --- | --- | --- | --- | --- |
| 2023 | 0.000905 / 0.000040 | 0.000806 / 0.000618 | 0.002033 / 0.000752 | 0.000817 / 0.000842 |
| 2024 | 0.000209 / 0.000509 | −0.000301 / −0.000329 | 0.001065 / 0.001766 | −0.000097 / −0.000404 |
| 2025 | 0.000920 / 0.000269 | 0.000195 / −0.000396 | −0.000742 / 0.000278 | 0.001790 / 0.001410 |

The update wins four of twelve months. Both arms have three negative months. The worst month improves from −0.000742 to −0.000404, and monthly score standard deviation falls from 0.000819 to 0.000688. Daily variability improves in two periods and worsens in 2024. The updated cutoff reduces some downside without improving the average overall. [Monthly scores, distributions, and contributions](outputs/monthly_scores.csv), [daily scores](outputs/daily_scores.csv).

## Competition decision and reuse by other members

**Keep E3 as the static reference and close this XGBoost tuning track.** The later cutoff is technically feasible but the holdout remains reserved, so E21 has no Phase 1 or Phase 2 deployment candidate. Phase 1 scores are an already-examined development benchmark and played no role in this comparison. There is insufficient evidence to adopt the cutoff change team-wide.

If a future team decision authorizes a separate cutoff comparison, freeze both artifacts before the first eligible Phase 2 session. Use that day’s actual template and the same eligible input snapshot for both. Archive both local prediction files, artifact and input hashes, and their generation timestamps before 09:00 Asia/Taipei. Score those frozen files against independently available realized adjusted returns once available for the exact same stock-date keys. Keep any missing session explicit and coordinate the single official submission choice with the team. Existing historical files cannot be retroactively called prospective predictions.

The protocol is model-independent. Select eligible observations and pin both cutoffs first, then give each member the same paired training and evaluation keys. Each member fits only their model’s own preprocessing on the rows eligible for that cutoff, trains their existing architecture from scratch, and generates predictions with their previously fixed decision rule. Compare the paired predictions with the shared official scorer and whole-day uncertainty. The training and deployment choices can be documented independently of the model family. Sharing implementation can wait for team agreement on the eligibility policy and evidence from additional models or genuinely unused periods.

## Reproduce

From `final_project`, using fresh output directories. The run used XGBoost 3.4.1 and scikit-learn 1.9.1.

```powershell
uv run python -m members.zhenci.xgboost.e21.audit_e21 --output-dir members/zhenci/xgboost/e21/audit_reproduction
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e21.train_e21 --output-dir members/zhenci/xgboost/e21/reproduction
```

[Run settings and input hashes](outputs/summary.json).
