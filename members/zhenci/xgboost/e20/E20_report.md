# E20 structurally regularized XGBoost

**Keep E3 as the reference.** E20 reduced the training–validation gaps and improved validation BCE and Brier in all three years. Validation AUC and the official return-weighted score both fell in every year, and the monthly and four-month results were less stable. An exploratory Phase 1 evaluation of E20 is unwarranted on this evidence.

## Controlled experiment

E20 changes only `max_depth` from 6 to 4, `min_child_weight` from 1 to 10, and `reg_lambda` from 1 to 5, using XGBoost’s `lambda` alias in the existing recipe. Both arms use 100 rounds, learning rate 0.30, L1 regularization 0, gamma 0, full row and column sampling, uniform weights, and seed 42. The objective is binary logistic on the original absolute-direction labels. Both arms use the original 21 F0 features and the fixed prior-TAIEX cutoffs of 0.525, 0.475, or 0.5 for missing data. [E19 audit](../e19/E3_complexity_audit.md).

The runner used the saved E3 fold models and verified each frozen preprocessor against a fresh fit on the same training rows. Each fold then trained one E20 classifier on those same inputs. Expanding training windows contain 73,465, 97,364, and 121,560 observations from the 100-stock universe. The common validation rows number 11,950, 12,099, and 12,140. All E3 probabilities at 100 rounds matched their archive within `3.0e-8`, and every binary decision and official score reproduced. The [saved predictions](outputs/validation_predictions.csv) preserve the same stock-date keys at every checkpoint. The comparison uses historical 2023–2025 results only.

## Primary comparison at 100 rounds

| Year | E3 official score | E20 official score | E20 − E3 | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| 2023 | 0.00061150 | 0.00000386 | −0.00060764 | [−0.00112145, −0.00011511] |
| 2024 | 0.00039108 | 0.00036721 | −0.00002388 | [−0.00044683, +0.00039963] |
| 2025 | 0.00091892 | 0.00068966 | −0.00022926 | [−0.00069321, +0.00025704] |
| Equal-year mean | **0.00064050** | **0.00035357** | **−0.00028692** | **[−0.00056051, −0.00001771]** |

The paired circular bootstrap uses 3,000 draws with blocks of five trading days and seed 42. It keeps all stocks on each sampled day together and weights by that day’s observation count within each year. The interval is conditional on these fitted models and the repeatedly inspected folds. The largest loss is in 2023. [Yearly and equal-year uncertainty at all checkpoints](outputs/paired_uncertainty.csv).

| Year | Model | BCE train / validation | AUC train / validation | Accuracy train / validation | Validation Brier | Validation up calls | Daily Spearman |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 2023 | E3 | 0.53263 / 0.72192 | 0.8408 / 0.5245 | 75.22% / 52.08% | 0.26235 | 47.60% | 0.0470 |
| 2023 | E20 | 0.61361 / 0.71143 | 0.7332 / 0.5061 | 66.59% / 50.68% | 0.25852 | 47.15% | 0.0420 |
| 2024 | E3 | 0.56020 / 0.72181 | 0.8151 / 0.5107 | 73.02% / 52.65% | 0.26219 | 34.90% | 0.0348 |
| 2024 | E20 | 0.62765 / 0.70781 | 0.7136 / 0.4946 | 65.25% / 52.20% | 0.25671 | 30.10% | 0.0371 |
| 2025 | E3 | 0.57312 / 0.70752 | 0.7992 / 0.5259 | 71.90% / 53.88% | 0.25568 | 33.51% | 0.0434 |
| 2025 | E20 | 0.63535 / 0.69346 | 0.6981 / 0.5201 | 64.10% / 54.49% | 0.24977 | 24.88% | 0.0485 |

The mean BCE gap fell from **0.16176 to 0.07869** and the mean AUC gap from **0.2981 to 0.2080**. Mean validation BCE improved from 0.71708 to 0.70423 and Brier from 0.26008 to 0.25500. The probability distribution narrowed in every fold. For example, the middle 80% in 2024 went from 0.285–0.622 for E3 to 0.332–0.566 for E20. The improvement in probability error came with a decline in mean validation AUC from **0.5203 to 0.5069**, driven by a loss in every year. Within-day Spearman improved modestly in 2024 and 2025 and fell in 2023. Validation accuracy improved only in 2025, where up calls dropped by 8.63 percentage points. These metrics measure distinct properties. The smaller gap reflects weaker training fit and a real improvement in validation probability loss. The lower validation AUC is consistent with giving up usable direction discrimination.

## Boosting checkpoints

Values are equal-year means. Both arms use prefixes of their original 100-round fold fits. The primary comparison remains fixed at 100 rounds.

| Rounds | E3 train / validation BCE | E20 train / validation BCE | E3 / E20 validation AUC | E3 / E20 official score |
| ---: | --- | --- | --- | --- |
| 10 | 0.65426 / 0.69197 | 0.66996 / 0.69014 | 0.5175 / 0.5060 | 0.000597 / 0.000509 |
| 25 | 0.62893 / 0.69762 | 0.65866 / 0.69287 | 0.5178 / 0.5104 | 0.000704 / 0.000524 |
| 50 | 0.60010 / 0.70438 | 0.64593 / 0.69830 | 0.5221 / 0.5052 | 0.000761 / 0.000444 |
| 75 | 0.57639 / 0.71261 | 0.63487 / 0.70104 | 0.5189 / 0.5050 | 0.000584 / 0.000356 |
| 100 | 0.55532 / 0.71708 | 0.62554 / 0.70423 | 0.5203 / 0.5069 | 0.000640 / 0.000354 |

As rounds increase, E20 still improves in-sample fit while validation BCE deteriorates. Its validation BCE degrades less over this range than E3’s, and its average official score is lower at all five matched checkpoints. Per-fold training and validation accuracy, Brier, up-call rates, probability quantiles and Spearman for each checkpoint are in the [complete metrics](outputs/checkpoint_metrics.csv).

![E3 and E20 curves at the five fixed diagnostic checkpoints](outputs/boosting_curves.png)

## Temporal stability at 100 rounds

| Measure | E3 | E20 |
| --- | ---: | ---: |
| Negative months out of 36 | 10 | 14 |
| Monthly score SD | 0.000948 | 0.001015 |
| Worst month | −0.000742, Aug 2025 | −0.001235, Feb 2024 |
| Negative four-month blocks out of nine | 0 | 3 |
| Four-month score SD | 0.000431 | 0.000582 |
| Worst four-month block | +0.000287, Sep–Dec 2024 | −0.000186, May–Aug 2025 |

E20 beat E3 in 12 of 36 months and two of nine fixed blocks. Daily score standard deviation rose in all three years. The blocks are January–April, May–August, and September–December within each validation year. [Daily scores](outputs/daily_scores.csv), [monthly scores](outputs/monthly_scores.csv), [four-month scores](outputs/four_month_scores.csv).

## Actual tree size and leaf support

Every E3 tree reaches depth 6 and every E20 tree reaches depth 4. Counts below cover all 100 trees per model. Training observations per leaf are actual row counts from routing the original training matrix through the trees. Hessian cover is a separate statistic.

| Year | Total leaves E3 / E20 | Median leaves per tree E3 / E20 | Min training rows per leaf E3 / E20 | Median training rows per leaf E3 / E20 | Leaves with fewer than 25 training rows E3 / E20 |
| --- | --- | --- | --- | --- | --- |
| 2023 | 5,297 / 1,457 | 53 / 15 | 5 / 41 | 45 / 317 | 2,059 / 0 |
| 2024 | 5,427 / 1,506 | 55 / 15 | 5 / 41 | 51 / 366 | 1,966 / 0 |
| 2025 | 5,584 / 1,529 | 57 / 16 | 5 / 41 | 63 / 451 | 1,923 / 0 |

The intended reduction in capacity occurred. E20’s minimum leaf Hessian cover is just above 10 in every fold, compared with just above 1 for E3. This controlled comparison tests the three structural changes together. It cannot attribute the effect to any one setting. [Tree structure](outputs/trees.csv), [leaf observations and Hessian cover](outputs/leaves.csv).

## Decision and reproduction

The lower direction probability loss supports the idea that E3’s capacity contributed to overfitting that objective. The weaker discrimination, official score, and temporal stability argue against adopting this particular regularization package. Further depth or regularization optimization on the same outer folds is unlikely to resolve the trade-off. A fresh comparison of a prespecified model on unused data would provide more information than additional tuning on 2023–2025. Phase 1 submission preparation is not recommended for E20.

Run from `final_project` with a fresh output directory. The completed experiment used XGBoost 3.4.1, scikit-learn 1.9.1, and Matplotlib 3.11.2.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.xgboost.e20.train_e20 --output-dir members/zhenci/xgboost/e20/reproduction
uv run --with matplotlib==3.11.2 python -m members.zhenci.xgboost.e20.plot_e20 --output-dir members/zhenci/xgboost/e20/reproduction
```

The [three fold artifacts](outputs/models) contain the trained E20 boosters and their shared-recipe fitted preprocessing. [Run provenance](outputs/summary.json) records the exact parameters, source hashes, and library versions. [Runtime](outputs/runtime.csv) records the actual fitting time per fold.
