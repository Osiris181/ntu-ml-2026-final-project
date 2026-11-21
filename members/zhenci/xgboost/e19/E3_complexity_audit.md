# E3 XGBoost complexity audit

The three plain F0 controls saved under E18 are the fold models for this audit. Their 100-round probabilities reproduce the original E3 archive. The fourth booster is the frozen full-sample E1 refit through December 2025 that supplies E3’s deployment probabilities. Its tree structure and training coverage were measured separately. E3 applies its fixed prior-TAIEX rule to the same absolute-direction model scores.

## Recorded settings

| Setting | E3 value |
| --- | --- |
| Objective and booster | `binary:logistic`, `gbtree` with `hist`, one tree per round |
| Capacity | 100 rounds, depth cap 6, depthwise growth, `max_leaves=0` |
| Learning rate | 0.30 |
| Regularization | L2 `lambda=1`, L1 `alpha=0`, `min_child_weight=1`, minimum split gain `gamma=0` |
| Sampling and seed | `subsample=1`, `colsample_bytree=1`, seed 42, uniform row weights |

The recipe and defaults describe the training settings. Tree counts, depths, Hessian cover, and routed row counts below come from the actual saved models and their original fold inputs. For logistic boosting, leaf cover is a sum of the second derivatives used when fitting that tree. It differs from a count of stock-date rows. The row counts here come from `pred_leaf=True`. [XGBoost prediction API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict).

## Actual structure and leaf support

| Saved model | Rounds | Trees with maximum depth 6 | Total leaves | Leaves per tree min / median / max | Training rows per leaf min / p10 / median |
| --- | ---: | ---: | ---: | --- | --- |
| 2023 fold | 100 | 100 | 5,297 | 37 / 53 / 64 | 5 / 6 / 45 |
| 2024 fold | 100 | 100 | 5,427 | 41 / 55 / 64 | 5 / 6 / 51 |
| 2025 fold | 100 | 100 | 5,584 | 43 / 57 / 64 | 5 / 7 / 63 |
| Final 2020–2025 refit | 100 | 100 | 5,639 | 42 / 57 / 64 | 5 / 7 / 67 |

All four boosters contain one tree per round. In 2023 the leaf depths 2–6 have counts 3, 35, 133, 414, and 4,712. The corresponding 2024 counts are 2, 27, 124, 382, and 4,892, and the 2025 counts are 1, 19, 99, 371, and 5,094. No training leaf was unused. In all four models the smallest Hessian cover is close to 1, the configured minimum child Hessian, and median Hessian cover ranges from 9.99 to 14.92.

| Saved model | Leaves with fewer than 25 training rows | Share of training routes through those leaves | Share of validation routes through those leaves | Leaves with zero validation routes |
| --- | ---: | ---: | ---: | ---: |
| 2023 fold | 38.87% | 0.30% | 0.34% | 2,590 (48.90%) |
| 2024 fold | 36.23% | 0.22% | 0.34% | 2,159 (39.78%) |
| 2025 fold | 34.44% | 0.17% | 0.23% | 2,375 (42.53%) |
| Final 2020–2025 refit | 34.19% | 0.15% | — | — |

The denominator for route shares is rows times trees. Many leaves have little support, yet they account for a small proportion of row-tree routes. Empty validation leaves also reflect the move from 100 training stocks to 50 targets, alongside new dates. These figures establish the presence of fine splits and limited coverage. They do not measure how much those splits affect the official score. Exact node-level values and per-tree summaries are saved in the [leaf table](outputs/leaves.csv) and [tree table](outputs/trees.csv).

## What E1–E18 already tested

| Study | Complexity-related changes and results |
| --- | --- |
| E1 | Full-sample depth 6 at 100 rounds won the fixed 0.5 comparison. Sampling 80% of rows and columns with otherwise similar settings scored 0.00044715 versus 0.00058145. |
| E2 grid | Four depth-3 or depth-5 fits with smaller learning rates, 200 or 300 rounds, 80% row/column sampling, child Hessian 10, L2 5 and L1 0.1. Best fixed-cutoff score 0.00050899, depth 3 and 200 rounds. Other E2 arms studied features, outcome weights, and regression. |
| E3 Optuna | The 20-trial archive includes depth 2–6, child Hessian requirements from 1 through 20, and sampled L2 settings reaching 8.54. The unchanged full-sample trial 0 won at 0.00058145 under the fixed threshold. The separate DART arms at dropout rates 0.05 and 0.10 scored 0.00048776 and 0.00051748. E3’s TAIEX study changed decisions while keeping the fitted trees. |
| E4–E5 | E4 added lagged features and E5 ablated F0 groups. E5 documented a large in-sample AUC gap for the unchanged tree recipe and motivated E6. The feature arms did not isolate capacity. |
| E6 | Inner log-loss selection, with up to 300 rounds, used an earlier year inside each training window. The E1-capacity arm chose **1 / 1 / 1** rounds. The depth-3 arm with `eta=0.1`, child Hessian 20, L2 5, and L1 0.1 chose **12 / 1 / 1**. With E3 thresholds, their means were **0.00035691** and **0.00023159** versus E3’s **0.00064050**. |
| E7–E10 | Studied payoff decisions, previous-direction overrides, external-data feasibility, and bearish behavior. These supplied diagnoses and comparisons while leaving E3’s tree capacity unchanged. |
| E11–E14 | E11 changed the modeled outcome, E12–E13 tested an LSTM, and E14 ranked model outputs. Their available checkpoints address separate objectives or model families. |
| E15–E18 | Changed the feature representation, target, sample weights, or feature set. The saved E16–E18 deployment artifacts and E18 alternative fold models bundle those interventions. E19 takes only the plain F0 E3 controls from E18. |

Shallower trees and stronger regularization have already received substantial testing. E2 and Optuna varied several settings at once. E6’s shallow, heavily regularized arm also selected an almost immediate cutoff. Those studies leave one narrower comparison open: moderate structural regularization with the same E3 learning rate, fixed round budget, sample, and decision rule. [Prior search records](outputs/prior_search.csv), [E6 inner selected rounds](outputs/e6_selected_rounds.csv).
