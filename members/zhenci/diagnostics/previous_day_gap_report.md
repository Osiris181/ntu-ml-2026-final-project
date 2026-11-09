# Previous-day baseline gap: E1 and E3 diagnostic

## What the baseline is, and what remains unknown

The available [project brief](../../../final_project.pdf) does not define the leaderboard's **“Previous day”** rule or provide its row-level predictions. The competition scoring rows are also unavailable. The user-reported leaderboard scores are approximately **+0.000772 for “Previous day”** and **+0.000722 for “AR.”** Therefore this report cannot verify or decompose the actual “Previous day” baseline.

The plausible **candidate** rule predicts each stock's prior completed trading session's intraday direction: `1` when `adjusted_close[t-1] > adjusted_open[t-1]`, otherwise `0` (including ties). Its local Phase 1 score is **+0.000721667**, matching the reported AR score to six decimals, but **not** the “Previous day” score (short by about `0.00005033`). This numerical match does **not** establish the AR rule's official definition. Two definition checks also miss +0.000772: treating prior intraday ties as up scores `+0.000732667`; using prior close-to-close direction scores `-0.000584223`. These checks identify a plausible comparator; they do not select a new Phase 1 model. See [rule checks](previous_day_gap_outputs/rule_checks.csv).

## Same-row comparison

The scores below use [`shared.evaluation.evaluate_predictions`](../../../shared/evaluation.py) on identical target-stock keys within each period. E1 uses saved seed-42 CV probabilities and its `p > 0.5` rule; all five saved E1 seeds have identical probabilities. E3 applies its previously selected prior-TAIEX thresholds to those **same** probabilities: `0.525` after a negative prior TAIEX return, `0.475` otherwise, and `0.5` when missing. The Jan–May holdout uses existing saved predictions, and Phase 1 uses the original submission files. No model was trained or threshold optimized for this report.

| Period | Rows | Candidate score / accuracy | E1 score / accuracy | E3 score / accuracy | E3 − candidate |
|---|---:|---:|---:|---:|---:|
| 2023 validation | 11,950 | +0.000174 / 51.46% | +0.000634 / 51.92% | +0.000611 / 52.08% | +0.000437 |
| 2024 validation | 12,099 | −0.000008 / 50.79% | +0.000211 / 52.65% | +0.000391 / 52.65% | +0.000399 |
| 2025 validation | 12,140 | −0.000268 / 51.03% | +0.000900 / 53.88% | +0.000919 / 53.88% | +0.001187 |
| Jan–May 2026 holdout | 4,750 | −0.000710 / 50.38% | +0.001520 / 54.95% | +0.001448 / 55.43% | +0.002158 |
| June–September 2026 Phase 1 | 4,200 | **+0.000722 / 51.62%** | −0.000318 / 50.62% | −0.000062 / 51.33% | **−0.000784** |

The equal-year 2023–2025 CV means are approximately `−0.000034` for the candidate, `+0.000581` for E1, and `+0.000640` for E3. Thus the candidate lost to both XGBoost decisions in **all three validation folds and the holdout**, then won on Phase 1. The reported “Previous day” leaderboard score exceeds E3 by approximately `0.000834`, but its accuracy and pairwise contribution breakdown cannot be calculated without its predictions. The Jan–May set has already been examined by the earlier E7 study, so it is no longer untouched for future selection; this report only reuses its archived predictions.

## Disagreements and daily losses

On Phase 1, E3 disagrees with the candidate on **2,098/4,200** rows. The candidate is correct and E3 wrong on 1,055 of these; E3 is correct and the candidate wrong on 1,043. The 2,098 disagreements fully account for E3's `−0.00078361` score difference. E3 changes **1,556 candidate-up calls to down**, contributing `−3.47608` to the unnormalized signed-return difference, while its **542 down-to-up changes** contribute `+0.18493`; dividing the net by 4,200 gives the score gap. For example, stock `3661` on July 2 rose 13.84%, while E3 changed the candidate's up call to down. The ten largest harmful disagreements explain only **5.0%** of gross harmful disagreement contribution; losses also occur across many smaller errors.

The daily score distribution is in [daily_distribution.csv](previous_day_gap_outputs/daily_distribution.csv), with every session in [daily_scores.csv](previous_day_gap_outputs/daily_scores.csv). Phase 1 had **42 candidate-negative, 39 E1-negative, and 41 E3-negative days out of 84**. Candidate and E3 median daily scores were `+0.000049` and `+0.000176`, respectively: E3 beat the candidate on **45 days** and lost on **39**, yet its larger losing days drove the lower mean. The Phase 1 worst E3 sessions were July 2 (`−0.019332`, candidate `+0.002468`), July 3 (`−0.018060`, candidate `+0.006354`), and June 9 (`−0.016783`, candidate `+0.012805`). The five largest daily E3 deficits account for **35.6%** of gross negative daily score difference; the ten largest account for **58.4%**. Historical daily distributions and all disagreement totals are in [disagreements.csv](previous_day_gap_outputs/disagreements.csv) and [paired_rows.csv](previous_day_gap_outputs/paired_rows.csv).

**Observed explanation for the candidate gap:** E3 predicts up on only **23.45%** of Phase 1 rows versus **47.60%** for the candidate. On the 1,556 candidate-up/E3-down rows, stocks rose by an average **0.1117%**, so these down calls lost score in aggregate. The same type of E3 override **gained** score in each validation year and the Jan–May holdout, especially in 2025 and the holdout. The Phase 1 result is therefore a reversal of the observed historical relationship. E3 was only 12 correct predictions behind the candidate, yet lost substantially more signed-return score, showing why accuracy alone misses part of the gap. A shift in return patterns or calibration is plausible, but unverified; the negative competition score alone does not prove overfitting. These findings explain the gap to the **candidate**, not the unknown official “Previous day” implementation.

## Recommendation

**Investigate further.** Keep E3 as the CV-selected XGBoost comparator for now. The historical folds and holdout do not support replacing it with the candidate or using that candidate as an unconditional fallback. A targeted next experiment is a **walk-forward, validation-only override study**: examine when E3 changes a prior-day up call to down, and allow a candidate-direction fallback only if a rule based on information available before prediction consistently improves official score across the existing chronological folds. Freeze such a rule before any future holdout or competition evaluation. `intraday_return_1d` is already in F0, so an added binary previous-direction column alone would duplicate existing information. Do **not** tune an override from Phase 1 labels or treat the candidate's AR-score match as validation of the leaderboard rule.

## Reproduce

From the repository root run `uv run python -m members.zhenci.diagnostics.previous_day_gap --output-dir members/zhenci/diagnostics/previous_day_gap_recheck`. The [script](previous_day_gap.py) reads original/updated market data, F0, saved E1 CV and holdout predictions, the recorded E3 rule, and the two Phase 1 submission files. It verifies identical evaluation keys, strict prior stock-history dates and F0 prior-return values, E1/E3 decision rules, and every score decomposition; it calls the shared official scorer. It writes only to a new output directory and refuses to overwrite previous results. Updated labels are never loaded for training or selection.
