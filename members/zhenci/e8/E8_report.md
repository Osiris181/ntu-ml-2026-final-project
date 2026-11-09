# E8: confidence-limited previous-direction override

## Decision

**Retain E3.** A narrow override helped slightly in 2024 but hurt in 2025. The strictly forward-selected E8 mean across 2023–2025 was `0.00063470`, below E3's `0.00064050`. After freezing the final candidate distance at `0.01`, one later October check also favored E3. These reused validation folds are exploratory evidence, and the short October window is one additional check, not proof of future performance.

The previous-direction signal here means `F0 intraday_return_1d > 0`: the stock's previous completed session rose from adjusted open to adjusted close. It is **not** the verified definition of the leaderboard's “Previous day” baseline; see the earlier [baseline diagnostic](../diagnostics/previous_day_gap_report.md) and [E1/E3 diagnostic](../diagnostics/e1_e3_gap_report.md).

## Design and chronology

E8 changes only the decision after the saved XGBoost probability and fixed E3 TAIEX threshold are available. For stock-date row `i`, define `d_i = E3_threshold_i − p_up_i`. E8 changes E3's down prediction to up only when the previous stock session was up, `d_i >= 0`, and `d_i <= distance`. Distance `0` is an explicit no-override E3 control. The **only** candidate distances are `0`, `0.01`, `0.025`, and `0.05`. E3's prior-TAIEX thresholds remain `0.525` (negative), `0.475` (nonnegative), and `0.5` (missing). The original XGBoost model, fitted preprocessing, and 21 F0 features are unchanged.

Saved out-of-sample probabilities exist for the 2023, 2024, and 2025 expanding-window folds. Five E1 seeds produce identical saved probabilities, so seed 42 represents each fold without picking a winning seed. For 2023, **no earlier saved out-of-sample fold exists**; E8 therefore uses distance `0`, rather than selecting on 2023 itself or generating an unplanned training run. For 2024, the distance is chosen by official score on 2023 only; for 2025, by the equal-year mean on 2023 and 2024 only. Exact ties prefer the smaller distance. These choices are `0`, `0.01`, and `0.01`, respectively. Thus no E8 distance is selected using its evaluation fold. The E3 TAIEX rule was previously selected on these same 2023–2025 folds, so this is a **nested override comparison conditional on a fixed, previously tuned E3**, not a wholly independent assessment of E3.

The 2023–2025 equal-year grid selects `0.01` as the **frozen later-period E8 candidate** (its same-year exploratory mean gain over E3 is about `+0.00000909`). That grid mean is not the walk-forward E8 estimate. Phase 1 and the previously inspected Jan–May holdout were not used to select or evaluate E8 here.

## Historical disagreement evidence

Only rows with **previous-direction up and E3 down** can be overridden. In the table, each group contains out-of-sample target-stock predictions. `Mean return` is the realized adjusted intraday return. `E3 sum` and `up sum` are the sums of signed-return contributions if the group remains down or is overridden up; `Δ fold score` divides their difference by **all target rows in that fold**, as in the official metric. The full TAIEX-stratified data, including medians and rising-stock fractions, are in [historical_disagreement_bins.csv](outputs/historical_disagreement_bins.csv).

| Year | Distance below E3 threshold | Rows | Mean return | E3 sum | Up sum | Δ fold score if overridden |
|---|---|---:|---:|---:|---:|---:|
| 2023 | 0–0.01 | 140 | +0.001907 | −0.266954 | +0.266954 | +0.00004468 |
| 2023 | 0.01–0.025 | 237 | −0.000652 | +0.154591 | −0.154591 | −0.00002587 |
| 2023 | 0.025–0.05 | 391 | −0.000573 | +0.224030 | −0.224030 | −0.00003749 |
| 2023 | >0.05 | 2,116 | −0.000228 | +0.483443 | −0.483443 | −0.00008091 |
| 2024 | 0–0.01 | 145 | +0.000327 | −0.047351 | +0.047351 | +0.00000783 |
| 2024 | 0.01–0.025 | 224 | −0.000784 | +0.175658 | −0.175658 | −0.00002904 |
| 2024 | 0.025–0.05 | 414 | +0.000346 | −0.143412 | +0.143412 | +0.00002371 |
| 2024 | >0.05 | 2,774 | −0.000520 | +1.443381 | −1.443381 | −0.00023860 |
| 2025 | 0–0.01 | 159 | −0.000963 | +0.153093 | −0.153093 | −0.00002522 |
| 2025 | 0.01–0.025 | 287 | −0.001769 | +0.507622 | −0.507622 | −0.00008363 |
| 2025 | 0.025–0.05 | 463 | −0.001066 | +0.493622 | −0.493622 | −0.00008132 |
| 2025 | >0.05 | 2,762 | −0.001113 | +3.074356 | −3.074356 | −0.00050648 |

The narrowest group helps in 2023 and 2024, then hurts in 2025. It does not show a stable confidence relationship. TAIEX conditioning also reverses: within the 0–0.01 group after **nonnegative** prior TAIEX, override contribution differences were `+0.5191` (99 rows) in 2023, `+0.3552` (105) in 2024, and `−0.5736` (100) in 2025. After **negative** prior TAIEX they were `+0.0453` (40), `−0.3530` (37), and `+0.2407` (57). The missing-TAIEX groups had only 1–3 rows per year. These inconsistent signs do not justify an added regime rule.

## Same-row evaluation

Each cell below is **official score / accuracy / fraction predicted up**. All four approaches use identical target-stock dates and the shared official scorer. E8 uses the chronologically selected distance for that year; October uses the frozen `0.01` distance. [Fold metrics](outputs/fold_metrics.csv), [candidate grid](outputs/candidate_grid.csv), and [reserved metrics](outputs/reserved_metrics.csv) retain full precision.

| Period | Rows | E1 | E3 | Previous-direction candidate | E8 | E8 − E3 | E8 overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 11,950 | +0.000634 / 51.92% / 45.67% | +0.000611 / 52.08% / 47.60% | +0.000174 / 51.46% / 45.35% | +0.000611 / 52.08% / 47.60% | 0 | 0 |
| 2024 | 12,099 | +0.000211 / 52.65% / 33.24% | +0.000391 / 52.65% / 34.90% | −0.000008 / 50.79% / 45.04% | +0.000399 / 52.54% / 36.09% | +0.00000783 | 145 |
| 2025 | 12,140 | +0.000900 / 53.88% / 32.01% | +0.000919 / 53.88% / 33.51% | −0.000268 / 51.03% / 43.54% | +0.000894 / 53.67% / 34.82% | −0.00002522 | 159 |
| Equal-year CV mean | — | +0.000581 | +0.000640 | −0.000034 | +0.000635 | **−0.00000580** | 304 total |
| Reserved Oct 1–22 | 750 | −0.001309 / 48.40% / 16.27% | −0.001264 / 48.80% / 19.87% | −0.000636 / 48.13% / 50.13% | −0.001506 / 48.93% / 21.07% | **−0.00024228** | 9 |

The 2024 overrides add `+0.094703` in unnormalized signed-return contribution (`+0.00000783` after dividing by 12,099 rows). The 2025 overrides subtract `0.306187` (`−0.00002522` over 12,140 rows). The nine October overrides subtract `0.181713` (`−0.00024228` over 750 rows). October's slightly higher E8 accuracy did not yield a higher return-weighted score.

## Daily variation and uncertainty

E3 versus E8 daily medians are `+0.000575`/`+0.000575` in 2023, `+0.000414`/`+0.000571` in 2024, `+0.000521`/`+0.000571` in 2025, and `−0.000525`/`−0.000525` in reserved October. Negative-day counts are E3/E8 **104/104**, **113/112**, **112/109**, and **8/8**, respectively. These daily summaries can look favorable even where the official mean score is worse, because the size of daily gains and losses matters. [Daily distribution](outputs/daily_distribution.csv) also gives 10th and 90th percentiles for all four approaches; [daily scores](outputs/daily_scores.csv) and [reserved daily scores](outputs/reserved_daily_scores.csv) provide every session.

Paired uncertainty uses **5,000 circular five-trading-session block bootstrap draws**, seed 42, resampling the same dates for E3 and E8 within each period and preserving the number of stocks per date. The table gives percentile 95% intervals for the E8−E3 official-score difference. This is an exploratory dependence-aware estimate; blocks do not guarantee coverage under market regime change.

| Period | Observed E8−E3 | 95% block-bootstrap interval |
|---|---:|---:|
| 2024 | +0.00000783 | [−0.00005411, +0.00007029] |
| 2025 | −0.00002522 | [−0.00010343, +0.00005128] |
| Equal-year CV mean | **−0.00000580** | [−0.00003840, +0.00002766] |
| Reserved October | **−0.00024228** | [−0.00045686, +0.00008504] |

For 2023 the E8−E3 difference and its interval are exactly zero by design. See [paired_uncertainty.csv](outputs/paired_uncertainty.csv) for full precision. The intervals include zero; the observed nested validation mean and later-period score both favor E3.

## Reserved-period integrity and limits

The October check used 15 complete sessions, **October 1–22, 2026**, after freezing `0.01` from 2023–2025 only. It used the saved E1 model trained through December 2025, its saved fitted preprocessor, and shared updated F0; neither was fitted or modified. The update files mark **October 23** adjusted prices as `−1` for all target stocks, so that invalid session and all subsequent dates were excluded from this final check. No local earlier October evaluation was found, but this report cannot establish whether someone examined those dates outside the repository. October is therefore a limited later-period check, not official leaderboard evidence.

The raw CSV dates still do not verify publication times, and supplied technical-indicator settings remain undocumented. E8 inherits those central F0 assumptions. No Phase 1 or Jan–May labels, leaderboard feedback, new training, or new feature definitions were used.

## Reproduce

From the project root, run `uv run python -m members.zhenci.e8.evaluate_e8 --output-dir members/zhenci/e8/recheck`. The [script](evaluate_e8.py) checks saved fold keys and identical seed probabilities, selects each override distance only from earlier OOS folds, computes scores with `shared.evaluation.evaluate_predictions`, verifies the frozen artifact and previous-session alignment for October, and writes fresh outputs without overwriting prior results. Exact selections and input hashes are in [selection.json](outputs/selection.json).
