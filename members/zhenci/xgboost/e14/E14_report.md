# E14: Daily 25/25 ranking of frozen E3 and E12 probabilities

## Decision

**Historical selection: no ranking candidate.** On the identical complete-date historical evaluation rows, E3 ranking scores below original E3 in **all three** years. E12 ranking helps in 2023 but loses to original E12 in 2024 and 2025; its equal-year mean is also below original E3. The uncertainty intervals include zero, so the historical evidence does not support replacing either original decision rule with a fixed 25/25 split. E3 remains the project reference. No Phase 1 labels, Phase 1 outcomes, or leaderboard feedback were used here. At the user's later explicit request, an **exploratory E12-ranking Phase 1 CSV** was generated on 2026-11-18; this does not change the historical selection result.

The numerical tables below were calculated directly from the already saved E12/E3 out-of-sample prediction file for a read-only assessment. The user then ran the reproducible [E14 historical runner](rank_e14.py), which verified the archived sources, wrote the comparison CSVs, and recorded `selected_candidate: null`. Its ordinary Phase 1 export mode remains blocked when no candidate was selected. The separate [exploratory exporter](export_e14_exploratory.py) records the user's explicit override without changing that selection file.

## Inputs and ranking rule

The archived `p_up` is the upward-class probability from the existing XGBoost `binary:logistic` model used by E3; E3's original binary labels then apply the previously selected, historically lagged TAIEX threshold rule. The archived `p_lstm` is E12's sigmoid probability of the upward class; original E12 uses `p_lstm > 0.5`. Thus **higher** values mean a stronger upward prediction for both. E14 ranks each model's probabilities from high to low within each trading date, breaks exact ties by ascending stock ID, predicts 1 for the top 25, and predicts 0 for the other 25. No model weights, fitted preprocessing, features, or original decisions are changed.

The saved 2023–2025 validation folds have **11 incomplete dates with 49 target stocks**: one in 2024 and ten in 2025. An exact 25/25 rule is undefined for those dates. The comparison therefore excludes the whole incomplete date for **all four arms**, leaving 239 complete dates (11,950 observations) in 2023, 241 (12,050) in 2024, and 233 (11,650) in 2025. All scores and intervals below use these identical 35,650 stock-date observations. This explains why some original-model scores differ slightly from the earlier full-fold E12/E13 reports. The E14 runner lists the excluded dates and missing stocks in `excluded_incomplete_dates.csv` when run.

## Historical official score and accuracy

| Year | Decision rule | Official score | Accuracy | Predicted up | Positive-score days |
| --- | --- | ---: | ---: | ---: | ---: |
| 2023 | E3 original | 0.000611 | 52.08% | 47.60% | 56.49% |
| 2023 | E3 daily ranking | 0.000310 | 51.76% | 50.00% | 56.49% |
| 2023 | E12 original | 0.000211 | 53.77% | 20.93% | 51.88% |
| 2023 | E12 daily ranking | 0.000535 | 52.38% | 50.00% | 59.41% |
| 2024 | E3 original | 0.000392 | 52.65% | 34.98% | 53.11% |
| 2024 | E3 daily ranking | 0.000323 | 50.85% | 50.00% | 53.94% |
| 2024 | E12 original | 0.000709 | 54.86% | 17.19% | 58.92% |
| 2024 | E12 daily ranking | 0.000583 | 51.50% | 50.00% | 55.19% |
| 2025 | E3 original | 0.000933 | 53.85% | 33.67% | 54.51% |
| 2025 | E3 daily ranking | 0.000507 | 51.39% | 50.00% | 57.08% |
| 2025 | E12 original | 0.001064 | 56.03% | 12.24% | 52.79% |
| 2025 | E12 daily ranking | 0.000490 | 52.27% | 50.00% | 57.51% |
| **Equal-year mean** | **E3 original** | **0.000646** | — | — | — |
| **Equal-year mean** | **E3 daily ranking** | **0.000380** | — | — | — |
| **Equal-year mean** | **E12 original** | **0.000661** | — | — | — |
| **Equal-year mean** | **E12 daily ranking** | **0.000536** | — | — | — |

The scoring function is the shared official mean of `(2 × prediction − 1) × adjusted intraday return`. Accuracy and the proportion of positive days are descriptive; they are not the selection objective. E12 ranking's 2023 gain is not stable across years.

## Daily distribution and cross-sectional signal

| Year | Rule | Daily score p10 | Daily median | Daily score p90 |
| --- | --- | ---: | ---: | ---: |
| 2023 | E3 original / ranking | −0.003931 / −0.002616 | 0.000575 / 0.000368 | 0.005731 / 0.003384 |
| 2023 | E12 original / ranking | −0.005102 / −0.003131 | 0.000166 / 0.000806 | 0.005637 / 0.004363 |
| 2024 | E3 original / ranking | −0.005065 / −0.003079 | 0.000432 / 0.000172 | 0.005618 / 0.003946 |
| 2024 | E12 original / ranking | −0.005993 / −0.003639 | 0.000836 / 0.000391 | 0.006849 / 0.005112 |
| 2025 | E3 original / ranking | −0.004799 / −0.003325 | 0.000558 / 0.000699 | 0.006922 / 0.004067 |
| 2025 | E12 original / ranking | −0.006126 / −0.003338 | 0.000388 / 0.000703 | 0.008203 / 0.004171 |

The ranking rule narrows both tails of daily returns. It improves some bad-day outcomes but also gives up favorable large days. The mean official score still declines overall.

| Year | Probability source | Mean daily Spearman with realized return | Mean top-25 minus bottom-25 return | Days with positive top-bottom spread |
| --- | --- | ---: | ---: | ---: |
| 2023 | E3 XGBoost | 0.0470 | 0.000621 | 56.49% |
| 2023 | E12 LSTM | 0.0654 | 0.001070 | 59.41% |
| 2024 | E3 XGBoost | 0.0355 | 0.000646 | 53.94% |
| 2024 | E12 LSTM | 0.0565 | 0.001165 | 55.19% |
| 2025 | E3 XGBoost | 0.0422 | 0.001015 | 57.08% |
| 2025 | E12 LSTM | 0.0643 | 0.000980 | 57.51% |

These are small but positive cross-sectional associations. With exactly 25 up and 25 down calls, each day's ranking score equals **half** the top-bottom mean-return spread. The original rules can also express a net daily market-direction view; fixed 25/25 ranking removes that exposure. This is a plausible explanation for positive ranking signal alongside lower total official score, not proof that one component caused the full difference.

## Changed predictions and paired uncertainty

| Year | Ranking applied to | Changed stock predictions | Official-score change caused by those changes |
| --- | --- | ---: | ---: |
| 2023 | E3 | 3,129 | −0.000301 |
| 2024 | E3 | 3,148 | −0.000069 |
| 2025 | E3 | 3,365 | −0.000426 |
| 2023 | E12 | 3,574 | +0.000324 |
| 2024 | E12 | 4,160 | −0.000127 |
| 2025 | E12 | 4,515 | −0.000574 |

The differences are computed only from rows whose binary prediction changed, divided by the same 50-stock daily evaluation universe. Unchanged rows contribute zero to the paired difference.

| Paired score difference | 2023 point [95% interval] | 2024 point [95% interval] | 2025 point [95% interval] | Equal-year point [95% interval] |
| --- | --- | --- | --- | --- |
| E3 ranking − E3 | −0.000301 [−0.000764, +0.000148] | −0.000069 [−0.000603, +0.000440] | −0.000426 [−0.001124, +0.000175] | −0.000265 [−0.000594, +0.000038] |
| E12 ranking − E12 | +0.000324 [−0.000113, +0.000751] | −0.000127 [−0.000675, +0.000434] | −0.000574 [−0.001364, +0.000166] | −0.000126 [−0.000481, +0.000210] |
| E12 ranking − E3 | −0.000077 [−0.000702, +0.000527] | +0.000190 [−0.000372, +0.000740] | −0.000443 [−0.001154, +0.000232] | −0.000110 [−0.000484, +0.000241] |

Intervals use 3,000 paired circular-bootstrap draws of five consecutive **trading days** within each year, seed 42. All 50 stocks on a day move together; years are resampled separately before averaging their scores equally. These intervals describe dependence within repeatedly inspected validation years and are not a fresh generalization test.

## Selection and Phase 1 preparation

The E14 runner selects at most one ranked candidate only if its equal-year score exceeds **both** its own original rule and original E3, its improvement over its own rule is positive in at least two of three years, and its paired equal-year 95% interval versus its own rule has a positive lower bound. **Neither candidate passes even the mean-score comparison.** The official leaderboard's “Previous day” baseline remains unverified; the separately investigated previous-direction candidate is not part of E14.

The ordinary Phase 1 branch remains gated by the historical selection file. In a later, explicit user-requested exploratory run, the stronger of the two historical ranking variants—**E12 ranking**—was exported using the existing frozen E12 model, fitted preprocessing, historical F0, and official 84-date, 50-stock template. The new file is [zhenci_e14_e12_rank_exploratory_phase1.csv](../../../../submissions/zhenci_e14_e12_rank_exploratory_phase1.csv). It contains exactly 25 up and 25 down predictions on each of the 84 dates, preserves template headers and stock order, and changes 1,577 of 4,200 predictions relative to the original E12 Phase 1 CSV. No Phase 1 labels were read, no model was retrained, and the file was not uploaded. This is an exploratory submission, **not** a historically selected replacement for E3.

To generate the saved E14 comparison files yourself from `final_project`:

```powershell
uv run --extra e12 python -m members.zhenci.xgboost.e14.rank_e14
```

The default command performs historical analysis only. It creates `members/zhenci/xgboost/e14/outputs/` with archived-input hashes, full paired predictions, yearly and daily metrics, ranking-quality and changed-prediction tables, uncertainty intervals, exclusions, and `selection.json`. It refuses to overwrite an existing E14 output directory. The optional `--mode phase1` export requires a supported candidate in that selection file; on the present evidence it will stop without writing a CSV. The later explicit exploratory export can be reproduced with `uv run --extra e12 python -m members.zhenci.xgboost.e14.export_e14_exploratory` using a new `--output` path, because the existing CSV cannot be overwritten. The original E1–E13 results, shared F0, and earlier submission files remain unchanged.
