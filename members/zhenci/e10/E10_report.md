# E10: why E3 predicts down so often

**Status:** diagnostic only. No model was trained, tuned, selected, or changed. E3 remains the reference approach; this report does not use later outcomes to change it.

## Scope and reproduction

The [diagnostic script](diagnose_e10.py) reuses archived 2023-2025 out-of-sample XGBoost probabilities, the saved January-May 2026 holdout predictions, the archived Phase 1 probabilities, and the frozen 2020-2025 model and preprocessor for October 1-22. It reconstructs the **unchanged E3 rule**: predict up if `p_up > 0.525` after a negative eligible prior TAIEX return, `> 0.475` after a nonnegative return, or `> 0.5` if missing. The script verifies saved E3 labels, frozen artifact and F0 hashes, exact stock/date keys, and the official score calculation. The official score is the mean of `+ intraday_return` for an up prediction and `- intraday_return` for a down prediction. Here, actual up means adjusted close **strictly above** adjusted open; flat days count as not up. It excludes October 23's placeholder and later October/November observations.

Run from `final_project`:

```text
uv run python -m members.zhenci.e10.diagnose_e10 --output-dir members/zhenci/e10/reproduction
```

The script refuses to overwrite [existing outputs](outputs/). Its [manifest](outputs/manifest.json) records source hashes and the selected rule. Detailed per-period, [monthly](outputs/monthly_summary.csv), [daily](outputs/daily_summary.csv), [probability-bin](outputs/probability_bins.csv), [feature](outputs/feature_shift.csv), and [threshold-condition](outputs/threshold_conditions.csv) tables are included. All 2026 comparisons below use **the same frozen model**. The three historical folds use different models trained on expanding windows, so differences across 2023-2025 are not pure probability drift. The January-May holdout and October check were previously inspected; Phase 1 was competition feedback. They are diagnostic observations, not fresh selection data.

## 1. Predictions versus actual direction

| Period | Rows / days | E3 up | Actually up | Mean / median `p_up` | Mean intraday return | Official score |
|---|---:|---:|---:|---:|---:|---:|
| 2023 validation | 11,950 / 239 | 47.6% | 45.4% | .490 / .485 | +.0045% | +.000611 |
| 2024 validation | 12,099 / 242 | 34.9% | 45.0% | .449 / .442 | -.0338% | +.000391 |
| 2025 validation | 12,140 / 243 | 33.5% | 43.5% | .452 / .444 | -.0447% | +.000919 |
| Jan-May 2026 holdout | 4,750 / 95 | 30.3% | 42.4% | .444 / .441 | -.1550% | +.001448 |
| Jun-Sep 2026 Phase 1 | 4,200 / 84 | 23.5% | 47.2% | .427 / .423 | +.0204% | -.000062 |
| Oct 1-22 2026 | 750 / 15 | 19.9% | 51.3% | .414 / .405 | +.1879% | -.001264 |

Within the frozen-model sequence, predicted-up rates and mean probabilities fall while actual-up rates and average returns rise. Thus the increasingly bearish **aggregate** predictions do not match an increasingly bearish market. This is not a monotonic day-by-day trend: the model predicted up for 41.0% of May rows and 29.3% of July rows, versus 19.1% in September. The [monthly up-rate plot](outputs/up_rates.svg) and daily table show that variation.

The E3 threshold rule is **not the main source** of the bearish rate. At the plain `0.5` cutoff, frozen-model up rates are 29.1%, 23.4%, and 16.3% in the holdout, Phase 1, and October, respectively; E3 gives 30.3%, 23.5%, and 19.9%. E3 changes 371, 254, and 53 labels in those periods. Its prior-TAIEX rule can help one condition and hurt another: in Phase 1, it raises the overall score from E1's -.000318 to -.000062, while October's seven negative-TAIEX sessions have only 5.1% E3-up predictions despite 54.6% actually-up observations. Lower model probabilities account for most of the down-prediction increase.

## 2. F0 distribution shift

For each period, [feature_shift.csv](outputs/feature_shift.csv) compares all **19 numeric** F0 fields with that fold's earlier training years restricted to the same 50 target stocks. [training_all100_feature_shift.csv](outputs/training_all100_feature_shift.csv) repeats the comparison against the actual 100-stock training universe; its leading shifts are similar. Means, medians, 10th/90th percentiles, IQRs, and missing rates are saved. PSI is a descriptive divergence based on training deciles, not a significance test; October estimates use only 15 sessions. The two categorical F0 fields are in [categorical_shift.csv](outputs/categorical_shift.csv): all 50 evaluation stock IDs and five weekdays were represented in training, with no unseen category. Stock-ID frequency total variation is about .0003; weekday total variation is .028 in Phase 1 and .070 in October.

| F0 feature | Training median, 50 stocks | Phase 1 median | Phase 1 PSI | October median | Frozen-model gain rank |
|---|---:|---:|---:|---:|---:|
| Normalized VIX | .1906 | .1606 | 3.38 | .1552 | 1 |
| Normalized ATR | .0228 | .0361 | 1.03 | .0292 | 6 |
| Previous price range | .0188 | .0323 | .66 | .0265 | 19 |
| Bollinger width | .0440 | .0725 | .38 | .0566 | 14 |
| Previous Nasdaq return | .00144 | -.00100 | .16 | -.00023 | 4 |
| Previous TAIEX return | .00106 | .00157 | .17 | .00222 | 3 |
| Relative volume, 20 sessions | .8847 | .8432 | .01 | .9602 | 20 |

The frozen model's top four total-gain fields are VIX, S&P 500 return, TAIEX return, and Nasdaq return, together 49.1% of training split gain. VIX is strongly shifted and influential, but split gain does **not** establish that VIX caused the predictions or the score decline. The October S&P 500 and Nasdaq PSI values are 2.22 and 1.30; these short-period global series repeat across 50 stocks each date, so they are 15 daily observations, not 750 independent global observations. [global_daily.csv](outputs/global_daily.csv) gives the once-per-date view.

Stock-history volatility also changed: the prior intraday-return 10th/90th percentiles widen from -1.92%/+1.82% in 2020-2025 target-stock training to -3.09%/+3.11% in Phase 1. Prior adjusted daily-return tails widen from -2.06%/+2.33% to -3.09%/+3.88%. Relative-volume median moves only modestly overall (.885 to .843), though stocks 2912, 3231, 2207, and 2603 have Phase 1 medians about .14-.16 below their own training medians. In October, stocks 1301 and 6505 instead have notably higher relative volume. The [stock table](outputs/stock_shift.csv) and [monthly F0 table](outputs/monthly_feature_summary.csv) locate these changes; ATR's monthly median is around .036 in June-August versus .0228 in training, while VIX's monthly median falls near .153 in August-September. The [shift plot](outputs/feature_shift.svg) ranks Phase 1 median changes in training IQR units. These are associations, not causal explanations.

## 3. Direction calibration and return-weighted score

| Period | Direction Brier score | Fixed 0.1-bin ECE |
|---|---:|---:|
| 2023 | .2624 | .0933 |
| 2024 | .2622 | .0930 |
| 2025 | .2557 | .0742 |
| Jan-May 2026 | .2512 | .0677 |
| Phase 1 | .2678 | .1029 |
| October 1-22 | .2726 | .1194 |

For reference, a constant `p_up = 0.5` has Brier score .25 for binary direction on **any** of these samples. Thus these saved probabilities have weak direction calibration, especially in Phase 1 and October. ECE depends on the fixed bins and small extreme bins; the [calibration plot](outputs/calibration.svg) shows the 2026 frozen-model periods. In Phase 1, the 388 predictions in the `.2-.3` bin average probability .262 but 52.3% actually rise, with mean return +.288%; E3 predicts all of them down. The 1,209 predictions in `.3-.4` average .356, yet 47.0% rise and mean return is +.128%. October's `.2-.3` bin has 56 observations, 69.6% rising and mean return +.974%; that small bin should not be generalized on its own.

Direction calibration does **not** equal calibration of expected intraday returns. A stock can be less than 50% likely to rise yet have positive expected return if its possible gains are larger than its possible losses. The official score rewards the signed **size** of each return, while the XGBoost model was fitted for binary direction. Phase 1 score components, each expressed as a contribution to the period-wide mean, are:

| E3 decision and actual direction | Rows | Score contribution |
|---|---:|---:|
| Correct down | 1,695 | +.007511 |
| Correct up | 461 | +.002367 |
| Incorrect down | 1,520 | -.007644 |
| Incorrect up | 524 | -.002296 |
| **Total** | **4,200** | **-.000062** |

The missed rising stocks cancel the large correct-down contribution. On the January-May holdout, the corresponding incorrect-down contribution is -.004965, against correct-down +.006464; the bearish decisions still produce a positive overall score because that period's mean return is negative. October's incorrect-down contribution is -.007657, again larger than correct-down +.006086. This is direct evidence of payoff mismatch, not proof that a different training objective would work.

## 4. Daily losses and interpretation

The [daily score plot](outputs/daily_score.svg) relates each session's mean stock return to its E3 score. Phase 1 loses on **41 of 84** days; its ten worst losing days account for **53.1% of the sum of negative daily scores**. Losses are therefore both concentrated and spread across many sessions. On the 16 Phase 1 days when the 50-stock mean intraday return exceeds +1%, E3's mean daily score is **-1.10%**. The worst sessions are July 2 (E3 up 12%, actually up 68%, score -1.93%), July 3 (6% / 82%, -1.81%), June 9 (20% / 76%, -1.68%), June 8 (16% / 76%, -1.53%), and September 1 (6% / 76%, -1.40%). Those dates show why large positive-return sessions are costly to a mostly-down rule. October has only 15 days, eight with negative E3 scores.

**Verified:** the frozen model's probability distribution becomes lower in later 2026 periods while actual up frequency rises; E3's TAIEX threshold only modestly changes the overall up fraction; Phase 1 has poorer direction calibration and large incorrect-down return losses; several influential F0 inputs have changed distributions. The 2023-2025 fold sequence cannot establish model drift because each fold used a different fit.

**Plausible, unproven:** changing market regimes and input distributions may affect the frozen model's probabilities; optimizing binary direction may be poorly aligned with return-weighted decisions. Feature shift and gain ranks alone do not prove either mechanism. Date-only sources still do not verify publication times for market, institutional, technical, or global fields, and the supplied technical-indicator settings remain unverified. This audit reuses the agreed F0 lag assumptions; it does not resolve those data-quality questions. Cross-stock rows within a trading day share market conditions, so this report makes descriptive comparisons and no stock-row independence or statistical-significance claim.

**Next experiment, if pursued:** test one predeclared, score-aligned two-part model on historical chronological folds: estimate up probability **and** conditional gain/loss magnitudes from F0, then predict the sign of estimated intraday return. This directly tests whether payoff size adds information beyond the current direction model. It differs from E7's fixed historical payoff averages and E2's single direct-return regressor. Select all configuration choices using earlier training/validation years only, compare against frozen E3 on identical keys, and reserve Phase 1, the inspected holdout, and October from selection. This is a recommendation for a future experiment, **not** an E10 implementation or a claim it will improve the score.
