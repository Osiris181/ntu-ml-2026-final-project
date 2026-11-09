# E1 versus E3: Phase 1 generalization diagnostic

## 1. Identical competition observations and predictions

The official Phase 1 template specifies the Cartesian product of its **50 stock IDs and 84 date columns, June 1–September 30, 2026**: 4,200 stock-date rows. The original market files supply adjusted open/close through August 31; `updated_data_7_Nov/data/market/` supplies September. Adjusted prices are present for all 4,200 keys. The competition server's private row-level scoring data are unavailable, but scoring these exact keys locally with `shared.evaluation.evaluate_predictions` reproduces both user-reported competition scores to six decimal places. This strongly supports, but does not independently prove, that the server used the whole template and the same supplied adjusted returns. The exact rows are in [`e1_e3_gap_outputs/paired_rows.csv`](e1_e3_gap_outputs/paired_rows.csv).

| Approach | 2023–2025 mean CV official score | Phase 1 official score | Phase 1 accuracy | Predicted up |
|---|---:|---:|---:|---:|
| E1: frozen XGBoost, `p > 0.5` | 0.00058145 | **-0.00031791** | 50.62% | 23.40% |
| E3: same `p`, prior-TAIEX rule | 0.00064050 | **-0.00006194** | 51.33% | 23.45% |

E3 **improved** the Phase 1 score by `0.00025597` over E1; its rule did not cause the negative result. The two submissions disagree on **254/4,200** rows: 126 E1-up to E3-down after a negative prior TAIEX return and 128 E1-down to E3-up after a nonnegative return. There were no missing-TAIEX template days. Among changed rows, 134 changes improved the score, 106 hurt it, and 14 involved a zero realized return. The saved model, frozen preprocessing, updated F0 rows, and selected E3 rule reproduce **every prediction** in both submitted CSVs: zero mismatches.

## 2. Leakage and implementation audit

**Verified in the local files and code:** The saved E1 artifact records 145,847 training rows from 100 stocks, ending December 31, 2025. Rechecking the shared holdout split found the same rows and no 2026 training labels. The September update loader drops `y`; F0 has no label or target-day OHLC column. The artifact's model, preprocessor, and feature-definition hashes match their files. Independently fitting the baseline preprocessor on the 2020–2025 training rows reproduced its saved medians, means, scales, category vocabulary, and output column order. Inference loads this frozen state and calls `transform`, without fitting it again.

The 4,200 updated F0 rows match a fresh build from the supplied original and updated market/global files across every numeric feature and weekday. For all template rows, stock-history source dates are strictly before the prediction date, and institutional source dates precede the stock-history `t-1` dates. An independent TAIEX as-of join reproduced the F0 `taiex_return` on all 84 dates: its source date is strictly earlier than prediction date, at most five calendar days old, and never missing here. E3 uses `0.525` if this prior return is negative, `0.475` otherwise, and `0.5` if missing. Stock IDs, date order, 21 F0 feature order, and all 4,200 template keys match. Both CSVs come from the **same probability vector**; only the intended decision rule differs. **No implementation error or date-based look-ahead was found in these checks.**

**Unverified assumptions:** Source CSV dates do not establish publication times before 09:00 Taiwan time. Actual availability of stock, institutional, technical, and cross-time-zone global observations, possible source revisions, and the undocumented settings of supplied technical indicators remain unresolved. The conservative `t-1`/`t-2`/strict-prior-global policy is verified as implemented, not externally certified as leakage-free. The serialized booster does not independently list its training-row keys; its provenance rests on the saved manifest and training code. The competition server's private scoring rows and labels cannot be inspected directly.

## 3. Daily performance

The complete 84-day breakdown is [`e1_e3_gap_outputs/daily_scores.csv`](e1_e3_gap_outputs/daily_scores.csv). Each daily score is the official signed-return mean over the same 50 stocks.

| Worst E3 date | Prior TAIEX | E1 score | E3 score | E1/E3 disagreements |
|---|---|---:|---:|---:|
| 2026-07-02 | Nonnegative | -0.019332 | **-0.019332** | 0 |
| 2026-07-03 | Negative | -0.018060 | **-0.018060** | 0 |
| 2026-06-09 | Negative | -0.015624 | **-0.016783** | 1 |
| 2026-06-08 | Negative | -0.009538 | **-0.015308** | 5 |
| 2026-09-01 | Negative | -0.013226 | **-0.013978** | 1 |

E3 had a negative score on **41 of 84 days**; the five worst negative days account for **31.2%** of the sum of negative daily scores, and the ten worst for **53.1%**. Losses therefore span many sessions, although a few large rising days matter. June and August had negative monthly mean E3 scores (`-0.000821` and `-0.001133`); July and September were positive (`0.001323` and `0.000336`). On July 2, 34 of 50 stocks rose but both approaches predicted up for only six; on July 3, 41 rose but only three were predicted up. They made identical predictions on both days. The largest single E3 loss was stock `3661` on July 2: it rose **13.84%** while both approaches predicted down. The largest *E3-versus-E1* harmful flips include `2345` on September 8 (E3 switched to up; stock fell **8.22%**) and `1303` on July 24 (switched to up; fell **7.31%**).

The E3 rule helped on average under **both** observed TAIEX conditions: across 39 negative-prior-TAIEX dates, E1/E3 scores were `-0.000077`/`0.000232`; across 45 nonnegative dates, they were `-0.000526`/`-0.000317`. E3 improved 39 days, hurt 33, and tied 12. Its advantage was too small to make the whole period positive.

## 4. Explanation and recommendation

**Supported explanation:** Both approaches' frozen XGBoost probabilities led to mostly down predictions (about 76.5%) on this Phase 1 period. The realized mean adjusted intraday return was slightly positive (`0.000204`), and E3 predicted down on **1,520 rising stock-days**. Correct down calls and some correct up calls did not fully offset those missed gains and other wrong calls. E3's threshold changed only 254 labels and improved the score relative to E1, but could not repair the shared model's losses on days where both made the same predictions. This explains how E3 could beat E1 while still posting a negative score. A changed market environment or probability calibration could be involved, but these are **hypotheses**, not established causes. The negative score alone does not prove overfitting.

**Recommendation: investigate further.** Keep E3 as the better observed Phase 1 comparator, while recognizing that its CV advantage did not make Phase 1 profitable and the previously recorded January–May holdout favored E1 (`0.00152019` versus `0.00144758`). Do not use this Phase 1 diagnosis to retune E3 thresholds or replace the independent holdout selection rule.

## Reproduce

From the repository root, run `uv run python -m members.zhenci.diagnostics.e1_e3_gap --output-dir members/zhenci/diagnostics/e1_e3_gap_recheck`. The script checks source and artifact hashes, rebuilds template F0 in memory, reproduces both submission files from the frozen model, calculates scores with the shared official scorer, and writes only new diagnostic files: [`summary.json`](e1_e3_gap_outputs/summary.json), `paired_rows.csv`, and `daily_scores.csv`. It does not train a model or modify an existing result. It refuses to overwrite an existing output directory.
