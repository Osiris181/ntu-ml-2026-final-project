# Team findings from the XGBoost track

**Static E3 is the selected XGBoost comparator.** It uses the original F0 inputs, a model fitted through December 2025 and a fixed decision rule based on prior TAIEX returns. It averaged **0.00064050** on the full-year 2023–2025 folds. E16-A has the best documented personal Phase 1 result, **+0.000051** as reported by the user, despite a lower historical mean. Phase 1 has already supplied feedback. The one archived Phase 2 file in this track establishes that the pipeline can produce its 50 binary calls. There is no verified Phase 2 score or upload receipt for it. [Frozen specification](XGBOOST_FINAL_MODEL.md), [research synthesis](XGBOOST_FINAL_REPORT.md).

## What transfers to Random Forest, MLP and other models

| Finding | What to carry into a team comparison |
| --- | --- |
| Direction accuracy and economic value can disagree. | Report the shared official score, direction accuracy, BCE/Brier, and AUC separately. For the official scoring rule, an up decision is favored when conditional expected intraday return is positive. Its sign depends on both probability and payoff magnitude. |
| Stock ordering and market exposure contribute separately. | Report the daily mean signed call times mean return, and the within-day covariance of signed calls and returns. A 25/25 output split removes the market-direction term on complete dates. E14 improved the selection term and lowered the total in both XGBoost and LSTM on the matched dates. |
| Better probability loss can come with weaker discrimination. | Examine probability spread, reliability and ranking separately. E19 found that later E3 trees improved in-sample fit and worsened validation probability loss. E20 lowered validation BCE and Brier with smaller trees, but also lowered validation AUC and official score. Compare model-appropriate complexity diagnostics on the same chronology. |
| Results can reverse when the calendar changes. | Match stock-date keys and retain outcomes by year, month and trading day. Freeze cutoffs, features, preprocessing, artifacts and decision rules before scoring a fresh period. E21’s specific May refit improved within-day ranking in each historical period but reduced the three-year official mean. The outcome of that refit should not determine another model’s cutoff. |
| A validation win needs context. | Compare paired predictions and resample trading-day blocks, keeping the full stock cross-section together. Report the number of alternatives and which data influenced choices. Reusing the same outer folds across many decisions reduces the strength of a small win. |

## What remains specific to this track

E3’s TAIEX thresholds and preferred complexity settings come from this model and input representation. Added technical features and normalization did not produce a stable benefit for these tested configurations. That does not establish that the features are ineffective for every model, or that Random Forest or MLP is inherently better. January–May 2026 remains reserved under the team’s decision. Actual feature publication timing, historical revisions and several supplied indicator settings remain unverified.

For the comparative report, pair each member’s already frozen predictions on common observations. Document each model’s features, fitting cutoff, search budget, decision rule and official score. For prospective sessions, archive model and input identifiers, dated prediction files and generation times before the deadline, then evaluate the archived files once independently available realized returns arrive. Keep Phase 1 in its own development section and treat limited Phase 2 outcomes as limited evidence. The XGBoost optimization track ends at E21.
