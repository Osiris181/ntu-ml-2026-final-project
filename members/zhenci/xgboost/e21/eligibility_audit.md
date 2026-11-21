# E21 data availability and deployment eligibility

**January–May 2026 is technically usable and remains reserved for team holdout evaluation.** The user confirmed that the existing holdout policy stays in force. This audit checks the supplied observations for availability and validity. The historical E21 fits use 2023–2025 evaluation periods and their corresponding earlier training windows.

## Supplied observations and labels

| Source | Coverage and role |
| --- | --- |
| Original market archive | The January–May 2026 interval has 9,500 observations, covering all metadata Top100 stocks on each of 95 actual trading dates from January 2 to May 29. The Top50 subset contains 4,750 rows. |
| Original global archive | The four F0 series are available under the existing strict prior-date alignment and five-calendar-day limit. February 23 has no eligible observation for any of the four series, resulting in 100 missing values per global field. |
| November 7 update | 4,500 rows from September 1 through November 6, all 100 stocks. It starts after the proposed May cutoff and contributes no training labels to the comparison. The audit detected 100 rows with nonpositive adjusted opens or closes. |
| November 17 update | 5,600 rows from September 1 through November 17, all 100 stocks. It likewise contributes no training labels. The audit detected 500 rows with nonpositive adjusted opens or closes. |

All 9,500 original January–May adjusted opens and closes are finite and positive. Every supplied binary label equals `adjusted_close > adjusted_open`. No stock-date row is missing in that interval. The original and both updated datasets passed the shared schema checks. The January–May rows come from the *same* original market files as the earlier training years and use the same adjusted price series and reported share volume, transaction count, and traded amount columns. The field checks found no nonpositive or missing values in those seven raw inputs in the January–May panel. These checks establish consistency of the supplied schema and observed values. The files do not independently document the provider’s adjustment methodology or verify the units or publication history of later updates. [Raw field distributions](audit_outputs/raw_numeric_checks.csv), [daily coverage](audit_outputs/daily_coverage.csv), [per-stock coverage](audit_outputs/stock_coverage.csv).

## Features and information timing

There is exactly one shared F0 row for every eligible January–May label row. The only F0 missingness in this interval is the 400 global values described above. The shared preprocessor’s training-fitted imputation policy covers such gaps.

The audit removed all market and global observations after May and removed `y` from the feature input. The independently rebuilt January–May F0 slice reproduced the archived slice, including missingness. Each audited price and activity source date preceded the prediction date, and the institutional date preceded the price source date. An independent date join reproduced each global field and verified source ages of one to five calendar days, with missing values when no eligible observation existed. The team’s conservative pre-09:00 policy remains a date-based proxy. Actual publication times and historical source revisions remain unverified. [F0 missingness](audit_outputs/f0_missingness.csv), [global source dates](audit_outputs/global_provenance.csv), [audit summary and source hashes](audit_outputs/summary.json).

## Competition and team protocol

The supplied course brief identifies `fin_pred_data.zip` as the training dataset on page 1 and dates Phase 1 from June 1 through September 30, 2026 on page 2. Its supplied-data section does not state an earlier January–May exclusion. The public competition site was unavailable for independent verification of any later guidance. The team’s own protocol reserves January–May 2026 for holdout evaluation and identifies the deployment cutoff as a separate decision. The user chose to keep that period reserved, which settles the deployment question for E21. [Course brief](../../../../final_project.pdf), [team protocol](../../../../AGENTS.md).

The January–May labels were checked for validity only. They were excluded from fitting and selection. Phase 1 is an already-examined development benchmark, and its outcomes were excluded from this experiment. E21 saved the three permitted historical fold models. No updated deployment model, Phase 1 CSV, or prospective Phase 2 prediction pair was produced under the retained cutoff policy.
