"""Render an evidence-based L1 report from the historical output tables."""

import pandas as pd

from members.zhenci.logistic_regression.l1.analysis import SELECTION_POLICY


def markdown(frame):
    names = list(frame.columns)
    formatted = frame.map(lambda x: f"{x:.7f}" if isinstance(x, float) and pd.notna(x)
                          else str(x) if pd.notna(x) else "—")
    return "\n".join(["| " + " | ".join(row) + " |" for row in
                      [names, ["---"] * len(names), *formatted.to_numpy().tolist()]])


def report(metrics, overall, uncertainty, probs, gaps, old_probs, stability, families, corr, times, selected):
    year = metrics.pivot(index="arm", columns="fold", values="official_score").reset_index()
    score_table = overall[["arm", "official_score", "delta_e3", "years_above_e3",
                           "worst_four_month_score", "worst_four_month_period"]].merge(year, on="arm")
    score_table = score_table[["arm", "2023", "2024", "2025", "official_score", "delta_e3",
                               "years_above_e3", "worst_four_month_score", "worst_four_month_period"]]
    decision = metrics.groupby("arm", sort=False)[["accuracy", "predicted_up_rate", "actual_up_rate",
        "mean_daily_spearman", "market_direction", "stock_selection", "daily_score_sd", "negative_day_rate"]
        ].mean().reset_index()
    intervals = uncertainty.loc[uncertainty["fold"].eq("equal_year") & uncertainty["reference"].eq("E3"),
                                ["arm", "delta", "low_95", "high_95"]]
    probability = probs.loc[probs["arm"].eq("L1-A"),
                            ["fold", "split", "bce", "constant_bce", "auc", "brier", "constant_brier",
                             "p10", "p50", "p90"]]
    gap = gaps.loc[gaps["arm"].eq("L1-A")].drop(columns="arm")
    top = stability.loc[stability["family"].eq("numeric")].head(6)
    intercept = stability.loc[stability["family"].eq("intercept"), ["2023", "2024", "2025"]]
    primary = overall.set_index("arm").loc["L1-A"]
    primary_gap = float(gap["validation_minus_train_top100_auc"].mean())
    e3 = old_probs.loc[old_probs["arm"].eq("E3")]
    e3_gap = float((e3["validation_auc"] - e3["train_auc"]).mean())
    l1_val = probs.loc[probs["arm"].eq("L1-A") & probs["split"].eq("validation")]
    l1_auc = l1_val["auc"].mean()
    l1_bce = l1_val["bce"].mean()
    interval = intervals.set_index("arm").loc["L1-A"]
    if selected:
        recommendation = (
            f"Prepare one exploratory Phase 1 file using **{selected}**. "
            "The historical screen supports checking the frozen configuration on the Phase 1 development benchmark."
        )
    else:
        recommendation = (
            "**Retain E3 as the reference and skip a Phase 1 L1 submission.** "
            "Keep L1 as the linear benchmark. These folds give no basis for tuning this Logistic Regression recipe."
        )
    wraps_zero = interval["low_95"] <= 0 <= interval["high_95"]
    interval_text = "includes zero" if wraps_zero else "excludes zero"
    return f"""# L1 — Logistic Regression baseline

{recommendation}

L1-A scored **{primary['official_score']:+.8f}** on the equal-year official metric, a
**{primary['delta_e3']:+.8f}** difference from E3, with {primary['years_above_e3']} of 3 yearly
wins. Its paired 95% interval {interval_text}. The 2023–2025 folds have been reused throughout
this project, so the results are exploratory.

## Historical performance

{markdown(score_table)}

One unweighted L2 model per fold used the 21 shared F0 fields, 100 training stocks, and identical
archived 50-stock evaluation keys. L1-A uses a strict cutoff of 0.5. L1-B applies the archived E3
cutoffs of 0.525 after a negative eligible TAIEX return, 0.475 after a nonnegative return, and
0.5 on missing values. E1, E3 and E20 scores were checked against their archives.
Previous direction uses the last completed stock session's adjusted intraday direction.
Its correspondence to the official leaderboard's Previous day baseline remains unverified.

{markdown(decision)}

Day-level market contribution equals mean signed position times mean realized return.
Stock-selection contribution is the residual. Period contributions weight days by their
observation counts. The reported cross-sectional Spearman coefficient is the mean of defined
trading-day correlations. Previous direction supplies a binary call only.
See [daily scores](daily_scores.csv), [monthly scores](monthly_scores.csv) and
[four-month blocks](four_month_scores.csv).

{markdown(intervals)}

Intervals use 3,000 paired circular bootstrap replicates with five-trading-day blocks within
each year. All stocks from a resampled date move together. See the
[full paired intervals](paired_uncertainty.csv) for comparisons against all four controls.

## Probability estimation and runtime

L1's mean validation AUC was **{l1_auc:.4f}**, with a mean validation-minus-Top100-training gap
of **{primary_gap:+.4f}**. Archived E3 had a mean validation AUC of **{e3['validation_auc'].mean():.4f}**
with a gap of **{e3_gap:+.4f}**. Mean validation BCE was **{l1_bce:.4f}** for L1
and **{e3['validation_bce'].mean():.4f}** for E3. Probability fit and return-weighted performance
measure different outcomes. The in-sample rows measure fit, and the validation rows measure
out-of-sample performance on the three reused folds.

{markdown(probability)}

The constant probability is each fold's Top100 training up rate, held fixed on every split.
Both L1 decision rules share these probability metrics. See [training–validation gaps](probability_metric_gaps.csv)
for both training universes and [archived probability metrics](archived_xgboost_probability_metrics.csv)
for the E19/E20 reference.

{markdown(times)}

Fit includes preprocessing and optimizer time. Inference includes validation transformation and
probability estimation.

## Coefficients and prediction explanations

These are the largest numerical coefficients by mean absolute magnitude across folds.

{markdown(top[["feature", "2023", "2024", "2025", "mean_abs", "sign_flip"]])}

The intercept is the fitted log-odds for an all-zero encoded vector. The mean validation
probabilities above measure actual model output after feature offsets.

{markdown(intercept)}

{markdown(families)}

Family effects sum the signed log-odds contributions within a family before taking absolute
values. The fitted logit and sigmoid probability were reconstructed for every validation
observation. [Example explanations](prediction_explanations.csv) provide all encoded contributions
for deterministic L1-A examples. See [all coefficients](coefficients.csv),
[cross-fold stability](coefficient_stability.csv) and [training numerical correlations](numerical_correlations.csv).
The correlation audit found {len(corr)} fold/pair entries at absolute Pearson correlation of at least 0.85.
Coefficients are conditional statistical associations. Correlated predictors can share effects
and coefficient sign changes are descriptive.

## Reproduction and Phase 1

Run from the repository root with a fresh output directory.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.logistic_regression.l1.train_l1 --output-dir members/zhenci/logistic_regression/l1/reproduction
```

Selection rule used before fitting. {SELECTION_POLICY}
January–May 2026, Phase 1 and October outcomes were excluded from fitting and selection.
[Run manifest](summary.json) records the input and model checksums.
"""
