"""Render the L3 report from its historical and Phase 1 reference tables."""

from members.zhenci.logistic_regression.l1.report import markdown
from members.zhenci.ridge_regression.l3.diagnostics import SELECTION_POLICY


def render_report(tables, candidate):
    fold = tables["fold_metrics"]
    scores = fold.pivot(index="arm", columns="fold", values="official_score").reset_index()
    scores = scores.merge(tables["equal_year_scores"], on="arm")
    means = fold.groupby("arm", as_index=False)[["accuracy", "direction_auc", "mean_daily_spearman",
        "predicted_up_rate", "actual_up_rate", "market_direction", "stock_selection"]].mean()
    intervals = tables["paired_uncertainty"]
    equal = intervals.loc[intervals["period"].eq("equal_year")]
    reg = tables["regression_metrics"][["fold", "split", "rows", "rmse", "mae", "pearson", "spearman",
                                       "r2_vs_zero", "r2_vs_training_mean", "predicted_mean", "predicted_sd",
                                       "realized_mean", "realized_sd"]]
    reg = reg.rename(columns={name: name + "_pct" for name in
                              ("rmse", "mae", "predicted_mean", "predicted_sd", "realized_mean", "realized_sd")})
    columns = [name for name in reg if name.endswith("_pct")]
    reg[columns] *= 100
    validation = reg.loc[reg["split"].eq("validation")]
    quintiles = tables["prediction_magnitude_quintiles"]
    res = tables["residual_diagnostics"]
    strongest = quintiles.loc[quintiles["quintile"].eq(5)].set_index("fold")
    weakest = quintiles.loc[quintiles["quintile"].eq(1)].set_index("fold")
    quintiles_top_better = int(strongest["official_score"].gt(weakest["official_score"]).sum())
    coef = tables["coefficients"]
    numeric = tables["coefficient_stability"]
    numeric = numeric.loc[numeric["family"].eq("numeric")].head(7)
    taiex = tables["threshold_sensitivity"]
    taiex = taiex.loc[taiex["regime"].eq("all"), ["fold", "mean_abs_training_return", "delta",
        "changed", "changed_fraction", "down_to_up", "up_to_down", "official_score_delta",
        "market_direction_delta", "stock_selection_delta"]]
    screen = tables["phase1_screen"]
    a, b = tuple(screen.itertuples())
    recommendation = (f"**{candidate} qualifies for one exploratory Phase 1 submission.** Keep L2-B "
                      "as the Phase 1 reference while checking the Ridge candidate."
                      if candidate else "**Keep L2-B as the linear-model Phase 1 reference.** "
                      "The historical screen did not justify a Ridge submission.")
    return f"""# L3 — Ridge regression for next-session intraday return

{recommendation}

Against L2-B, L3-A changed the equal-year official score by **{a.delta:+.8f}** and won
{a.years_won} of three years. L3-B changed it by **{b.delta:+.8f}** and won {b.years_won} years.
The reused 2023–2025 folds provide exploratory evidence.

## Design

The conditional expected official contribution for binary action a is
(2a − 1) E[r | X]. Predicting up when E[r | X] is positive maximizes that expectation.
Under unrestricted squared-error minimization the population optimum is E[r | X].
Ridge approximates it with a regularized linear function. Linear restrictions, shrinkage,
finite samples, and distribution shift can separate the fitted model from that optimum.

The experiment fits one Ridge per fold with alpha 1, an intercept, solver auto, and uniform
observation weights. The target is the adjusted open-to-close decimal return, including flats.
Preprocessing is fitted without return weights on each fold's original 100 training stocks.
Each state exactly matches the saved L1 and L2 state. Every comparison uses identical
50-stock evaluation keys and the shared scorer.

L3-A predicts up strictly above zero. L3-B uses +delta after a negative eligible TAIEX
return, −delta after a nonnegative one, and zero if missing. Delta is 0.05 times the
training fold's mean absolute return. The earlier weighted classification identity gives
2p_w(X) − 1 = E[r | X] / E[|r| | X]. The 0.475/0.525 offsets correspond to ±0.05
on its left side. Substituting an unconditional training mean for the conditional expected
magnitude is an approximation. L3-B is an empirical extension. Comparisons between
market-conditioned Ridge and logistic rules also change the output scale and cutoff.

## Official score and stability

{markdown(scores)}

{markdown(means)}

Spearman here ranks the continuous outputs within each trading date. Market direction is
the daily mean signed position times the daily mean realized return. Stock selection is the
remaining daily official score. Period aggregates weight dates by their observation counts.
See [daily](daily_scores.csv), [monthly](monthly_scores.csv), [four-month](four_month_scores.csv)
and [stock-date scores and decisions](validation_predictions.csv).

{markdown(equal)}

The [paired intervals](paired_uncertainty.csv) use 3,000 circular bootstrap samples with
five-trading-day blocks drawn within each year, retaining every stock from a sampled day.
The [historical screen](phase1_screen.csv) used the following rule before Phase 1 verification.
{SELECTION_POLICY}

## Numerical return estimation

{markdown(reg)}

Columns ending in pct use percentage return units. The full [regression table](regression_metrics.csv)
uses decimal returns, with MSE in squared decimal return units and quantiles for both actual
and predicted returns. Baseline R² is 1 − model MSE / baseline MSE, against either an all-zero
prediction or the mean from the corresponding Top100 training fold. Training rows summarize
in-sample fit. The [metric gaps](regression_gaps.csv) compare validation with each training universe.
Validation R² exceeded zero in {int(validation['r2_vs_zero'].gt(0).sum())} of three years against the
zero prediction and in {int(validation['r2_vs_training_mean'].gt(0).sum())} years against the
training-mean prediction. Direction AUC uses the continuous predicted returns as scores.

{markdown(tables["runtime"])}

Fit time includes preprocessing and Ridge fitting. Inference time includes preprocessing
transformation and predicting all validation returns.

## Magnitude and residuals

The quintiles rank absolute predicted return within each fold without splitting exact ties.
The highest quintile earned a higher fixed-rule score than the lowest in {quintiles_top_better}
of three years. The [quintile table](prediction_magnitude_quintiles.csv) records mean predicted and
realized returns, accuracy, absolute realized return, and both within-group score and group
contribution to the fold score.

{markdown(res[["fold", "residual_p01", "residual_p50", "residual_p99",
               "top_realized_1pct_squared_error_share", "top_realized_5pct_squared_error_share",
               "top_realized_10pct_squared_error_share"]])}

The [residual diagnostics](residual_diagnostics.csv) use realized minus predicted return.
Tail error shares quantify squared-loss concentration among large realized returns.

## Coefficients

{markdown(numeric[["feature", "2023", "2024", "2025", "mean_abs", "sign_flip"]])}

{markdown(coef.loc[coef["family"].eq("intercept"), ["fold", "L3_return_coef"]])}

For every validation row, the fitted intercept plus the encoded coefficient products recovered
the predicted return. The intercept represents the return at an all-zero encoded vector.
Actual average predictions are in the regression table. The [coefficient file](coefficients.csv)
compares signs and signed coefficient shares, normalized by each model's sum of absolute
encoded coefficients. This makes relative weights comparable without equating Ridge return
units with logistic log-odds. See the [full stability table](coefficient_stability.csv) and
[numerical, stock-ID and weekday contributions](feature_family_contributions.csv).
Coefficients describe conditional associations.

## TAIEX decisions

{markdown(taiex)}

The [regime-specific breakdown](threshold_sensitivity.csv) includes up rates, threshold distances,
direction switches, and their incremental signed returns. Summed switch contributions
reproduced the official score difference, including its market-direction and stock-selection parts.

## Earlier Phase 1 score verification

{markdown(tables["phase1_reference_audit"])}

All claimed historical means matched the repository archives. The local replay checked the
saved files' hashes and scored each existing submission on the exact 4,200 keys with archived
adjusted intraday returns. Competition-server receipts were unavailable. This verification
ran after the historical selection and covered earlier submissions only.
January–May 2026 and Phase 1 outcomes did not enter fitting or selection.

## Reproduction

Run from the repository root with a fresh output directory.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.ridge_regression.l3.train_l3 --output-dir members/zhenci/ridge_regression/l3/reproduction
```

[Historical manifest](summary.json) · [Phase 1 audit provenance](phase1_reference_provenance.json)
"""
