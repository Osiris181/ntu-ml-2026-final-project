"""Render L2 findings from the saved evaluation tables."""

from members.zhenci.logistic_regression.l1.report import markdown
from members.zhenci.logistic_regression.l2.diagnostics import SELECTION_POLICY


def render_report(tables, candidate):
    fold = tables["fold_metrics"]
    years = fold.pivot(index="arm", columns="fold", values="official_score").reset_index()
    scores = years.merge(tables["equal_year_scores"], on="arm")
    means = fold.groupby("arm", as_index=False)[["accuracy", "direction_auc", "mean_daily_spearman",
        "predicted_up_rate", "actual_up_rate", "market_direction", "stock_selection"]].mean()
    intervals = tables["paired_uncertainty"]
    core = intervals.loc[intervals["arm"].str.startswith("L2") &
                         intervals["control"].isin(["L1-A", "L1-B", "L2-A"])]
    weights = tables["sample_weights"]
    weight_summary = weights[["fold", "rows", "normalizer", "zero_fraction", "p99", "max", "ess_fraction",
                              "top_1pct_weight_share", "top_10pct_weight_share", "weighted_up_share"]]
    losses = tables["loss_and_score_distributions"]
    matched = losses.loc[(losses["split"].eq("train")) | losses["arm"].isin(["L1-A", "L2-A"])]
    loss_table = matched.pivot(index=["fold", "split"], columns="arm", values="return_weighted_bce")
    loss_table["L1"] = loss_table["L1"].fillna(loss_table["L1-A"])
    loss_table["L2"] = loss_table["L2"].fillna(loss_table["L2-A"])
    loss_table = loss_table[["L1", "L2"]].assign(delta=loss_table["L2"] - loss_table["L1"]).reset_index()
    rules = tables["threshold_sensitivity"]
    rules = rules.loc[rules["regime"].eq("all"), ["period", "model", "changed", "down_to_up", "up_to_down",
        "fraction_in_0475_to_0525", "official_score_delta", "market_direction_delta", "stock_selection_delta"]]
    phase1 = tables["phase1_l1_threshold_sensitivity"]
    phase_row = phase1.loc[phase1["regime"].eq("all")].iloc[0]
    coef = tables["coefficients"]
    intercept = coef.loc[coef["family"].eq("intercept"), ["fold", "L1", "L2", "delta"]]
    numerical = coef.loc[coef["family"].eq("numeric")].copy()
    numerical["absolute_change"] = numerical["delta"].abs()
    feature_changes = numerical.groupby("feature", as_index=False).agg(
        mean_L1=("L1", "mean"), mean_L2=("L2", "mean"), mean_absolute_change=("absolute_change", "mean"))
    feature_changes = feature_changes.nlargest(6, "mean_absolute_change")
    family = tables["family_contributions"].groupby(["model", "family"], as_index=False)[
        ["mean_contribution", "mean_abs_contribution"]].mean()
    first, second = tuple(tables["phase1_screen"].itertuples())
    result = (f"**{candidate} qualifies for one exploratory Phase 1 evaluation.**"
              if candidate else "**Retain L1-B as the Logistic Regression submission reference. "
              "No L2 rule met the Phase 1 screen.**")
    return f"""# L2 — Return-weighted Logistic Regression

{result}

Return weighting changed the equal-year official score by **{first.delta:+.8f}** under the fixed
threshold and **{second.delta:+.8f}** under the TAIEX rule. The two arms beat their matched
L1 controls in {first.years_won} and {second.years_won} of the three years, respectively.
These reused 2023–2025 folds provide exploratory evidence.

## Controlled change and mathematical target

The archived L1 models and manifests confirm L2-regularized logistic regression with C=1,
the lbfgs solver, 1,000 maximum iterations, tolerance 1e-4, an intercept, and seed 42.
The installed scikit-learn API represents pure L2 with l1_ratio=0. L2 adds only sample weights.
The shared, unweighted F0 preprocessing matched each archived L1 fold state exactly.
The same 100-stock training windows, 50-stock evaluation keys, labels, and official scorer were used.
E1, E3 and E17-B were read from archived predictions or reconstructed with their saved rule.

The [earlier E2 XGBoost weighting implementation](../../../xgboost/e2/train_e2_weighted.py)
used the same decimal adjusted intraday return magnitude divided by the training-fold mean.
L2 reuses that formula, keeps every flat row with zero weight, and applies no clipping or
class balancing.

The conditional score of a binary action `a` is `(2a − 1) E[r | X]`. For any `X` with positive
expected return magnitude, the unrestricted weighted-BCE optimum is
`p_w(X) = E[|r|y | X] / E[|r| | X]`. Because `|r|(2y − 1) = r`, it follows that
`2p_w(X) − 1 = E[r | X] / E[|r| | X]`. A strict cutoff of 0.5 therefore implements the sign
decision at that population optimum. The fitted regularized linear model is a restricted
approximation. Its output describes the return-weighted target. L2-B's thresholds of
0.525, 0.475, and 0.5 for negative, nonnegative, and missing eligible TAIEX returns are an
empirical comparison.

## Historical evaluation

{markdown(scores)}

{markdown(means)}

Market direction uses the daily mean signed position times the daily mean realized return.
Stock selection is the residual. Period aggregates weight trading days by their observation counts.
See [daily](daily_scores.csv), [monthly](monthly_scores.csv) and [four-month scores](four_month_scores.csv).
The [validation prediction file](validation_predictions.csv) distinguishes weighted scores from
ordinary direction probabilities.

{markdown(core)}

Intervals use 3,000 paired circular bootstrap samples of five-trading-day blocks within years.
The sampling unit keeps every stock from a trading day together. See the
[full paired comparisons](paired_uncertainty.csv) and the [candidate screen](phase1_screen.csv).

## Weights and loss

{markdown(weight_summary)}

The largest-return 1% received {weights["top_1pct_weight_share"].min():.1%} to
{weights["top_1pct_weight_share"].max():.1%} of training weight. The top 10% received
{weights["top_10pct_weight_share"].min():.1%} to {weights["top_10pct_weight_share"].max():.1%}.
The effective sample size was {weights["ess_fraction"].min():.1%} to
{weights["ess_fraction"].max():.1%} of the row count. This measures concentration of the loss weights.
Whether the largest returns caused the performance differences remains open.
The [weight table](sample_weights.csv) includes means, standard deviations, quantiles,
zero counts, label totals, and absolute ESS.

{markdown(loss_table)}

Both models use the same absolute-return diagnostic weights, scaled by each training fold's
mean absolute return on both splits. Validation weights are applied only when calculating loss.
See [loss and score distributions](loss_and_score_distributions.csv) for all validation arms,
score quantiles, and ordinary BCE and Brier comparisons. The official score governs the recommendation.

{markdown(tables["runtime"])}

## Coefficients and prediction contributions

{markdown(intercept)}

{markdown(feature_changes)}

{markdown(family)}

The contribution audit multiplies every encoded feature by its fitted coefficient and verifies the
intercept plus the sum against the model decision function and score for every validation row.
Family rows are equal-year means of the signed and absolute validation log-odds contributions.
The [coefficient table](coefficients.csv) includes signs, magnitudes and matched changes.
See [cross-fold stability](coefficient_stability.csv) and [individual feature contributions](feature_contributions.csv).
The estimates describe conditional associations.

## Threshold sensitivity and Phase 1

{markdown(rules)}

The [full sensitivity table](threshold_sensitivity.csv) includes yearly and regime-specific counts,
quantiles and up rates. Summed changes from switched predictions recover the official-score
differences, including the separate market-direction and stock-selection changes.

In the saved L1 Phase 1 predictions, **{phase_row['fraction_in_0475_to_0525']:.1%}** of scores
fell between 0.475 and 0.525. The TAIEX rule changed **{phase_row['changed']:,}** calls
({phase_row['changed_fraction']:.1%}), including {phase_row['down_to_up']:,} down-to-up and
{phase_row['up_to_down']:,} up-to-down. The user-reported aggregate score increased by
**{phase_row['reported_taiex_score'] - phase_row['reported_fixed_score']:+.6f}**. Concentration
near the threshold gave the rule room to change calls. Identifying which switched predictions
earned the gain would require row-level scoring evidence. This diagnostic used
[saved probabilities and calls](phase1_l1_threshold_sensitivity.csv) and
[reported aggregate scores](phase1_reported_scores.csv).

The screen was set before examining L2 results. {SELECTION_POLICY}
January–May 2026, Phase 1 and October outcomes remained outside model fit and selection.

## Reproduction

From the repository root, pass a fresh output directory.

```powershell
uv run --with scikit-learn==1.9.1 python -m members.zhenci.logistic_regression.l2.train_l2 --output-dir members/zhenci/logistic_regression/l2/reproduction
```

The [run manifest](summary.json) records the frozen configuration and input and model digests.
"""
