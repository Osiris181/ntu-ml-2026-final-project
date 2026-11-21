"""Return accuracy, score decomposition, and paired historical inference for L3."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from members.zhenci.logistic_regression.l1.analysis import BLOCK_DAYS, DRAWS, coefficient_family
from shared.evaluation import cv_score
from shared.splits import FOLDS, SEED

ARMS = ("L3-A", "L3-B")
CONTROLS = ("L1-A", "L1-B", "L2-A", "L2-B", "E1", "E3")
REFERENCE = "L2-B"
SELECTION_POLICY = (
    "Compare each arm with L2-B on the same historical keys. Require a positive equal-year "
    "official-score difference, wins in at least two years, and a worst four-month score at "
    "least as high as L2-B's. Prefer the higher equal-year score among eligible arms, with "
    "ties going to L3-A. Use paired intervals to describe uncertainty."
)
COMPARISONS = (*((arm, ref) for arm in ARMS for ref in CONTROLS), ("L3-B", "L3-A"))


def quantiles(values, prefix):
    return {f"{prefix}_mean": values.mean(), f"{prefix}_sd": values.std(),
            **{f"{prefix}_p{q:02}": np.quantile(values, q / 100) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}}


def correlations(predicted, realized):
    frame = pd.DataFrame({"predicted": predicted, "realized": realized})
    return (frame["predicted"].corr(frame["realized"]),
            frame["predicted"].rank().corr(frame["realized"].rank()))


def regression_metrics(fold, split, predicted, realized, train_mean):
    squared = np.square(realized - predicted)
    mse = squared.mean()
    zero_mse = np.square(realized).mean()
    mean_mse = np.square(realized - train_mean).mean()
    pearson, spearman = correlations(predicted, realized)
    return dict(fold=fold, split=split, rows=len(predicted), mse=mse, rmse=np.sqrt(mse),
                mae=np.abs(realized - predicted).mean(), pearson=pearson, spearman=spearman,
                r2_vs_zero=1 - mse / zero_mse if zero_mse else np.nan,
                r2_vs_training_mean=1 - mse / mean_mse if mean_mse else np.nan,
                zero_baseline_mse=zero_mse, training_mean_baseline_mse=mean_mse, training_mean=train_mean,
                **quantiles(predicted, "predicted"), **quantiles(realized, "realized"))


def magnitude_groups(fold, predicted, realized, calls):
    ranked = pd.Series(np.abs(predicted)).rank(method="average", pct=True)
    groups = pd.cut(ranked, bins=np.linspace(0, 1, 6), labels=False, include_lowest=True) + 1
    frame = pd.DataFrame({"quintile": groups, "predicted_return": predicted, "realized_return": realized,
                          "absolute_prediction": np.abs(predicted), "absolute_realized": np.abs(realized),
                          "correct": calls == (realized > 0), "contribution": (2 * calls - 1) * realized})
    out = frame.groupby("quintile", as_index=False).agg(
        observations=("correct", "size"), mean_predicted_return=("predicted_return", "mean"),
        mean_realized_return=("realized_return", "mean"), direction_accuracy=("correct", "mean"),
        mean_absolute_realized_return=("absolute_realized", "mean"),
        min_absolute_predicted_return=("absolute_prediction", "min"),
        max_absolute_predicted_return=("absolute_prediction", "max"),
        official_score=("contribution", "mean"), contribution_sum=("contribution", "sum"))
    return out.assign(fold=fold, contribution_to_fold_score=out["contribution_sum"] / len(realized))


def residuals(fold, predicted, realized):
    residual = realized - predicted
    squared = np.square(residual)
    ranked = pd.Series(np.abs(realized)).rank(method="average", pct=True)
    return dict(fold=fold, **quantiles(residual, "residual"), rmse=np.sqrt(squared.mean()),
                **{f"top_realized_{pct}pct_actual_fraction": ranked.gt(1 - pct / 100).mean()
                   for pct in (1, 5, 10)},
                **{f"top_realized_{pct}pct_squared_error_share":
                   squared[ranked.gt(1 - pct / 100)].sum() / squared.sum() for pct in (1, 5, 10)})


def score_auc(y, scores):
    return roc_auc_score(y, scores)


def threshold_effect(fold, predicted, realized, fixed, dynamic, taiex, daily, mean_magnitude, delta):
    regime = np.where(np.isnan(taiex), "missing", np.where(taiex < 0, "negative", "nonnegative"))
    up, down = (fixed == 0) & (dynamic == 1), (fixed == 1) & (dynamic == 0)
    records = []
    for label in ("all", "negative", "nonnegative", "missing"):
        mask = np.ones(len(taiex), dtype=bool) if label == "all" else regime == label
        n = int(mask.sum())
        if not n:
            continue
        subset = daily if label == "all" else daily.loc[daily["regime"].eq(label)]
        summed = subset.groupby("arm")[["score_sum", "market_sum", "selection_sum"]].sum()
        score, market, stock = (summed.loc["L3-B"] - summed.loc["L3-A"]) / n
        up_sum, down_sum = 2 * realized[mask & up].sum(), -2 * realized[mask & down].sum()
        np.testing.assert_allclose([score, market + stock], (up_sum + down_sum) / n, atol=1e-14, rtol=0)
        changed = up[mask] | down[mask]
        records.append(dict(fold=fold, regime=label, mean_abs_training_return=mean_magnitude,
                            delta=delta, observations=n, changed=int(changed.sum()), changed_fraction=changed.mean(),
                            down_to_up=int(up[mask].sum()), up_to_down=int(down[mask].sum()),
                            fixed_up_rate=fixed[mask].mean(), taiex_up_rate=dynamic[mask].mean(),
                            near_zero_rate=(np.abs(predicted[mask]) <= delta).mean(),
                            down_to_up_contribution_sum=up_sum, up_to_down_contribution_sum=down_sum,
                            incremental_contribution_sum=up_sum + down_sum, official_score_delta=score,
                            market_direction_delta=market, stock_selection_delta=stock))
    return pd.DataFrame(records)


def coefficient_tables(fold, ridge, l1, l2, features, predicted):
    contributions = features.to_numpy(dtype=float) * ridge.coef_
    np.testing.assert_allclose(ridge.intercept_ + contributions.sum(axis=1), predicted, atol=1e-14, rtol=0)
    names = features.columns.tolist()
    if any(model.feature_names_in_.tolist() != names for model in (ridge, l1, l2)):
        raise ValueError("L3 coefficient comparisons require identical encoded features")
    family = pd.Index([coefficient_family(name) for name in names])
    coeff = pd.DataFrame({"fold": fold, "feature": names, "family": family, "L3_return_coef": ridge.coef_})
    for name, values in (("L1", l1.coef_[0]), ("L2", l2.coef_[0]), ("L3", ridge.coef_)):
        coeff[f"{name}_relative_signed"] = values / np.abs(values).sum()
    for name in ("L1", "L2"):
        coeff[f"{name}_same_sign"] = np.sign(coeff[f"{name}_relative_signed"]) == np.sign(ridge.coef_)
    coeff["abs_L3_return_coef"] = coeff["L3_return_coef"].abs()
    coeff["validation_mean_contribution"] = contributions.mean(axis=0)
    coeff["validation_mean_abs_contribution"] = np.abs(contributions).mean(axis=0)
    families = []
    for group in family.unique():
        summed = contributions[:, family == group].sum(axis=1)
        families.append(dict(fold=fold, family=group, mean_contribution=summed.mean(),
                             mean_abs_contribution=np.abs(summed).mean()))
    intercept = pd.concat([coeff, pd.DataFrame([dict(fold=fold, feature="(intercept)", family="intercept",
        L3_return_coef=ridge.intercept_, abs_L3_return_coef=abs(ridge.intercept_))])], ignore_index=True)
    return intercept, pd.DataFrame(families)


def paired_intervals(daily):
    scores = daily.pivot(index=["fold", "date"], columns="arm", values="score_sum")
    sizes = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if scores.isna().any().any() or not sizes.eq(sizes[REFERENCE], axis=0).all().all():
        raise ValueError("Paired intervals require identical daily coverage")
    rng = np.random.default_rng(SEED)
    samples = {}
    for fold in FOLDS:
        n = sizes.loc[fold.name, REFERENCE].to_numpy()
        days = len(n)
        starts = rng.integers(days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
        index = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days).reshape(DRAWS, -1)[:, :days]
        samples[fold.name] = (n, index)

    def row(arm, control, period, delta, draws):
        low, high = np.quantile(draws, [.025, .975])
        return dict(arm=arm, control=control, period=period, delta=delta, low_95=low, high_95=high)

    records = []
    for arm, control in COMPARISONS:
        points, draws = [], []
        for fold, (n, index) in samples.items():
            change = (scores[arm] - scores[control]).loc[fold].to_numpy()
            points.append(change.sum() / n.sum())
            draws.append(change[index].sum(axis=1) / n[index].sum(axis=1))
            records.append(row(arm, control, fold, points[-1], draws[-1]))
        records.append(row(arm, control, "equal_year", np.mean(points), np.mean(draws, axis=0)))
    return pd.DataFrame(records)


def summarize(metrics, blocks, intervals):
    yearly = metrics.pivot(index="fold", columns="arm", values="official_score")
    worst = blocks.loc[blocks.groupby("arm")["official_score"].idxmin()].set_index("arm")
    overall = pd.DataFrame([dict(arm=arm, official_score=cv_score(yearly[arm].to_dict()),
                                 worst_four_month_score=worst.at[arm, "official_score"],
                                 worst_block=worst.at[arm, "block"]) for arm in yearly]).set_index("arm")
    rows = []
    for arm in ARMS:
        difference = overall.at[arm, "official_score"] - overall.at[REFERENCE, "official_score"]
        wins = int(yearly[arm].gt(yearly[REFERENCE]).sum())
        worst_delta = overall.at[arm, "worst_four_month_score"] - overall.at[REFERENCE, "worst_four_month_score"]
        ci = intervals.loc[intervals["arm"].eq(arm) & intervals["control"].eq(REFERENCE)
                           & intervals["period"].eq("equal_year")].iloc[0]
        rows.append(dict(arm=arm, control=REFERENCE, delta=difference, years_won=wins, worst_delta=worst_delta,
                         low_95=ci["low_95"], high_95=ci["high_95"],
                         qualifies=difference > 0 and wins >= 2 and worst_delta >= 0))
    screen = pd.DataFrame(rows)
    eligible = [row.arm for row in screen.itertuples() if row.qualifies]
    selected = max(eligible, key=lambda arm: overall.at[arm, "official_score"], default=None)
    return overall.reset_index(), screen, selected
