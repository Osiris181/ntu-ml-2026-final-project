"""Return weights, time-block inference, and threshold diagnostics for L2."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from members.zhenci.logistic_regression.l1.analysis import BLOCK_DAYS, DRAWS
from members.zhenci.xgboost.e2.train_e2_weighted import training_return_weights
from shared.evaluation import CLOSE, OPEN, cv_score
from shared.splits import FOLDS, SEED

MATCHED = {"L2-A": "L1-A", "L2-B": "L1-B"}
XGB_REFERENCES = ("E1", "E3", "E17-B")
COMPARISONS = (*MATCHED.items(), ("L2-B", "L2-A"),
               *((arm, ref) for arm in MATCHED for ref in XGB_REFERENCES))
SELECTION_POLICY = ("Positive equal-year difference against the matched L1 rule, at least two winning "
                    "years, and no deterioration in the worst four-month score. Prefer the higher official "
                    "mean among eligible arms, with a tie going to L2-A. Report paired uncertainty "
                    "to qualify the strength of exploratory evidence.")


def realized_returns(rows):
    returns = ((rows[CLOSE] - rows[OPEN]) / rows[OPEN]).to_numpy()
    if not np.isfinite(returns).all() or not np.array_equal(returns > 0, rows["y"].to_numpy()):
        raise ValueError("Expected finite adjusted intraday returns with aligned direction labels")
    return returns


def return_weights(rows):
    returns = realized_returns(rows)
    mean_magnitude = float(np.abs(returns).mean())
    weights = training_return_weights(rows)
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.mean(), 1.0, atol=1e-12, rtol=0)
            or not np.array_equal(weights == 0, returns == 0)):
        raise ValueError("Return weights must be finite, nonnegative, mean one, and zero exactly on flat rows")
    return weights, mean_magnitude


def describe_weights(fold, y, weights, mean_magnitude):
    total = weights.sum()
    sorted_weights = np.sort(weights)
    ess = total**2 / np.square(weights).sum()
    return {
        "fold": fold, "rows": len(y), "normalizer": mean_magnitude,
        "mean": weights.mean(), "std": weights.std(), "min": weights.min(), "max": weights.max(),
        **{f"p{q:02}": np.quantile(weights, q / 100) for q in (25, 50, 75, 90, 95, 99)},
        "zero_count": int((weights == 0).sum()), "zero_fraction": (weights == 0).mean(),
        "up_weight_sum": weights[y == 1].sum(), "down_weight_sum": weights[y == 0].sum(),
        "weighted_up_share": weights[y == 1].sum() / total, "unweighted_up_share": y.mean(),
        **{f"top_{q}pct_weight_share": sorted_weights[-int(np.ceil(len(y) * q / 100)):].sum() / total
           for q in (1, 5, 10)},
        "ess": ess, "ess_fraction": ess / len(y),
    }


def score_distribution(scores):
    return {
        "score_mean": scores.mean(), "score_sd": scores.std(),
        **{f"p{q:02}": np.quantile(scores, q / 100) for q in (1, 10, 50, 90, 99)},
        "fraction_in_0475_to_0525": ((scores >= .475) & (scores <= .525)).mean(),
    }


def loss_metrics(fold, arm, split, y, score, weights):
    return {
        "fold": fold, "arm": arm, "split": split, "rows": len(y),
        "return_weighted_bce": log_loss(y, score, labels=[0, 1], sample_weight=weights),
        "ordinary_bce": log_loss(y, score, labels=[0, 1]),
        "ordinary_brier": np.square(y - score).mean(),
        "direction_auc": roc_auc_score(y, score), **score_distribution(score),
    }


def decision_diagnostics(period, model, scores, fixed, dynamic, taiex, returns=None, daily=None):
    up = (fixed == 0) & (dynamic == 1)
    down = (fixed == 1) & (dynamic == 0)
    regimes = np.where(np.isnan(taiex), "missing", np.where(taiex < 0, "negative", "nonnegative"))
    records = []
    for name in ("all", "negative", "nonnegative", "missing"):
        mask = np.ones(len(scores), dtype=bool) if name == "all" else regimes == name
        n = int(mask.sum())
        if not n:
            continue
        row = dict(period=period, model=model, regime=name, rows=n,
                   changed=int((up[mask] | down[mask]).sum()), changed_fraction=(up[mask] | down[mask]).mean(),
                   down_to_up=int(up[mask].sum()), up_to_down=int(down[mask].sum()),
                   fixed_up_rate=fixed[mask].mean(), dynamic_up_rate=dynamic[mask].mean(),
                   **score_distribution(scores[mask]))
        if returns is not None:
            up_sum = 2 * returns[mask & up].sum()
            down_sum = -2 * returns[mask & down].sum()
            days = daily if name == "all" else daily.loc[daily["regime"].eq(name)]
            delta = days.pivot(index="date", columns="arm", values=["score_sum", "market_sum", "selection_sum"])
            score, market, stock = (float((delta[key][f"{model}-B"] - delta[key][f"{model}-A"]).sum() / n)
                                    for key in ("score_sum", "market_sum", "selection_sum"))
            np.testing.assert_allclose([score, market + stock], (up_sum + down_sum) / n, atol=1e-14, rtol=0)
            row.update(incremental_sum=up_sum + down_sum, down_to_up_contribution_sum=up_sum,
                       up_to_down_contribution_sum=down_sum, official_score_delta=score,
                       market_direction_delta=market, stock_selection_delta=stock)
        records.append(row)
    return pd.DataFrame(records)


def paired_intervals(daily):
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="score_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"], axis=0).all().all():
        raise ValueError("Paired L2 intervals require identical daily coverage")
    rng = np.random.default_rng(SEED)
    samples = {}
    for fold in FOLDS:
        n = counts.loc[fold.name, "E3"].to_numpy()
        days = len(n)
        starts = rng.integers(days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
        index = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days).reshape(DRAWS, -1)[:, :days]
        samples[fold.name] = (n, index)

    def record(arm, control, years, point, draws):
        low, high = np.quantile(draws, [.025, .975])
        return dict(arm=arm, control=control, period=years, delta=point, low_95=low, high_95=high)

    results = []
    for arm, control in COMPARISONS:
        points, draws = [], []
        for fold, (n, index) in samples.items():
            delta = (sums[arm] - sums[control]).loc[fold].to_numpy()
            points.append(delta.sum() / n.sum())
            draws.append(delta[index].sum(axis=1) / n[index].sum(axis=1))
            results.append(record(arm, control, fold, points[-1], draws[-1]))
        results.append(record(arm, control, "equal_year", np.mean(points), np.mean(draws, axis=0)))
    return pd.DataFrame(results)


def summarize(metrics, blocks, intervals):
    scores = metrics.pivot(index="fold", columns="arm", values="official_score")
    worst = blocks.loc[blocks.groupby("arm")["official_score"].idxmin()].set_index("arm")
    overall = pd.DataFrame([
        dict(arm=arm, official_score=cv_score(scores[arm].to_dict()),
             worst_four_month_score=worst.at[arm, "official_score"], worst_block=worst.at[arm, "block"])
        for arm in scores
    ]).set_index("arm")
    results = []
    for arm, control in MATCHED.items():
        delta = overall.at[arm, "official_score"] - overall.at[control, "official_score"]
        wins = int(scores[arm].gt(scores[control]).sum())
        worst_delta = overall.at[arm, "worst_four_month_score"] - overall.at[control, "worst_four_month_score"]
        ci = intervals.loc[intervals["arm"].eq(arm) & intervals["control"].eq(control)
                           & intervals["period"].eq("equal_year")].iloc[0]
        results.append(dict(arm=arm, control=control, delta=delta, years_won=wins,
                            worst_four_month_delta=worst_delta, low_95=ci["low_95"], high_95=ci["high_95"],
                            qualifies=delta > 0 and wins >= 2 and worst_delta >= 0))
    selection = pd.DataFrame(results)
    eligible = [row.arm for row in selection.itertuples() if row.qualifies]
    selected = max(eligible, key=lambda arm: overall.at[arm, "official_score"], default=None)
    return overall.reset_index(), selection, selected
