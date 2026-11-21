"""Historical score, probability, and coefficient diagnostics for L1."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from shared.data_loader import KEYS
from shared.evaluation import cv_score, evaluate_predictions
from shared.preprocessing import NUMERIC_FEATURES
from shared.splits import FOLDS, SEED

L1_ARMS = ("L1-A", "L1-B")
REFERENCES = ("E1", "E3", "E20", "Previous direction")
ARMS = (*L1_ARMS, *REFERENCES)
DRAWS, BLOCK_DAYS = 3000, 5
SELECTION_POLICY = ("Positive equal-year difference to E3, wins in at least two years, "
                    "and worst four-month score at least E3's. If both qualify, "
                    "higher equal-year score wins with L1-A breaking ties.")


def probability_metrics(y, probability, calls, constant):
    p = np.asarray(probability, dtype=float)
    return {
        "n": len(y), "bce": log_loss(y, p, labels=[0, 1]), "auc": roc_auc_score(y, p),
        "brier": np.mean((p - y) ** 2), "accuracy": np.mean(calls == y),
        "predicted_up_rate": np.mean(calls), "actual_up_rate": np.mean(y),
        "p_mean": p.mean(), "p_sd": p.std(),
        **{f"p{q:02}": np.quantile(p, q / 100) for q in (1, 10, 50, 90, 99)},
        "constant_p": constant, "constant_auc": 0.5,
        "constant_bce": log_loss(y, np.full(len(y), constant), labels=[0, 1]),
        "constant_brier": np.mean((constant - y) ** 2),
    }


def score_arm(fold, arm, valid, returns, p, calls):
    signed = 2 * calls - 1
    frame = valid[KEYS].assign(fold=fold, arm=arm, p_up=p, predicted_y=calls,
                               actual_up=valid["y"], realized_return=returns,
                               contribution=signed * returns, signed=signed)
    official = evaluate_predictions(valid, frame[KEYS + ["predicted_y"]])
    grouped = frame.groupby("date", sort=True)
    daily = grouped.agg(n=("contribution", "size"), score_sum=("contribution", "sum"),
                        mean_sign=("signed", "mean"), mean_return=("realized_return", "mean"))
    daily["market_sum"] = daily["mean_sign"] * daily["mean_return"] * daily["n"]
    daily["selection_sum"] = daily["score_sum"] - daily["market_sum"]
    daily["official_score"] = daily["score_sum"] / daily["n"]
    # Pearson correlation of the within-session average ranks is Spearman rho.
    daily["spearman"] = np.nan
    if p is not None:
        ranks = grouped[["p_up", "realized_return"]].rank()
        daily["spearman"] = (ranks.groupby(frame["date"]).corr()
                             .loc[(slice(None), "p_up"), "realized_return"].droplevel(1))
    np.testing.assert_allclose(daily["score_sum"].sum() / daily["n"].sum(),
                               official["official_score"], atol=1e-14, rtol=0)
    row = {"fold": fold, "arm": arm, **official, "predicted_up_rate": calls.mean(),
           "actual_up_rate": valid["y"].mean(), "mean_daily_spearman": daily["spearman"].mean(),
           "spearman_days": int(daily["spearman"].count()),
           "daily_score_sd": daily["official_score"].std(),
           "negative_day_rate": daily["official_score"].lt(0).mean(),
           "market_direction": daily["market_sum"].sum() / len(valid),
           "stock_selection": daily["selection_sum"].sum() / len(valid)}
    return row, frame.drop(columns="signed"), daily.reset_index().assign(fold=fold, arm=arm)


def periods(daily, by):
    group = daily.groupby(["fold", "arm", by], as_index=False)
    out = group.agg(n=("n", "sum"), days=("date", "size"), score_sum=("score_sum", "sum"),
                    market_sum=("market_sum", "sum"), selection_sum=("selection_sum", "sum"),
                    daily_score_sd=("official_score", "std"),
                    negative_day_rate=("official_score", lambda s: s.lt(0).mean()))
    for name, total in (("official_score", "score_sum"), ("market_direction", "market_sum"),
                        ("stock_selection", "selection_sum")):
        out[name] = out[total] / out["n"]
    return out


def bootstrap(daily):
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="score_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"], axis=0).all().all():
        raise ValueError("Paired bootstrap requires identical coverage")
    rng = np.random.default_rng(SEED)
    count_arrays, indices = {}, {}
    for fold in FOLDS:
        n = count_arrays[fold.name] = counts.loc[fold.name, "E3"].to_numpy()
        days = len(n)
        starts = rng.integers(days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
        indices[fold.name] = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days
                              ).reshape(DRAWS, -1)[:, :days]
    results = []

    def record(arm, reference, fold, point, draws):
        low, high = np.quantile(draws, [.025, .975])
        return dict(arm=arm, reference=reference, fold=fold, delta=point, low_95=low, high_95=high)

    for arm in L1_ARMS:
        for reference in REFERENCES:
            points, draws = [], []
            for fold in count_arrays:
                n, index = count_arrays[fold], indices[fold]
                change = (sums[arm] - sums[reference]).loc[fold].to_numpy()
                point = change.sum() / n.sum()
                sampled = change[index].sum(axis=1) / n[index].sum(axis=1)
                results.append(record(arm, reference, fold, point, sampled))
                points.append(point)
                draws.append(sampled)
            results.append(record(arm, reference, "equal_year", np.mean(points), np.mean(draws, axis=0)))
    return pd.DataFrame(results)


def summarize(metrics, blocks):
    scores = metrics.pivot(index="fold", columns="arm", values="official_score")
    summary = []
    for arm in ARMS:
        sub = blocks.loc[blocks["arm"].eq(arm)]
        worst = sub.loc[sub["official_score"].idxmin()]
        summary.append(dict(arm=arm, official_score=cv_score(scores[arm].to_dict()),
                            years_above_e3=int((scores[arm] > scores["E3"]).sum()),
                            worst_four_month_score=worst["official_score"],
                            worst_four_month_period=worst["block"]))
    result = pd.DataFrame(summary).set_index("arm")
    control = result.loc["E3"]
    result["delta_e3"] = result["official_score"] - control["official_score"]
    result["phase1_supported"] = (result.index.isin(L1_ARMS) & result["delta_e3"].gt(0)
                                 & result["years_above_e3"].ge(2)
                                 & result["worst_four_month_score"].ge(control["worst_four_month_score"]))
    eligible = [arm for arm in L1_ARMS if result.at[arm, "phase1_supported"]]
    selected = max(eligible, key=lambda arm: result.at[arm, "official_score"], default=None)
    return result.reset_index(), selected


def coefficient_family(feature):
    return feature.split("=", 1)[0] if "=" in feature else "numeric"


def correlation_pairs(features, fold):
    corr = features[list(NUMERIC_FEATURES)].corr()
    first, second = np.where(np.triu(corr.abs().to_numpy() >= .85, k=1))
    return pd.DataFrame({"fold": fold, "feature_1": corr.index[first], "feature_2": corr.columns[second],
                         "pearson": corr.to_numpy()[first, second]})


def contribution_tables(fold, keys, labels, probability, features, parts, logit, intercept):
    names = features.columns
    groups = pd.Index([coefficient_family(name) for name in names])
    calls = (probability > .5).astype(int)
    distance = np.abs(probability - .5)
    categories = {
        "correct_up": (calls == 1) & (labels == 1), "correct_down": (calls == 0) & (labels == 0),
        "incorrect_up": (calls == 1) & (labels == 0), "incorrect_down": (calls == 0) & (labels == 1),
        "high_confidence": np.ones(len(calls), dtype=bool),
        "near_threshold": np.ones(len(calls), dtype=bool),
    }
    stats = pd.DataFrame({"fold": fold, "feature": names, "family": groups,
                          "mean_contribution": parts.mean(axis=0),
                          "mean_abs_contribution": np.abs(parts).mean(axis=0)})
    families = []
    for group in groups.unique():
        total = parts[:, groups == group].sum(axis=1)
        families.append(dict(fold=fold, family=group, mean_contribution=total.mean(),
                             mean_abs_contribution=np.abs(total).mean()))
    examples = []
    for label, mask in categories.items():
        index = np.flatnonzero(mask)
        if len(index):
            pick = np.argmin(distance[index]) if label == "near_threshold" else np.argmax(distance[index])
            i = index[pick]
            examples.append(pd.DataFrame({
                "fold": fold, "example": label, **keys.iloc[i].to_dict(), "actual_up": labels[i],
                "predicted_y": calls[i], "p_up": probability[i], "intercept": intercept,
                "logit": logit[i], "feature": names, "family": groups,
                "transformed_value": features.iloc[i].to_numpy(), "log_odds_contribution": parts[i],
            }))
    return stats, pd.DataFrame(families), pd.concat(examples, ignore_index=True)


def coefficient_stability(coefficients):
    pivot = coefficients.pivot(index=["feature", "family"], columns="fold", values="coefficient")
    stability = pivot.assign(mean=pivot.mean(axis=1), mean_abs=pivot.abs().mean(axis=1),
                             sd=pivot.std(axis=1, ddof=0), folds_present=pivot.count(axis=1),
                             sign_flip=pivot.lt(0).any(axis=1) & pivot.gt(0).any(axis=1))
    return stability.sort_values("mean_abs", ascending=False).reset_index()
