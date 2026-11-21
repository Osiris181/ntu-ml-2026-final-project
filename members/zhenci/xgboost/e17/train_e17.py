"""Run fixed daily class balancing on 2023–2025 with archived E1/E3/E16-A controls."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci.xgboost.e1.save_e1 import _sha256
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import (
    DEFAULT_E1_PREDICTIONS, DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY,
    align_saved_probabilities, load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from members.zhenci.xgboost.e11.train_e11 import training_returns
from members.zhenci.xgboost.e15.train_e15 import BLOCK_DAYS, DRAWS, daily_metrics, interval
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import FOLDS, SEED, evaluation_keys, split_market

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data"
F0 = DATA / "processed/f0.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
E16 = ROOT / "members/zhenci/xgboost/e16/outputs/validation_predictions.csv"
E16_SUMMARY = E16.with_name("summary.json")
STRENGTHS = {"E3": 0.0, "E17-A": 0.5, "E17-B": 1.0}
COMPARISONS = (
    ("E17-A", "E3"), ("E17-B", "E3"), ("E17-B", "E17-A"),
    ("E17-A-050", "E1"), ("E17-B-050", "E1"),
    ("E17-A-050", "E16-A"), ("E17-B-050", "E16-A"),
)


class DMatrixClassifier(xgb.XGBClassifier):
    """Use the E3 matrix construction path for every sklearn-wrapper fit."""

    def _create_dmatrix(self, ref, **kwargs):
        return xgb.DMatrix(**kwargs, nthread=self.n_jobs)


def sample_weights(rows: pd.DataFrame, strength: float) -> pd.Series:
    if strength not in STRENGTHS.values() or rows.empty or not rows["y"].isin([0, 1]).all():
        raise ValueError("Expected binary training labels and lambda in {0, 0.5, 1}")
    if rows["date"].isna().any():
        raise ValueError("Weight construction requires a date for each eligible training row")
    by_date = rows.groupby("date")["y"]
    size = by_date.transform("size")
    up = by_date.transform("sum")
    class_n = up.where(rows["y"].eq(1), size - up)
    balanced = (size / (2 * class_n)).where(up.ne(0) & up.ne(size), 1.0)
    weights = (1 - strength) + strength * balanced
    if not np.allclose(weights.groupby(rows["date"]).sum(), by_date.size(), atol=1e-10, rtol=0):
        raise ValueError("Date totals changed after weighting")
    return weights.rename("weight")


def fit_classifier(recipe: dict, features: pd.DataFrame, rows: pd.DataFrame,
                   weights: pd.Series) -> DMatrixClassifier:
    params = dict(recipe["params"])
    threads = params.pop("nthread")
    model = DMatrixClassifier(**params, n_estimators=recipe["num_boost_round"],
                              random_state=SEED, n_jobs=threads)
    model.fit(features, rows["y"].to_numpy(), sample_weight=weights.to_numpy())
    return model


def weight_diagnostics(rows: pd.DataFrame, returns: np.ndarray, weights: pd.Series,
                       fold: str, arm: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    observations = rows[KEYS + ["y"]].assign(weight=weights, realized_return=returns)
    observations["abs_return"] = observations["realized_return"].abs()
    observations["weighted_up"] = observations["weight"] * observations["y"]
    by_date = observations.groupby("date")
    daily = by_date.agg(n=("y", "size"), up=("y", "sum"), total_weight=("weight", "sum"),
                        up_weight=("weighted_up", "sum"), max_weight=("weight", "max"))
    daily = daily.assign(fold=fold, arm=arm, down=daily["n"] - daily["up"],
                         down_weight=daily["total_weight"] - daily["up_weight"],
                         up_rate=daily["up"] / daily["n"],
                         weighted_up_rate=daily["up_weight"] / daily["total_weight"])
    daily["single_class"] = daily["up"].eq(0) | daily["down"].eq(0)
    daily["imbalance"] = (daily["up_rate"] - .5).abs()
    daily["weighted_imbalance"] = (daily["weighted_up_rate"] - .5).abs()
    observations["daily_up_rate"] = observations["date"].map(daily["up_rate"])
    observations["abs_return_percentile"] = by_date["abs_return"].rank(pct=True)
    high = observations["weight"].gt(2)
    others = observations.loc[~high]
    extreme = observations.loc[high].assign(fold=fold, arm=arm)
    summary = {"fold": fold, "arm": arm, "lambda": STRENGTHS[arm], "rows": len(rows),
               "days": len(daily), "single_class_days": int(daily["single_class"].sum()),
               "absolute_up_rate": rows["y"].mean(), "weighted_up_rate": observations["weighted_up"].sum() / weights.sum(),
               "min": weights.min(), "median": weights.median(), "p95": weights.quantile(.95),
               "p99": weights.quantile(.99), "max": weights.max(),
               "effective_sample_size": weights.sum()**2 / (weights**2).sum(),
               "above_2": int(high.sum()), "above_5": int(weights.gt(5).sum()), "above_10": int(weights.gt(10).sum()),
               "above_2_dates": extreme["date"].nunique(), "above_2_up_share": extreme["y"].mean(),
               "above_2_median_abs_return": extreme["abs_return"].median(),
               "other_median_abs_return": others["abs_return"].median(),
               "above_2_mean_within_day_abs_return_percentile": extreme["abs_return_percentile"].mean(),
               "other_mean_within_day_abs_return_percentile": others["abs_return_percentile"].mean(),
               "daily_imbalance_median": daily["imbalance"].median(), "daily_imbalance_p95": daily["imbalance"].quantile(.95),
               "weighted_daily_imbalance_median": daily["weighted_imbalance"].median(),
               "weighted_daily_imbalance_p95": daily["weighted_imbalance"].quantile(.95),
               "max_daily_weight_total_error": (daily["total_weight"] - daily["n"]).abs().max()}
    return daily.reset_index(), summary, extreme


def paired_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="contribution_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"], axis=0).all().all():
        raise ValueError("Paired scores require identical daily keys")
    rng = np.random.default_rng(SEED)
    results = []
    for arm, reference in COMPARISONS:
        points, bootstrap = [], []
        for fold in FOLDS:
            delta = (sums.loc[fold.name, arm] - sums.loc[fold.name, reference]).to_numpy()
            n = counts.loc[fold.name, arm].to_numpy()
            days = len(n)
            starts = rng.integers(0, days, size=(DRAWS, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
            indices = ((starts[..., None] + np.arange(BLOCK_DAYS)) % days).reshape(DRAWS, -1)[:, :days]
            point = delta.sum() / n.sum()
            draws = delta[indices].sum(axis=1) / n[indices].sum(axis=1)
            results.append(interval(arm, reference, fold.name, point, draws))
            points.append(point)
            bootstrap.append(draws)
        results.append(interval(arm, reference, "equal_year", np.mean(points), np.mean(bootstrap, axis=0)))
    return pd.DataFrame(results)


def evaluate(arm: str, valid: pd.DataFrame, p: np.ndarray, decision: np.ndarray,
             returns: np.ndarray, fold: str, target: str = "absolute_direction") -> tuple[dict, pd.DataFrame]:
    if p.shape != (len(valid),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"Invalid validation probabilities for {arm}")
    frame = valid[KEYS].assign(fold=fold, arm=arm, p_up=p, probability_target=target,
                               predicted_y=decision, actual_up=valid["y"], realized_return=returns,
                               contribution=(2 * decision - 1) * returns)
    official = evaluate_predictions(valid, frame[KEYS + ["predicted_y"]])
    y = valid["y"].to_numpy()
    q = np.clip(p, 1e-15, 1 - 1e-15)
    # E16-A estimates relative outperformance, so direction calibration losses do not apply.
    bce = -np.mean(np.where(y == 1, np.log(q), np.log1p(-q))) if target == "absolute_direction" else np.nan
    brier = np.mean((p - y)**2) if target == "absolute_direction" else np.nan
    metrics = {"fold": fold, "arm": arm, **official, "predicted_up_rate": decision.mean(),
               "direction_bce": bce, "direction_brier": brier, "p_mean": p.mean(), "p_std": p.std(),
               "p10": np.quantile(p, .1), "p50": np.median(p), "p90": np.quantile(p, .9)}
    return metrics, frame


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    target_ids, all_ids = load_metadata(DATA)
    market = load_market(DATA).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    f0 = load_f0_dataset(F0).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3 = load_fixed_rule(DEFAULT_E3_SUMMARY, recipe["config_id"])
    control = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    archived = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, recipe)
    e16 = pd.read_csv(E16, dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    e16 = e16.loc[e16["arm"].eq("E16-A")]
    if not e16["score_target"].eq("relative_performance").all():
        raise ValueError("E16-A probability target changed")
    metrics, predictions, weight_days, weights_summary, extremes, runtimes = [], [], [], [], [], []
    for fold in FOLDS:
        train, valid = split_market(market, target_ids, fold)
        if set(train["stock_id"]) != set(all_ids):
            raise ValueError(f"Training universe changed in {fold.name}")
        train_ret, valid_ret = training_returns(train), training_returns(valid)
        train_f0, valid_f0 = (join_f0(rows, f0, include_label=False) for rows in (train, valid))
        preprocessor = BaselinePreprocessor().fit(train_f0[list(F0_FEATURES)])
        x_train, x_valid = (preprocessor.transform(rows[list(F0_FEATURES)]) for rows in (train_f0, valid_f0))
        archived_p = align_saved_probabilities(valid, archived, fold.name, SEED)
        taiex = valid_f0["taiex_return"].to_numpy()
        for arm, strength in STRENGTHS.items():
            started = perf_counter()
            weights = sample_weights(train, strength)
            prepared = perf_counter()
            model = fit_classifier(recipe, x_train, train, weights)
            fitted = perf_counter()
            p = model.predict_proba(x_valid)[:, 1].astype(float)
            runtimes.append({"fold": fold.name, "arm": arm, "rows": len(train), "stocks": len(all_ids),
                             "weight_seconds": prepared - started, "fit_seconds": fitted - prepared,
                             "inference_seconds": perf_counter() - fitted})
            if arm == "E3" and not np.allclose(p, archived_p, atol=1e-7, rtol=0):
                raise ValueError(f"Unweighted XGBClassifier did not reproduce archived probabilities in {fold.name}")
            daily, summary, extreme = weight_diagnostics(train, train_ret, weights, fold.name, arm)
            weight_days.append(daily)
            weights_summary.append(summary)
            extremes.append(extreme)
            for name, decision in ((arm, regime_predictions(p, taiex, rule)),
                                   ("E1" if arm == "E3" else f"{arm}-050", (p > .5).astype(int))):
                result, frame = evaluate(name, valid, p, decision, valid_ret, fold.name)
                if name in {"E3", "E1"}:
                    expected = (control if name == "E3" else recipe)["fold_official_scores_by_seed"][str(SEED)][fold.name]
                    expected_calls = regime_predictions(archived_p, taiex, rule) if name == "E3" else (archived_p > .5)
                    if (not np.array_equal(decision, expected_calls)
                            or not np.isclose(result["official_score"], expected, atol=1e-12, rtol=0)):
                        raise ValueError(f"Archived {name} calls or score changed in {fold.name}")
                metrics.append(result)
                predictions.append(frame)
            print(f"{fold.name} {arm} fitted in {fitted - prepared:.2f}s", flush=True)
        source = e16.loc[e16["fold"].eq(fold.name)]
        if not evaluation_keys(valid).equals(evaluation_keys(source)):
            raise ValueError(f"E16-A keys differ for {fold.name}")
        aligned = valid[KEYS].merge(source[KEYS + ["score", "predicted_y"]], on=KEYS, validate="one_to_one")
        p, calls = aligned["score"].to_numpy(), aligned["predicted_y"].to_numpy()
        if not np.array_equal(calls, (p > .5).astype(int)):
            raise ValueError("E16-A archive is inconsistent with its fixed cutoff")
        result, frame = evaluate("E16-A", valid, p, calls, valid_ret, fold.name, "relative_performance")
        metrics.append(result)
        predictions.append(frame)

    paired = pd.concat(predictions, ignore_index=True)
    daily = daily_metrics(paired)
    daily_summary = daily.groupby(["fold", "arm"]).agg(
        daily_mean=("daily_score", "mean"), daily_std=("daily_score", "std"),
        negative_day_rate=("daily_score", lambda v: v.lt(0).mean()),
        daily_p10=("daily_score", lambda v: v.quantile(.1)), daily_median=("daily_score", "median"),
        daily_p90=("daily_score", lambda v: v.quantile(.9)), mean_spearman=("spearman", "mean"),
        defined_spearman_days=("spearman", "count"))
    fold_table = pd.DataFrame(metrics).join(daily_summary, on=["fold", "arm"])
    means = {arm: cv_score(dict(zip(group["fold"], group["official_score"])))
             for arm, group in fold_table.groupby("arm")}
    e16_history = json.loads(E16_SUMMARY.read_text(encoding="utf-8"))
    if not np.isclose(means["E16-A"], e16_history["equal_year_official_scores"]["E16-A"], atol=1e-12, rtol=0):
        raise ValueError("Archived E16-A score changed")
    paths = (F0, DATA / "info.json", DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY,
             E16, E16_SUMMARY, Path(__file__), ROOT / "shared/preprocessing.py", ROOT / "shared/evaluation.py")
    summary = {"seed": SEED, "params": recipe["params"], "rounds": recipe["num_boost_round"], "taiex_rule": rule,
               "equal_year_official_scores": means, "lambda": STRENGTHS, "block_days": BLOCK_DAYS,
               "bootstrap_draws": DRAWS, "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in paths},
               "market_sha256": {stock: _sha256(DATA / "market" / f"{stock}.csv") for stock in all_ids},
               "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                            "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    tables = {"fold_metrics": fold_table, "validation_predictions": paired, "daily_metrics": daily,
              "paired_uncertainty": paired_intervals(daily), "daily_weight_diagnostics": pd.concat(weight_days),
              "weight_summary": pd.DataFrame(weights_summary), "extreme_weight_rows": pd.concat(extremes),
              "runtime": pd.DataFrame(runtimes)}
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    run(parser.parse_args().output_dir)
