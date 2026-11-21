"""Run E16 with the original F0: uv run python -m members.zhenci.xgboost.e16.train_e16."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
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
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import cv_score, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import FOLDS, SEED, split_market

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
E14_ARCHIVE = ROOT / "members/zhenci/xgboost/e14/outputs/historical_predictions.csv"
COMPARISONS = (("E16-B", "E3-Rank"), ("E16-A", "E3"), ("E16-B", "E3"), ("E3-Rank", "E3"))


def relative_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Label the supplied panel only. Ties at its daily median are zero."""
    validate_keys(rows)
    realized = training_returns(rows)
    if not np.isfinite(realized).all():
        raise ValueError("Intraday returns must be finite")
    labels = rows[KEYS].assign(realized_return=realized, absolute_up=rows["y"].to_numpy())
    labels["peer_median"] = labels.groupby("date")["realized_return"].transform("median")
    labels["relative_up"] = labels["realized_return"].gt(labels["peer_median"]).astype(int)
    labels["median_tie"] = labels["realized_return"].eq(labels["peer_median"])
    return labels


def balanced_calls(rows: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Use E14 tie ordering and floor(N/2) up calls on incomplete dates."""
    validate_keys(rows)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected one finite logistic score per evaluation key")
    ordered = rows[KEYS].reset_index(drop=True).assign(score=scores)
    ordered = ordered.sort_values(["date", "score", "stock_id"], ascending=[True, False, True])
    groups = ordered.groupby("date")
    count = groups["score"].transform("size")
    if count.gt(50).any():
        raise ValueError("Output ranking expects at most 50 eligible targets per date")
    return groups.cumcount().lt(count // 2).sort_index().to_numpy(dtype=int)


def auc(labels: pd.Series, scores: pd.Series) -> float:
    """ROC AUC from average ranks, with half credit for score ties."""
    positive = labels.eq(1)
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = scores.rank(method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    sums = daily.pivot(index=["fold", "date"], columns="arm", values="contribution_sum")
    counts = daily.pivot(index=["fold", "date"], columns="arm", values="n")
    if sums.isna().any().any() or not counts.eq(counts["E3"], axis=0).all().all():
        raise ValueError("Arms must share daily evaluation rows")
    rng = np.random.default_rng(SEED)
    results = []
    for arm, reference in COMPARISONS:
        points, samples = [], []
        for fold in FOLDS:
            difference = (sums.loc[fold.name, arm] - sums.loc[fold.name, reference]).to_numpy()
            n_rows = counts.loc[fold.name, reference].to_numpy()
            n_days = len(difference)
            starts = rng.integers(0, n_days, size=(DRAWS, (n_days + BLOCK_DAYS - 1) // BLOCK_DAYS))
            indices = ((starts[..., None] + np.arange(BLOCK_DAYS)) % n_days).reshape(DRAWS, -1)[:, :n_days]
            sample = difference[indices].sum(axis=1) / n_rows[indices].sum(axis=1)
            point = float(difference.sum() / n_rows.sum())
            results.append(interval(arm, reference, fold.name, point, sample))
            points.append(point)
            samples.append(sample)
        results.append(interval(arm, reference, "equal_year", np.mean(points), np.mean(samples, axis=0)))
    return pd.DataFrame(results)


def verify_e14(archived: pd.DataFrame | None, fold: str, labels: pd.DataFrame,
               original: np.ndarray, ranking: np.ndarray) -> None:
    if archived is None:
        return
    complete = labels.groupby("date")["stock_id"].transform("size").eq(50)
    observed = labels[KEYS].assign(e3=original, e3_rank=ranking).loc[complete]
    expected = archived.loc[archived["fold"].eq(fold), KEYS + ["e3", "e3_rank"]]
    left = observed.sort_values(KEYS).reset_index(drop=True)
    right = expected.sort_values(KEYS).reset_index(drop=True)
    if (not left[KEYS].equals(right[KEYS])
            or not np.array_equal(left[["e3", "e3_rank"]], right[["e3", "e3_rank"]])):
        raise ValueError(f"E14 complete-date control mismatch for {fold}")


def run(data_dir: Path, f0_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a fresh E16 output directory: {output}")
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    f0 = load_f0_dataset(f0_path).loc[lambda rows: rows["date"].le(FOLDS[-1].eval_end)]
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, e3 = load_fixed_rule(DEFAULT_E3_SUMMARY, recipe["config_id"])
    control = next(row for row in e3["comparisons"] if row["config_id"] == rule["config_id"])
    saved = load_saved_e1_predictions(DEFAULT_E1_PREDICTIONS, recipe)
    e14 = (pd.read_csv(E14_ARCHIVE, dtype={"fold": str, "stock_id": str}, parse_dates=["date"])
           if E14_ARCHIVE.is_file() else None)
    daily_labels, runtime, predictions, scores = [], [], [], []
    for fold in FOLDS:
        started = perf_counter()
        train, valid = split_market(market, target_ids, fold)
        if set(train["stock_id"]) != set(all_ids):
            raise ValueError(f"Training stock universe changed for {fold.name}")
        train_labels = relative_labels(train)
        label_days = train_labels.groupby("date", as_index=False).agg(
            training_rows=("stock_id", "size"), absolute_up=("absolute_up", "sum"),
            relative_up=("relative_up", "sum"), median_ties=("median_tie", "sum"),
            median_return=("peer_median", "first"))
        label_days = label_days.assign(fold=fold.name, absent_stocks=100 - label_days["training_rows"])
        for target in ("absolute", "relative"):
            label_days[f"{target}_down_or_tied"] = label_days["training_rows"] - label_days[f"{target}_up"]
            label_days[f"{target}_up_rate"] = label_days[f"{target}_up"] / label_days["training_rows"]
        daily_labels.append(label_days)
        train_x = join_f0(train, f0, include_label=False)
        valid_x = join_f0(valid, f0, include_label=False)
        preprocessor = BaselinePreprocessor().fit(train_x[list(F0_FEATURES)])
        x_train = preprocessor.transform(train_x[list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid_x[list(F0_FEATURES)])
        dtrain = xgb.DMatrix(x_train, label=train_labels["relative_up"].to_numpy())
        dvalid = xgb.DMatrix(x_valid)
        prepared = perf_counter()
        model = xgb.train({**recipe["params"], "seed": SEED}, dtrain,
                          num_boost_round=recipe["num_boost_round"])
        trained = perf_counter()
        relative = model.predict(dvalid).astype(float)
        inferred = perf_counter()
        runtime.append({"fold": fold.name, "training_rows": len(train), "training_stocks": train["stock_id"].nunique(),
                        "covered_training_key_fraction": len(train_labels) / len(train),
                        "preparation_seconds": prepared - started, "fit_seconds": trained - prepared,
                        "inference_seconds": inferred - trained})

        # Validation outcomes enter only after model prediction, for scoring and diagnostics.
        evaluation = relative_labels(valid)
        absolute = align_saved_probabilities(valid, saved, fold.name, SEED)
        original = regime_predictions(absolute, valid_x["taiex_return"].to_numpy(), rule)
        ranked_absolute = balanced_calls(evaluation, absolute)
        ranked_relative = balanced_calls(evaluation, relative)
        verify_e14(e14, fold.name, evaluation, original, ranked_absolute)
        for arm, probability, calls, score_target in (
            ("E3", absolute, original, "absolute_direction"),
            ("E3-Rank", absolute, ranked_absolute, "absolute_direction"),
            ("E16-A", relative, (relative > .5).astype(int), "relative_performance"),
            ("E16-B", relative, ranked_relative, "relative_performance"),
        ):
            frame = evaluation.assign(fold=fold.name, arm=arm, predicted_y=calls, score=probability,
                                      score_target=score_target, contribution=(2 * calls - 1) * evaluation["realized_return"])
            result = evaluate_predictions(valid, frame[KEYS + ["predicted_y"]])
            if arm == "E3" and not np.isclose(result["official_score"],
                    control["fold_official_scores_by_seed"][str(SEED)][fold.name], atol=1e-12, rtol=0):
                raise ValueError(f"Archived E3 score mismatch for {fold.name}")
            day_auc = [auc(group["relative_up"], group["score"]) for _, group in frame.groupby("date")]
            scores.append({"fold": fold.name, "arm": arm, **result, "predicted_up_rate": calls.mean(),
                           "relative_accuracy": frame["predicted_y"].eq(frame["relative_up"]).mean(),
                           "relative_auc": auc(frame["relative_up"], frame["score"]),
                           "mean_daily_relative_auc": np.nanmean(day_auc),
                           "score_p10": np.quantile(probability, .1), "score_median": np.median(probability),
                           "score_p90": np.quantile(probability, .9)})
            predictions.append(frame)
        print(f"{fold.name}: fitted on {len(train):,} rows in {trained - prepared:.2f}s", flush=True)

    prediction_table = pd.concat(predictions, ignore_index=True)
    score_table = pd.DataFrame(scores)
    daily = daily_metrics(prediction_table.rename(columns={"score": "p_up"}))
    daily_summary = daily.groupby(["fold", "arm"]).agg(
        trading_days=("date", "size"), daily_std=("daily_score", "std"),
        daily_p10=("daily_score", lambda x: x.quantile(.1)), daily_median=("daily_score", "median"),
        daily_p90=("daily_score", lambda x: x.quantile(.9)),
        positive_day_rate=("daily_score", lambda x: x.gt(0).mean()),
        mean_spearman=("spearman", "mean"), defined_spearman_days=("spearman", "count"))
    score_table = score_table.join(daily_summary, on=["fold", "arm"])
    means = {arm: cv_score(dict(zip(group["fold"], group["official_score"])))
             for arm, group in score_table.groupby("arm")}
    label_table = pd.concat(daily_labels, ignore_index=True)
    label_summary = label_table.groupby("fold").agg(
        trading_days=("date", "size"), training_rows=("training_rows", "sum"), absent_stock_dates=("absent_stocks", "sum"),
        absolute_up=("absolute_up", "sum"), relative_up=("relative_up", "sum"), median_ties=("median_ties", "sum"),
        absolute_daily_up_std=("absolute_up_rate", "std"), relative_daily_up_std=("relative_up_rate", "std"),
        absolute_daily_up_min=("absolute_up_rate", "min"), absolute_daily_up_max=("absolute_up_rate", "max"),
        relative_daily_up_min=("relative_up_rate", "min"), relative_daily_up_max=("relative_up_rate", "max"))
    for target in ("absolute", "relative"):
        label_summary[f"{target}_up_rate"] = label_summary[f"{target}_up"] / label_summary["training_rows"]
        label_summary[f"{target}_down_or_tied"] = label_summary["training_rows"] - label_summary[f"{target}_up"]
    uncertainty = paired_intervals(daily)
    paths = [f0_path, data_dir / "info.json", DEFAULT_E1_SUMMARY, DEFAULT_E1_PREDICTIONS, DEFAULT_E3_SUMMARY, Path(__file__)]
    if e14 is not None:
        paths.append(E14_ARCHIVE)
    summary = {"seed": SEED, "params": recipe["params"], "rounds": recipe["num_boost_round"], "taiex_rule": rule,
               "equal_year_official_scores": means, "block_days": BLOCK_DAYS, "bootstrap_draws": DRAWS,
               "e14_complete_days_verified": e14 is not None,
               "input_sha256": {f"{path.parent.name}/{path.name}": _sha256(path) for path in paths},
               "market_sha256": {stock: _sha256(data_dir / "market" / f"{stock}.csv") for stock in all_ids},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "xgboost": xgb.__version__}}
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("fold_metrics", score_table), ("validation_predictions", prediction_table),
                        ("daily_metrics", daily), ("paired_uncertainty", uncertainty),
                        ("daily_training_labels", label_table), ("training_label_summary", label_summary.reset_index()),
                        ("runtime", pd.DataFrame(runtime))):
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
