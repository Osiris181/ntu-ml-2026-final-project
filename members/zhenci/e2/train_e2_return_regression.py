"""Compare XGBoost direction classification with direct return regression.

Run from the repository root:
    uv run python -m members.zhenci.e2.train_e2_return_regression
For seed-sensitive sampled models in a separate result directory:
    uv run python -m members.zhenci.e2.train_e2_return_regression --sampled

Both arms use the selected E1 tree settings, F0, baseline preprocessing,
three chronological folds, five seeds, and the official scorer. The regression
target is known only for training rows. No holdout or submission feedback is
used for fitting or selection; no deployment model is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.save_e1 import DEFAULT_CV_SUMMARY, _read_cv_summary
from members.zhenci.e1.train_e1 import MEMBER, MODEL, join_f0
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market


EXPERIMENT = "E2"
ARMS = ("classification_control", "return_regression_v1")
OPEN = "etl:adj_open"
CLOSE = "etl:adj_close"


def training_intraday_returns(train_rows: pd.DataFrame) -> np.ndarray:
    """Calculate the regression target from training rows only."""
    if {OPEN, CLOSE, "y"} - set(train_rows.columns):
        raise ValueError("Training returns require adjusted open, close, and y")
    prices = train_rows.loc[:, [OPEN, CLOSE]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices[:, 0] <= 0).any():
        raise ValueError("Training prices must be finite with positive adjusted open")
    returns = (prices[:, 1] - prices[:, 0]) / prices[:, 0]
    labels = train_rows["y"].to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.array_equal(labels.astype(int), (returns > 0).astype(int)):
        raise ValueError("Training labels disagree with adjusted price direction")
    return returns


def binary_from_regression(predicted_return: np.ndarray) -> np.ndarray:
    """Long when expected intraday return is positive, short otherwise."""
    values = np.asarray(predicted_return, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Regression predictions must be a finite one-dimensional array")
    return (values > 0).astype(int)


def comparison_params(reference_params: dict, sampled: bool) -> dict:
    """Give both arms identical tree settings, with optional seed-active sampling."""
    params = dict(reference_params)
    if sampled:
        params.update(subsample=0.8, colsample_bytree=0.8)
    return params


def run(data_dir: Path, f0_path: Path, e1_summary_path: Path, output_dir: Path, *, sampled: bool = False) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    baseline = _read_cv_summary(e1_summary_path)
    base_params = comparison_params(baseline["params"], sampled)
    config_prefix = f"{baseline['config_id']}__{'sampled_v2__' if sampled else ''}"
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    metric_rows = []
    prediction_frames = []
    fold_scores = {arm: {seed: {} for seed in E1_SEEDS} for arm in ARMS}

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)
        preprocessor = BaselinePreprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation feature columns differ")
        returns = training_intraday_returns(train_rows)
        dtrain = {
            "classification_control": xgb.DMatrix(x_train, label=train_rows["y"].to_numpy()),
            "return_regression_v1": xgb.DMatrix(x_train, label=returns),
        }
        dvalid = xgb.DMatrix(x_valid)

        for arm in ARMS:
            params = dict(base_params)
            if arm == "return_regression_v1":
                params.update(objective="reg:squarederror", eval_metric="rmse")
            for seed in E1_SEEDS:
                train_start = time.perf_counter()
                booster = xgb.train(
                    {**params, "seed": seed}, dtrain[arm],
                    num_boost_round=baseline["num_boost_round"],
                )
                train_seconds = time.perf_counter() - train_start
                predict_start = time.perf_counter()
                raw_output = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - predict_start
                if len(raw_output) != len(valid_rows) or not np.isfinite(raw_output).all():
                    raise ValueError(f"Invalid model output for {arm}, {fold.name}, seed {seed}")
                if arm == "classification_control":
                    if not np.logical_and(raw_output >= 0, raw_output <= 1).all():
                        raise ValueError("Classification output is not a probability")
                    predicted_y = (raw_output > 0.5).astype(int)
                else:
                    predicted_y = binary_from_regression(raw_output)

                predictions = valid.loc[:, KEYS].copy()
                predictions["predicted_y"] = predicted_y
                result = evaluate_predictions(valid_rows, predictions)
                fold_scores[arm][seed][fold.name] = result["official_score"]
                config_id = f"{config_prefix}{arm}"
                metric_rows.append({
                    "member": MEMBER,
                    "model": MODEL,
                    "experiment": EXPERIMENT,
                    "config_id": config_id,
                    "fold": fold.name,
                    "seed": seed,
                    **result,
                    "train_seconds": train_seconds,
                    "predict_seconds": predict_seconds,
                })
                predictions.insert(0, "seed", seed)
                predictions.insert(0, "fold", fold.name)
                predictions.insert(0, "config_id", config_id)
                predictions.insert(0, "experiment", EXPERIMENT)
                predictions.insert(0, "model", MODEL)
                predictions.insert(0, "member", MEMBER)
                predictions["raw_output"] = raw_output
                prediction_frames.append(predictions)
                print(
                    f"{arm}, {fold.name}, seed {seed}: "
                    f"{result['n_eval']} predictions; official score {result['official_score']:.8f}",
                    flush=True,
                )

    comparisons = []
    for arm in ARMS:
        comparisons.append({
            "arm": arm,
            "config_id": f"{config_prefix}{arm}",
            "objective": "binary:logistic" if arm == "classification_control" else "reg:squarederror",
            "decision_rule": "p_up > 0.5" if arm == "classification_control" else "predicted_return > 0",
            "fold_official_scores_by_seed": fold_scores[arm],
            **e1_seed_summary(fold_scores[arm]),
        })
    control, regression = comparisons
    delta = regression["mean_cv_official_score"] - control["mean_cv_official_score"]
    summary = {
        "member": MEMBER,
        "model": MODEL,
        "experiment": EXPERIMENT,
        "experiment_type": "paired_direction_classification_vs_return_regression",
        "reference_e1_config_id": baseline["config_id"],
        "reference_e1_summary_sha256": hashlib.sha256(e1_summary_path.read_bytes()).hexdigest(),
        "reference_params": baseline["params"],
        "sampled_mode": sampled,
        "params_used_by_both_arms": base_params,
        "num_boost_round": baseline["num_boost_round"],
        "feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "regression_target": "(training adjusted_close - training adjusted_open) / training adjusted_open",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "trial_budget": len(ARMS),
        "fit_budget": len(ARMS) * len(FOLDS) * len(E1_SEEDS),
        "comparisons": comparisons,
        "delta_regression_minus_classification_mean_cv_official_score": delta,
        "selected_arm": "return_regression_v1" if delta > 0 else "classification_control",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows, columns=METRIC_FIELDS).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).loc[:, [*PREDICTION_FIELDS, "raw_output"]].to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Regression minus classification five-seed mean CV score: {delta:+.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--sampled", action="store_true",
                        help="Use 0.8 row and column sampling in both arms so the five seeds can differ")
    parser.add_argument("--output-dir", type=Path,
                        help="New result directory (defaults to a separate path for sampled mode)")
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / (
        "results/zhenci_xgboost_e2_return_regression_sampled_v2"
        if args.sampled else "results/zhenci_xgboost_e2_return_regression_v1"
    )
    run(args.data_dir, args.f0_path, args.e1_summary, output_dir, sampled=args.sampled)
