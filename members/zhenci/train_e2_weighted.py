"""Test whether training weights aligned with signed-return scoring help XGBoost.

Run from the repository root:
    uv run python -m members.zhenci.train_e2_weighted

The two arms use the selected E1 model settings, shared F0/preprocessing,
chronological folds, five seeds, and the official validation scorer. Weights
come only from each fold's training rows. No holdout or submission feedback is
used for fitting or selection; this script does not save a deployment model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.save_e1 import DEFAULT_CV_SUMMARY, _read_cv_summary
from members.zhenci.train_e1 import MEMBER, MODEL, join_f0
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market


EXPERIMENT = "E2"
THRESHOLD = 0.5
ARMS = ("unweighted_control", "return_weighted_v1")
OPEN = "etl:adj_open"
CLOSE = "etl:adj_close"


def training_return_weights(train_rows: pd.DataFrame) -> np.ndarray:
    """Scale absolute training returns to mean weight one within a fold.

    This keeps the overall weight scale comparable with the unweighted XGBoost
    setup. Flat sessions have zero weight because their signed score is zero.
    """
    required = {OPEN, CLOSE, "y"}
    if required - set(train_rows.columns):
        raise ValueError(f"Training weights require {sorted(required)}")
    prices = train_rows.loc[:, [OPEN, CLOSE]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices[:, 0] <= 0).any():
        raise ValueError("Training prices must be finite with positive adjusted open")
    returns = (prices[:, 1] - prices[:, 0]) / prices[:, 0]
    labels = train_rows["y"].to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.array_equal(labels.astype(int), (returns > 0).astype(int)):
        raise ValueError("Training labels disagree with adjusted price direction")
    magnitudes = np.abs(returns)
    average = magnitudes.mean()
    if not np.isfinite(average) or average <= 0:
        raise ValueError("Training returns have no usable magnitude")
    return magnitudes / average


def run(data_dir: Path, f0_path: Path, e1_summary_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    baseline = _read_cv_summary(e1_summary_path)
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    metric_rows = []
    prediction_frames = []
    fold_scores = {arm: {seed: {} for seed in E1_SEEDS} for arm in ARMS}
    weight_diagnostics = {}

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)
        preprocessor = BaselinePreprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation feature columns differ")
        weights = training_return_weights(train_rows)
        weight_diagnostics[fold.name] = {
            "train_rows": len(weights),
            "mean_weight": float(weights.mean()),
            "zero_weight_rows": int(np.count_nonzero(weights == 0)),
            "max_weight": float(weights.max()),
        }
        dtrain = {
            "unweighted_control": xgb.DMatrix(x_train, label=train_rows["y"].to_numpy()),
            "return_weighted_v1": xgb.DMatrix(x_train, label=train_rows["y"].to_numpy(), weight=weights),
        }
        dvalid = xgb.DMatrix(x_valid)

        for arm in ARMS:
            for seed in E1_SEEDS:
                params = {**baseline["params"], "seed": seed}
                train_start = time.perf_counter()
                booster = xgb.train(params, dtrain[arm], num_boost_round=baseline["num_boost_round"])
                train_seconds = time.perf_counter() - train_start
                predict_start = time.perf_counter()
                p_up = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - predict_start
                if len(p_up) != len(valid_rows) or not np.isfinite(p_up).all() or not np.logical_and(p_up >= 0, p_up <= 1).all():
                    raise ValueError(f"Invalid probabilities for {arm}, {fold.name}, seed {seed}")

                predictions = valid.loc[:, KEYS].copy()
                predictions["predicted_y"] = (p_up > THRESHOLD).astype(int)
                result = evaluate_predictions(valid_rows, predictions)
                fold_scores[arm][seed][fold.name] = result["official_score"]
                config_id = f"{baseline['config_id']}__{arm}"
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
                predictions["p_up"] = p_up
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
            "config_id": f"{baseline['config_id']}__{arm}",
            "fold_official_scores_by_seed": fold_scores[arm],
            **e1_seed_summary(fold_scores[arm]),
        })
    control, weighted = comparisons
    delta = weighted["mean_cv_official_score"] - control["mean_cv_official_score"]
    summary = {
        "member": MEMBER,
        "model": MODEL,
        "experiment": EXPERIMENT,
        "experiment_type": "paired_unweighted_vs_return_weighted_training",
        "reference_e1_config_id": baseline["config_id"],
        "reference_e1_summary_sha256": hashlib.sha256(e1_summary_path.read_bytes()).hexdigest(),
        "params": baseline["params"],
        "num_boost_round": baseline["num_boost_round"],
        "feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "training_weights": "abs((adjusted_close - adjusted_open) / adjusted_open), divided by training-fold mean absolute return; zero for flat sessions",
        "weight_diagnostics": weight_diagnostics,
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "trial_budget": len(ARMS),
        "fit_budget": len(ARMS) * len(FOLDS) * len(E1_SEEDS),
        "comparisons": comparisons,
        "delta_weighted_minus_unweighted_mean_cv_official_score": delta,
        "selected_arm": "return_weighted_v1" if delta > 0 else "unweighted_control",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows, columns=METRIC_FIELDS).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Weighted minus unweighted five-seed mean CV score: {delta:+.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e2_weighted_v1")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.e1_summary, args.output_dir)
