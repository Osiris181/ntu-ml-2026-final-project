"""Compare XGBoost training universes and recency weighting on shared CV.

Run from the repository root:
    uv run python -m members.zhenci.e3.train_e3_regime

This is a member E3 experiment. It does not use the 2026 holdout,
updated market data, submission feedback, or any target-day predictors.
"""

from __future__ import annotations

import argparse
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

from members.zhenci.e1.train_e1 import MEMBER, MODEL, join_f0
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market


EXPERIMENT = "E2"  # Shared schema calls every member optimization E2.
THRESHOLD = 0.5
NUM_BOOST_ROUND = 100
PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
    "max_depth": 6, "eta": 0.3, "subsample": 1.0, "colsample_bytree": 1.0,
    "lambda": 1.0, "alpha": 0.0, "nthread": 4,
}

# The first configuration is the exact full-sample E1 training policy.
# All alternatives preserve the same folds, target validation keys and scorer.
CANDIDATES = (
    {"config_id": "e3_all_ordinary", "training_universe": "Top100@Y", "half_life_days": None},
    {"config_id": "e3_target_ordinary", "training_universe": "Top50@Y", "half_life_days": None},
    {"config_id": "e3_all_halflife_730", "training_universe": "Top100@Y", "half_life_days": 730},
    {"config_id": "e3_target_halflife_730", "training_universe": "Top50@Y", "half_life_days": 730},
    {"config_id": "e3_all_halflife_365", "training_universe": "Top100@Y", "half_life_days": 365},
    {"config_id": "e3_target_halflife_365", "training_universe": "Top50@Y", "half_life_days": 365},
)


def select_training_rows(rows: pd.DataFrame, target_ids: list[str], universe: str) -> pd.DataFrame:
    """Restrict only training rows; validation target keys never change."""
    if universe == "Top100@Y":
        selected = rows
    elif universe == "Top50@Y":
        selected = rows.loc[rows["stock_id"].isin(target_ids)]
        if set(selected["stock_id"]) != set(target_ids):
            raise ValueError("Training window lacks at least one target stock")
    else:
        raise ValueError(f"Unknown training universe: {universe}")
    if selected.empty:
        raise ValueError("No training rows after universe selection")
    return selected.reset_index(drop=True)


def recency_weights(dates: pd.Series, train_end: str, half_life_days: int | None) -> np.ndarray | None:
    """Weight historical training observations using the fold's fixed cutoff."""
    if half_life_days is None:
        return None
    if half_life_days <= 0:
        raise ValueError("Half-life must be positive")
    age = (pd.Timestamp(train_end) - dates).dt.days.to_numpy(dtype=float)
    if not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("Training dates exceed their fold cutoff")
    weights = np.exp2(-age / half_life_days)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid recency weights")
    return weights / weights.mean()


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    metrics = []
    scores_by_config = {}
    best_predictions = None
    best_config_id = None
    best_score = -np.inf
    for candidate in CANDIDATES:
        config_id = candidate["config_id"]
        scores = {seed: {} for seed in E1_SEEDS}
        prediction_frames = []
        for fold in FOLDS:
            all_train_rows, valid_rows = split_market(market, target_ids, fold)
            train_rows = select_training_rows(all_train_rows, target_ids, candidate["training_universe"])
            train = join_f0(train_rows, f0, include_label=True)
            valid = join_f0(valid_rows, f0, include_label=False)
            preprocessor = BaselinePreprocessor()
            x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
            x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
            if not x_train.columns.equals(x_valid.columns):
                raise ValueError("Training and validation feature columns differ")
            weights = recency_weights(train["date"], fold.train_end, candidate["half_life_days"])
            dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy(), weight=weights)
            dvalid = xgb.DMatrix(x_valid)
            for seed in E1_SEEDS:
                started = time.perf_counter()
                booster = xgb.train({**PARAMS, "seed": seed}, dtrain, num_boost_round=NUM_BOOST_ROUND)
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_up = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - started
                if len(p_up) != len(valid) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
                    raise ValueError(f"Invalid probabilities for {config_id}, {fold.name}, seed {seed}")
                prediction = valid.loc[:, KEYS].copy()
                prediction["predicted_y"] = (p_up > THRESHOLD).astype(int)
                result = evaluate_predictions(valid_rows, prediction)
                scores[seed][fold.name] = result["official_score"]
                metrics.append({"member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
                                "config_id": config_id, "fold": fold.name, "seed": seed,
                                **result, "train_seconds": train_seconds,
                                "predict_seconds": predict_seconds})
                prediction.insert(0, "seed", seed)
                prediction.insert(0, "fold", fold.name)
                prediction.insert(0, "config_id", config_id)
                prediction.insert(0, "experiment", EXPERIMENT)
                prediction.insert(0, "model", MODEL)
                prediction.insert(0, "member", MEMBER)
                prediction["p_up"] = p_up
                prediction_frames.append(prediction)
        scores_by_config[config_id] = scores
        score = e1_seed_summary(scores)["mean_cv_official_score"]
        if score > best_score:
            best_score = score
            best_config_id = config_id
            best_predictions = pd.concat(prediction_frames, ignore_index=True)
        print(f"{config_id}: mean CV official score {score:.8f}", flush=True)

    configurations = [
        {**candidate, "params": PARAMS, "num_boost_round": NUM_BOOST_ROUND,
         "fold_official_scores_by_seed": scores_by_config[candidate["config_id"]],
         **e1_seed_summary(scores_by_config[candidate["config_id"]])}
        for candidate in CANDIDATES
    ]
    # Candidate order breaks ties; index zero is the E1 control.
    best = max(configurations, key=lambda item: item["mean_cv_official_score"])
    if best["config_id"] != best_config_id:
        raise ValueError("Selected configuration and saved predictions differ")
    baseline = configurations[0]
    summary = {
        "member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
        "experiment_label": "E3 training-universe and recency experiment",
        "search_method": "prespecified 2 training universes x 3 recency policies",
        "trial_budget": len(CANDIDATES), "fit_budget": len(CANDIDATES) * len(FOLDS) * len(E1_SEEDS),
        "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "evaluation_universe": "Top50@Y", "split_version": "1.2",
        "seeds": list(E1_SEEDS), "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "selection_rule": "highest five-seed mean of equal-weight 2023/2024/2025 official CV scores; candidate order breaks ties",
        "selected_config_id": best["config_id"],
        "baseline_cv_score": baseline["mean_cv_official_score"],
        "improvement_over_e1": best["mean_cv_official_score"] - baseline["mean_cv_official_score"],
        "configurations": configurations,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "xgboost": xgb.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics, columns=METRIC_FIELDS).to_csv(output_dir / "fold_metrics.csv", index=False)
    best_predictions.loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "selected_validation_predictions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Selected {best['config_id']}: {best['mean_cv_official_score']:.8f}; E1 control {baseline['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e3_regime_v1")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
