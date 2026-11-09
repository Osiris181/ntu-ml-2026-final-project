"""Run one fixed sampled XGBoost E1 configuration for five seeds and three folds.

Run from the repository root with:
    uv run python -m members.zhenci.e1.train_e1

Direct file execution also works:
    uv run python members/zhenci/e1/train_e1.py

This script does not use the Jan-May 2026 holdout, tune a configuration,
refit a deployment model, or create a competition submission.
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

from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market

MEMBER = "鄭振慈"
MODEL = "XGBoost"
EXPERIMENT = "E1"
CONFIG_ID = "xgboost_sampling_v1"
NUM_BOOST_ROUND = 100
THRESHOLD = 0.5

# Use the same fixed sampling rates for every fold and seed. The seed controls
# which rows and features are sampled; all shared E1 inputs and scoring stay fixed.
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.0,
    "nthread": 4,
}


def join_f0(rows: pd.DataFrame, f0: pd.DataFrame, *, include_label: bool) -> pd.DataFrame:
    """Keep split row order and fail if any required stock/date feature row is absent."""
    columns = KEYS + (["y"] if include_label else [])
    joined = rows.loc[:, columns].merge(
        f0, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", KEYS]
        raise ValueError(f"Missing F0 rows for {len(missing)} split stock/date keys")
    return joined.drop(columns="_merge")


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    metric_rows = []
    prediction_frames = []
    fold_scores = {seed: {} for seed in E1_SEEDS}

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)

        # Refit only on this fold's training rows. The untouched F0 file may contain NaNs.
        preprocessor = BaselinePreprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation feature columns differ")

        dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy())
        dvalid = xgb.DMatrix(x_valid)
        for seed in E1_SEEDS:
            params = {**PARAMS, "seed": seed}
            train_start = time.perf_counter()
            booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
            train_seconds = time.perf_counter() - train_start
            predict_start = time.perf_counter()
            p_up = booster.predict(dvalid)
            predict_seconds = time.perf_counter() - predict_start
            if len(p_up) != len(valid) or not np.isfinite(p_up).all():
                raise ValueError(f"Invalid probability output for fold {fold.name}, seed {seed}")

            predictions = valid.loc[:, KEYS].copy()
            predictions["predicted_y"] = (p_up > THRESHOLD).astype(int)
            result = evaluate_predictions(valid_rows, predictions)
            fold_scores[seed][fold.name] = result["official_score"]
            metric_rows.append({
                "member": MEMBER,
                "model": MODEL,
                "experiment": EXPERIMENT,
                "config_id": CONFIG_ID,
                "fold": fold.name,
                "seed": seed,
                **result,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            })
            predictions.insert(0, "seed", seed)
            predictions.insert(0, "fold", fold.name)
            predictions.insert(0, "config_id", CONFIG_ID)
            predictions.insert(0, "experiment", EXPERIMENT)
            predictions.insert(0, "model", MODEL)
            predictions.insert(0, "member", MEMBER)
            predictions["p_up"] = p_up
            prediction_frames.append(predictions)
            print(f"{fold.name}, seed {seed}: {result['n_eval']} predictions; official score {result['official_score']:.8f}")

    selection = e1_seed_summary(fold_scores)

    summary = {
        "member": MEMBER,
        "model": MODEL,
        "experiment": EXPERIMENT,
        "config_id": CONFIG_ID,
        "feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "fold_official_scores_by_seed": fold_scores,
        **selection,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows, columns=METRIC_FIELDS).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).loc[
        :, [*PREDICTION_FIELDS, "p_up"]
    ].to_csv(output_dir / "validation_predictions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Five-seed mean CV official score: {summary['mean_cv_official_score']:.8f}")
    print(f"Five-seed CV standard deviation: {summary['std_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/zheng_xgboost_e1_sampling_v1"
    )
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
