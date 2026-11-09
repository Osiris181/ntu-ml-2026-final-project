"""Compare XGBoost's DART booster with the fixed E1 tree booster on F0.

Run from the repository root:
    uv run python -m members.zhenci.e3.train_e3_dart

Validation only. DART prediction explicitly uses all trained rounds.
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
from members.zhenci.e3.train_e3_regime import NUM_BOOST_ROUND, PARAMS
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market

EXPERIMENT = "E2"
CANDIDATES = (
    {"config_id": "e3_gbtree_control", "booster": "gbtree", "rate_drop": None},
    {"config_id": "e3_dart_rate_005", "booster": "dart", "rate_drop": 0.05},
    {"config_id": "e3_dart_rate_010", "booster": "dart", "rate_drop": 0.10},
)


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    fold_data = []
    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)
        processor = BaselinePreprocessor()
        x_train = processor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = processor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation columns differ")
        fold_data.append((fold.name, xgb.DMatrix(x_train, label=train["y"].to_numpy()),
                          xgb.DMatrix(x_valid), valid_rows, valid.loc[:, KEYS]))

    metrics = []
    comparisons = []
    best_score = -np.inf
    best_predictions = None
    best_id = None
    for candidate in CANDIDATES:
        config_id = candidate["config_id"]
        scores = {seed: {} for seed in E1_SEEDS}
        prediction_frames = []
        for fold_name, dtrain, dvalid, valid_rows, valid_keys in fold_data:
            for seed in E1_SEEDS:
                params = {**PARAMS, "seed": seed, "booster": candidate["booster"]}
                if candidate["booster"] == "dart":
                    params.update(rate_drop=candidate["rate_drop"], sample_type="uniform",
                                  normalize_type="tree", skip_drop=0.0)
                started = time.perf_counter()
                booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_up = booster.predict(dvalid, iteration_range=(0, NUM_BOOST_ROUND))
                predict_seconds = time.perf_counter() - started
                if len(p_up) != len(valid_keys) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
                    raise ValueError(f"Invalid probabilities for {config_id}, {fold_name}, seed {seed}")
                prediction = valid_keys.copy()
                prediction["predicted_y"] = (p_up > 0.5).astype(int)
                result = evaluate_predictions(valid_rows, prediction)
                scores[seed][fold_name] = result["official_score"]
                metrics.append({"member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
                                "config_id": config_id, "fold": fold_name, "seed": seed,
                                **result, "train_seconds": train_seconds,
                                "predict_seconds": predict_seconds})
                prediction.insert(0, "seed", seed)
                prediction.insert(0, "fold", fold_name)
                prediction.insert(0, "config_id", config_id)
                prediction.insert(0, "experiment", EXPERIMENT)
                prediction.insert(0, "model", MODEL)
                prediction.insert(0, "member", MEMBER)
                prediction["p_up"] = p_up
                prediction_frames.append(prediction)
        selection = e1_seed_summary(scores)
        comparisons.append({**candidate, "params": {**PARAMS, "booster": candidate["booster"]},
                            "num_boost_round": NUM_BOOST_ROUND,
                            "fold_official_scores_by_seed": scores, **selection})
        value = selection["mean_cv_official_score"]
        if value > best_score:
            best_score = value
            best_id = config_id
            best_predictions = pd.concat(prediction_frames, ignore_index=True)
        print(f"{config_id}: mean CV official score {value:.8f}", flush=True)

    summary = {
        "member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
        "experiment_label": "E3 DART booster comparison",
        "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "prediction_rule": "use all 100 trained rounds with iteration_range=(0,100)",
        "selection_rule": "highest five-seed mean of equal-weight 2023/2024/2025 official CV scores; candidate order breaks ties",
        "trial_budget": len(CANDIDATES), "fit_budget": len(CANDIDATES) * len(FOLDS) * len(E1_SEEDS),
        "selected_config_id": best_id,
        "improvement_over_gbtree": best_score - comparisons[0]["mean_cv_official_score"],
        "comparisons": comparisons,
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
    print(f"Selected {best_id}: {best_score:.8f}; gbtree control {comparisons[0]['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e3_dart_v1")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
