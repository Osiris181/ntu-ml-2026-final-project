"""Compare four prespecified F0 and market-relative XGBoost feature sets.

Run from the repository root:
    uv run python -m members.zhenci.e3.train_e3_feature_sets

This runs validation only; it does not train a deployment model or submit.
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

from members.zhenci.e3.features_e3_relative import GLOBAL_FEATURES, RELATIVE_FEATURES, RelativePreprocessor, build_relative_dataset
from members.zhenci.e1.train_e1 import MEMBER, MODEL, join_f0
from members.zhenci.e3.train_e3_regime import NUM_BOOST_ROUND, PARAMS
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES
from shared.splits import E1_SEEDS, FOLDS, split_market

EXPERIMENT = "E2"  # E3 is the member's experiment number.
VARIANTS = (
    {"config_id": "e3_f0_control", "include_relative": False, "include_global": True},
    {"config_id": "e3_f0_no_global", "include_relative": False, "include_global": False},
    {"config_id": "e3_relative", "include_relative": True, "include_global": True},
    {"config_id": "e3_relative_no_global", "include_relative": True, "include_global": False},
)


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    relative = build_relative_dataset(f0)
    metrics = []
    comparisons = []
    best_score = -np.inf
    best_predictions = None
    best_id = None
    for variant in VARIANTS:
        config_id = variant["config_id"]
        scores = {seed: {} for seed in E1_SEEDS}
        prediction_frames = []
        for fold in FOLDS:
            train_rows, valid_rows = split_market(market, target_ids, fold)
            train = join_f0(train_rows, relative, include_label=True)
            valid = join_f0(valid_rows, relative, include_label=False)
            processor = RelativePreprocessor(include_relative=variant["include_relative"],
                                             include_global=variant["include_global"])
            x_train = processor.fit_transform(train)
            x_valid = processor.transform(valid)
            if not x_train.columns.equals(x_valid.columns):
                raise ValueError("Training and validation columns differ")
            dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy())
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
                prediction["predicted_y"] = (p_up > 0.5).astype(int)
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
        selection = e1_seed_summary(scores)
        comparisons.append({**variant, "feature_order": [name for name in F0_FEATURES
                             if variant["include_global"] or name not in GLOBAL_FEATURES]
                             + (list(RELATIVE_FEATURES) if variant["include_relative"] else []),
                            "fold_official_scores_by_seed": scores, **selection})
        value = selection["mean_cv_official_score"]
        if value > best_score:
            best_score = value
            best_id = config_id
            best_predictions = pd.concat(prediction_frames, ignore_index=True)
        print(f"{config_id}: mean CV official score {value:.8f}", flush=True)

    summary = {
        "member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
        "experiment_label": "E3 market-relative and global-ablation feature study",
        "feature_source": "shared F0 plus same-date cross-stock aggregation of lagged F0 values",
        "relative_feature_definitions": {
            RELATIVE_FEATURES[0]: "F0 intraday_return_1d minus same-date median over available supplied stocks",
            RELATIVE_FEATURES[1]: "F0 momentum_5d minus same-date median over available supplied stocks",
            RELATIVE_FEATURES[2]: "within-date percentile rank of F0 relative_volume_20d over available supplied stocks",
        },
        "global_ablation": list(GLOBAL_FEATURES),
        "params": PARAMS, "num_boost_round": NUM_BOOST_ROUND,
        "preprocessing": "shared F0 fold-fitted baseline; extra numeric fold-fitted median and standard scale",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "selection_rule": "highest five-seed mean of equal-weight 2023/2024/2025 official CV scores; variant order breaks ties",
        "trial_budget": len(VARIANTS), "fit_budget": len(VARIANTS) * len(FOLDS) * len(E1_SEEDS),
        "selected_config_id": best_id,
        "improvement_over_f0": best_score - comparisons[0]["mean_cv_official_score"],
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
    print(f"Selected {best_id}: {best_score:.8f}; F0 control {comparisons[0]['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e3_feature_sets_v1")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
