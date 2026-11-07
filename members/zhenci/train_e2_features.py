"""Compare unchanged F0 with member-owned F1 using one fixed XGBoost setup.

Run from the repository root:
    uv run python -m members.zhenci.train_e2_features

The model settings come from the already selected E2 grid configuration.
Both feature sets use the same folds, seeds, target keys, and official scorer.
No 2026 holdout, Phase 1 result, model refit, or submission is used here.
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

from members.zhenci.features_e2 import EXTRA_FEATURES, F1Preprocessor, build_f1_dataset
from members.zhenci.train_e1 import MEMBER, MODEL, join_f0
from members.zhenci.train_e2 import BASE_PARAMS, CANDIDATES, select_best_config
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market


EXPERIMENT = "E2"
THRESHOLD = 0.5
FEATURE_SETS = ("F0_control", "F1_stock_history_v1")
DEFAULT_E2_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e2_grid_v1/summary.json"


def load_selected_grid_config(path: Path) -> dict:
    """Freeze the preceding E2 grid's selected model settings for both arms."""
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    required = {
        "model": "XGBoost",
        "experiment": "E2",
        "feature_set": "F0",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
    }
    if any(summary.get(field) != value for field, value in required.items()):
        raise ValueError("Incompatible preceding E2 grid summary")
    if summary.get("feature_order") != list(F0_FEATURES) or summary.get("seeds") != list(E1_SEEDS):
        raise ValueError("Preceding E2 grid changed F0 features or seed schedule")
    configurations = summary.get("configurations")
    if not isinstance(configurations, list) or select_best_config(configurations) != summary.get("selected_config_id"):
        raise ValueError("Preceding E2 grid has an inconsistent selected configuration")
    selected_id = summary["selected_config_id"]
    selected = next(item for item in configurations if item["config_id"] == selected_id)
    candidate = next(item for item in CANDIDATES if item["config_id"] == selected_id)
    expected_params = {**BASE_PARAMS, "max_depth": candidate["max_depth"], "eta": candidate["eta"]}
    if selected.get("params") != expected_params or selected.get("num_boost_round") != candidate["num_boost_round"]:
        raise ValueError("Selected E2 grid settings differ from the recorded search")
    return {
        "config_id": selected_id,
        "params": expected_params,
        "num_boost_round": candidate["num_boost_round"],
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run(data_dir: Path, f0_path: Path, e2_summary_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    fixed = load_selected_grid_config(e2_summary_path)
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    f1 = build_f1_dataset(market, f0)
    metric_rows = []
    prediction_frames = []
    fold_scores = {feature_set: {seed: {} for seed in E1_SEEDS} for feature_set in FEATURE_SETS}

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train_f0 = join_f0(train_rows, f0, include_label=True)
        valid_f0 = join_f0(valid_rows, f0, include_label=False)
        train_f1 = join_f0(train_rows, f1, include_label=True)
        valid_f1 = join_f0(valid_rows, f1, include_label=False)

        base = BaselinePreprocessor()
        extended = F1Preprocessor()
        prepared = {
            "F0_control": (
                base.fit_transform(train_f0.loc[:, list(F0_FEATURES)]),
                base.transform(valid_f0.loc[:, list(F0_FEATURES)]),
            ),
            "F1_stock_history_v1": (
                extended.fit_transform(train_f1),
                extended.transform(valid_f1),
            ),
        }
        for feature_set in FEATURE_SETS:
            x_train, x_valid = prepared[feature_set]
            if not x_train.columns.equals(x_valid.columns):
                raise ValueError(f"Training and validation columns differ for {feature_set}")
            dtrain = xgb.DMatrix(x_train, label=train_rows["y"].to_numpy())
            dvalid = xgb.DMatrix(x_valid)
            for seed in E1_SEEDS:
                params = {**fixed["params"], "seed": seed}
                train_start = time.perf_counter()
                booster = xgb.train(params, dtrain, num_boost_round=fixed["num_boost_round"])
                train_seconds = time.perf_counter() - train_start
                predict_start = time.perf_counter()
                p_up = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - predict_start
                if len(p_up) != len(valid_rows) or not np.isfinite(p_up).all() or not np.logical_and(p_up >= 0, p_up <= 1).all():
                    raise ValueError(f"Invalid probabilities for {feature_set}, {fold.name}, seed {seed}")

                predictions = valid_rows.loc[:, KEYS].copy()
                predictions["predicted_y"] = (p_up > THRESHOLD).astype(int)
                result = evaluate_predictions(valid_rows, predictions)
                fold_scores[feature_set][seed][fold.name] = result["official_score"]
                config_id = f"{fixed['config_id']}__{feature_set}"
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
                    f"{feature_set}, {fold.name}, seed {seed}: "
                    f"{result['n_eval']} predictions; official score {result['official_score']:.8f}",
                    flush=True,
                )

    comparisons = []
    for feature_set in FEATURE_SETS:
        comparisons.append({
            "feature_set": feature_set,
            "config_id": f"{fixed['config_id']}__{feature_set}",
            "feature_order": list(F0_FEATURES) + (list(EXTRA_FEATURES) if feature_set != "F0_control" else []),
            "fold_official_scores_by_seed": fold_scores[feature_set],
            **e1_seed_summary(fold_scores[feature_set]),
        })
    control, extended = comparisons
    delta = extended["mean_cv_official_score"] - control["mean_cv_official_score"]
    summary = {
        "member": MEMBER,
        "model": MODEL,
        "experiment": EXPERIMENT,
        "experiment_type": "paired_F0_vs_F1_feature_test",
        "fixed_grid_config": fixed,
        "feature_definitions": {
            "prior_overnight_gap": "open(t-1) / close(t-2) - 1",
            "momentum_10d": "close(t-1) / close(t-11) - 1",
            "close_volatility_5d": "population std of five close-to-close returns ending at t-1",
        },
        "preprocessing": "BaselinePreprocessor for F0; training-fold median and standard scaling for extra F1 numeric fields",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "trial_budget": len(FEATURE_SETS),
        "fit_budget": len(FEATURE_SETS) * len(FOLDS) * len(E1_SEEDS),
        "comparisons": comparisons,
        "delta_f1_minus_f0_mean_cv_official_score": delta,
        "selected_feature_set": "F1_stock_history_v1" if delta > 0 else "F0_control",
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
    print(f"F1 minus F0 five-seed mean CV score: {delta:+.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--e2-summary", type=Path, default=DEFAULT_E2_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e2_features_v1")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.e2_summary, args.output_dir)
