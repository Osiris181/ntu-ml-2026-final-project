"""Compare four prespecified XGBoost E2 settings on the shared CV folds.

Run from the repository root:
    uv run python -m members.zhenci.e2.train_e2

This tunes XGBoost only. It uses the original historical F0, the same shared
preprocessing and target keys as E1, and never reads the 2026 holdout or the
Phase 1 submission score. It does not save a deployment model.
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


EXPERIMENT = "E2"
THRESHOLD = 0.5
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "lambda": 5.0,
    "alpha": 0.1,
    "nthread": 4,
}

# A fixed 2 x 2 search. Learning-rate/round pairs are prespecified together;
# only the shared CV official score chooses among these four configurations.
CANDIDATES = (
    {"config_id": "xgb_e2_d3_eta005_r300", "max_depth": 3, "eta": 0.05, "num_boost_round": 300},
    {"config_id": "xgb_e2_d5_eta005_r300", "max_depth": 5, "eta": 0.05, "num_boost_round": 300},
    {"config_id": "xgb_e2_d3_eta010_r200", "max_depth": 3, "eta": 0.10, "num_boost_round": 200},
    {"config_id": "xgb_e2_d5_eta010_r200", "max_depth": 5, "eta": 0.10, "num_boost_round": 200},
)


def select_best_config(summaries: list[dict]) -> str:
    """Use only the five-seed, equal-weight three-fold official CV mean.

    Ties use the prespecified candidate order, never a seed or holdout score.
    """
    if [item["config_id"] for item in summaries] != [item["config_id"] for item in CANDIDATES]:
        raise ValueError("Configuration summaries must follow the prespecified grid")
    values = np.asarray([item["mean_cv_official_score"] for item in summaries], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite E2 selection score")
    return summaries[int(np.argmax(values))]["config_id"]


def run(data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    metric_rows = []
    prediction_frames = []
    fold_scores = {
        candidate["config_id"]: {seed: {} for seed in E1_SEEDS}
        for candidate in CANDIDATES
    }

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)

        # The same F0 and baseline preprocessing as E1 isolate model tuning.
        preprocessor = BaselinePreprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation feature columns differ")
        dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy())
        dvalid = xgb.DMatrix(x_valid)

        for candidate in CANDIDATES:
            config_id = candidate["config_id"]
            for seed in E1_SEEDS:
                params = {
                    **BASE_PARAMS,
                    "max_depth": candidate["max_depth"],
                    "eta": candidate["eta"],
                    "seed": seed,
                }
                train_start = time.perf_counter()
                booster = xgb.train(
                    params, dtrain, num_boost_round=candidate["num_boost_round"]
                )
                train_seconds = time.perf_counter() - train_start
                predict_start = time.perf_counter()
                p_up = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - predict_start
                if len(p_up) != len(valid) or not np.isfinite(p_up).all() or not np.logical_and(p_up >= 0, p_up <= 1).all():
                    raise ValueError(f"Invalid probabilities for {config_id}, {fold.name}, seed {seed}")

                predictions = valid.loc[:, KEYS].copy()
                predictions["predicted_y"] = (p_up > THRESHOLD).astype(int)
                result = evaluate_predictions(valid_rows, predictions)
                fold_scores[config_id][seed][fold.name] = result["official_score"]
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
                    f"{config_id}, {fold.name}, seed {seed}: "
                    f"{result['n_eval']} predictions; official score {result['official_score']:.8f}",
                    flush=True,
                )

    config_summaries = []
    for candidate in CANDIDATES:
        config_id = candidate["config_id"]
        config_summaries.append({
            "config_id": config_id,
            "params": {**BASE_PARAMS, "max_depth": candidate["max_depth"], "eta": candidate["eta"]},
            "num_boost_round": candidate["num_boost_round"],
            "fold_official_scores_by_seed": fold_scores[config_id],
            **e1_seed_summary(fold_scores[config_id]),
        })
    selected_id = select_best_config(config_summaries)
    summary = {
        "member": MEMBER,
        "model": MODEL,
        "experiment": EXPERIMENT,
        "search_method": "prespecified_four_configuration_grid",
        "trial_budget": len(CANDIDATES),
        "fit_budget": len(CANDIDATES) * len(FOLDS) * len(E1_SEEDS),
        "feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "selection_rule": "highest five-seed mean of equal-weight 2023/2024/2025 official CV scores; candidate order breaks ties",
        "selected_config_id": selected_id,
        "configurations": config_summaries,
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
    selected = next(item for item in config_summaries if item["config_id"] == selected_id)
    print(f"Selected {selected_id}: five-seed mean CV score {selected['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e2_grid_v1"
    )
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir)
