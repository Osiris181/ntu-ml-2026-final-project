"""Tune XGBoost on F0 with Optuna, using only the agreed chronological CV.

Run from the repository root:
    uv run python -m members.zhenci.train_e3

The first two trials reproduce full-sample and sampled E1 settings. The
remaining trials search XGBoost parameters. No holdout, updated data,
submission feedback, deployment refit, or submission is used here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from members.zhenci.train_e1 import MEMBER, MODEL, join_f0
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market


# "E3" is the member's third experiment; the shared result schema calls all
# optimized experiments E2. The two anchors make the baseline comparison exact.
EXPERIMENT = "E2"
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "alpha": 0.0,
    "nthread": 4,
}
ANCHORS = (
    {"max_depth": 6, "eta": 0.3, "num_boost_round": 100,
     "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "lambda": 1.0},
    {"max_depth": 6, "eta": 0.3, "num_boost_round": 100,
     "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0},
)


def suggested_params(trial: optuna.Trial) -> dict:
    """Keep both E1 anchor settings inside the declared search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "eta": trial.suggest_float("eta", 0.03, 0.3, log=True),
        "num_boost_round": trial.suggest_int("num_boost_round", 100, 300, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
    }


def mean_importance(models: list[dict[str, float]], feature_names: list[str]) -> pd.DataFrame:
    """Average within-model normalized total gain across 15 fold/seed fits."""
    if len(models) != len(FOLDS) * len(E1_SEEDS):
        raise ValueError("Importance requires every fold and seed")
    all_names = set(feature_names)
    if any(set(scores) - all_names for scores in models):
        raise ValueError("Unexpected XGBoost feature name in importance")
    normalized = []
    for scores in models:
        values = np.array([scores.get(name, 0.0) for name in feature_names], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid feature gain")
        total = values.sum()
        normalized.append(values / total if total > 0 else np.zeros_like(values))
    means = np.stack(normalized).mean(axis=0)
    logical = [name.split("=", 1)[0] if name.startswith(("stock_id=", "day_of_week=")) else name
               for name in feature_names]
    return pd.DataFrame({"encoded_feature": feature_names, "logical_feature": logical,
                         "mean_normalized_total_gain": means}).sort_values(
                             "mean_normalized_total_gain", ascending=False, kind="stable"
                         ).reset_index(drop=True)


def run(data_dir: Path, f0_path: Path, output_dir: Path, n_trials: int) -> Path:
    if n_trials < len(ANCHORS):
        raise ValueError("At least two trials are needed for the E1 anchors")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    fold_data = []
    feature_names = None
    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        train = join_f0(train_rows, f0, include_label=True)
        valid = join_f0(valid_rows, f0, include_label=False)
        preprocessor = BaselinePreprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
        x_valid = preprocessor.transform(valid.loc[:, list(F0_FEATURES)])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError("Training and validation feature columns differ")
        names = x_train.columns.tolist()
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("Encoded feature order differs between folds")
        fold_data.append((fold.name, xgb.DMatrix(x_train, label=train["y"].to_numpy()),
                          xgb.DMatrix(x_valid), valid_rows, valid.loc[:, KEYS]))

    metrics = []
    trial_summaries = []
    best_predictions = None
    best_importance = None
    best_score = -np.inf
    best_number = None
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    for index, anchor in enumerate(ANCHORS):
        study.enqueue_trial(anchor, user_attrs={"anchor": "full_e1" if index == 0 else "sampled_e1"})

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_predictions, best_importance, best_score, best_number
        choice = suggested_params(trial)
        config_id = f"xgb_e3_trial_{trial.number:03d}"
        scores = {seed: {} for seed in E1_SEEDS}
        trial_predictions = []
        gains = []
        trial_metrics = []
        for fold_name, dtrain, dvalid, valid_rows, valid_keys in fold_data:
            for seed in E1_SEEDS:
                params = {**BASE_PARAMS, **{key: value for key, value in choice.items()
                                           if key != "num_boost_round"}, "seed": seed}
                started = time.perf_counter()
                booster = xgb.train(params, dtrain, num_boost_round=choice["num_boost_round"])
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_up = booster.predict(dvalid)
                predict_seconds = time.perf_counter() - started
                if len(p_up) != len(valid_keys) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
                    raise ValueError(f"Invalid probabilities for {config_id}, {fold_name}, seed {seed}")
                predictions = valid_keys.copy()
                predictions["predicted_y"] = (p_up > 0.5).astype(int)
                result = evaluate_predictions(valid_rows, predictions)
                scores[seed][fold_name] = result["official_score"]
                trial_metrics.append({"member": MEMBER, "model": MODEL,
                                      "experiment": EXPERIMENT, "config_id": config_id,
                                      "fold": fold_name, "seed": seed, **result,
                                      "train_seconds": train_seconds,
                                      "predict_seconds": predict_seconds})
                predictions.insert(0, "seed", seed)
                predictions.insert(0, "fold", fold_name)
                predictions.insert(0, "config_id", config_id)
                predictions.insert(0, "experiment", EXPERIMENT)
                predictions.insert(0, "model", MODEL)
                predictions.insert(0, "member", MEMBER)
                predictions["p_up"] = p_up
                trial_predictions.append(predictions)
                gains.append(booster.get_score(importance_type="total_gain"))
        selection = e1_seed_summary(scores)
        value = selection["mean_cv_official_score"]
        metrics.extend(trial_metrics)
        trial_summaries.append({"trial_number": trial.number, "config_id": config_id,
                                "anchor": trial.user_attrs.get("anchor"),
                                "params": {**BASE_PARAMS, **choice},
                                "fold_official_scores_by_seed": scores, **selection})
        if value > best_score:
            best_score = value
            best_number = trial.number
            best_predictions = pd.concat(trial_predictions, ignore_index=True)
            best_importance = mean_importance(gains, feature_names)
        print(f"Trial {trial.number + 1}/{n_trials}: mean CV {value:.8f}; best {best_score:.8f}", flush=True)
        return value

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    if best_number != study.best_trial.number:
        raise ValueError("Optuna's best trial and recorded predictions differ")
    selected = trial_summaries[best_number]
    summary = {
        "member": MEMBER, "model": MODEL, "experiment": EXPERIMENT,
        "experiment_label": "E3 Optuna F0 tuning",
        "search_method": "Optuna TPESampler(seed=42), sequential, two enqueued E1 anchors",
        "trial_budget": n_trials, "fit_budget": n_trials * len(FOLDS) * len(E1_SEEDS),
        "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "preprocessing": "shared.preprocessing.BaselinePreprocessor; fitted separately per fold",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "selection_rule": "highest five-seed mean of equal-weight 2023/2024/2025 official CV scores; first trial breaks ties",
        "selected_config_id": selected["config_id"],
        "selected_trial_number": best_number,
        "full_e1_cv_score": trial_summaries[0]["mean_cv_official_score"],
        "improvement_over_full_e1": best_score - trial_summaries[0]["mean_cv_official_score"],
        "trials": trial_summaries,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "xgboost": xgb.__version__,
                     "optuna": optuna.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics, columns=METRIC_FIELDS).to_csv(output_dir / "fold_metrics.csv", index=False)
    best_predictions.loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "selected_validation_predictions.csv", index=False)
    best_importance.to_csv(output_dir / "selected_feature_importance.csv", index=False)
    best_importance.groupby("logical_feature", as_index=False)["mean_normalized_total_gain"].sum().sort_values(
        "mean_normalized_total_gain", ascending=False, kind="stable"
    ).to_csv(output_dir / "selected_logical_feature_importance.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Selected {selected['config_id']}: {best_score:.8f}; full E1: {summary['full_e1_cv_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e3_optuna_v1")
    parser.add_argument("--trials", type=int, default=20, help="Total trials, including two E1 anchors (default: 20)")
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.output_dir, args.trials)
