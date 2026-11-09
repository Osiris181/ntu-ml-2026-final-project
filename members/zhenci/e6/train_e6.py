"""Test chronological inner early stopping against the frozen E3 CV control.

From the repository root:
    uv run python -m members.zhenci.e6.train_e6

Only historical data through 2025 enters training or validation. This script
does not evaluate the independent 2026 holdout or produce a submission.
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

from members.zhenci.e1.save_e1 import _sha256
from members.zhenci.e1.train_e1 import join_f0
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e4.train_e4 import (
    E1_CONFIG_ID, E3_RULE_ID, align_saved_probabilities, checked_source_score,
    load_control_summary, load_fixed_rule, load_saved_e1_predictions,
)
from members.zhenci.e5.train_e5_ablation import _auc
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, Split, split_market

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_E1_SUMMARY = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
DEFAULT_E1_PREDICTIONS = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"
DEFAULT_E3_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_xgboost_e6_nested_stop_v1"

# Each inner validation year ends before the corresponding outer validation
# year. The outer full training range is unchanged from the locked protocol.
INNER_SPLITS = {
    "2023": Split("inner_2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    "2024": Split("inner_2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    "2025": Split("inner_2024", "2023-12-31", "2024-01-01", "2024-12-31"),
}
MAX_ROUNDS = 300
PATIENCE = 20
CONFIGS = (
    {
        "config_id": "e6_e1_capacity_nested_stop",
        "overrides": {},
        "description": "Original E1 tree settings with an earlier-year-selected round count",
    },
    {
        "config_id": "e6_shallow_nested_stop",
        "overrides": {"max_depth": 3, "eta": 0.1, "min_child_weight": 20, "lambda": 5.0, "alpha": 0.1},
        "description": "Shallower, more regularized trees with an earlier-year-selected round count",
    },
)
RULES = ("taiex_rule", "threshold_0500")


def checked_inner_split(outer: Split) -> Split:
    inner = INNER_SPLITS.get(outer.name)
    if inner is None or inner.eval_end != outer.train_end:
        raise ValueError(f"No valid inner split for outer fold {outer.name}")
    if not (pd.Timestamp(inner.train_end) < pd.Timestamp(inner.eval_start)
            <= pd.Timestamp(inner.eval_end) < pd.Timestamp(outer.eval_start)):
        raise ValueError(f"Inner split overlaps outer validation year {outer.name}")
    return inner


def _score(rows: pd.DataFrame, labels: np.ndarray) -> dict:
    prediction = rows.loc[:, KEYS].copy()
    prediction["predicted_y"] = labels
    return evaluate_predictions(rows, prediction)


def _labels(p_up: np.ndarray, taiex: np.ndarray, rule: str, fixed_rule: dict) -> np.ndarray:
    if rule == "taiex_rule":
        return regime_predictions(p_up, taiex, fixed_rule)
    if rule == "threshold_0500":
        return (p_up > 0.5).astype(int)
    raise ValueError(f"Unknown rule: {rule}")


def _prepare_fold(train_rows: pd.DataFrame, valid_rows: pd.DataFrame, f0: pd.DataFrame):
    train = join_f0(train_rows, f0, include_label=True)
    valid = join_f0(valid_rows, f0, include_label=False)
    if not train.loc[:, KEYS].equals(train_rows.loc[:, KEYS]) or not valid.loc[:, KEYS].equals(valid_rows.loc[:, KEYS]):
        raise ValueError("F0 join changed split row order")
    processor = BaselinePreprocessor()
    x_train = processor.fit_transform(train.loc[:, list(F0_FEATURES)])
    x_valid = processor.transform(valid.loc[:, list(F0_FEATURES)])
    if not x_train.columns.equals(x_valid.columns):
        raise ValueError("Training and validation encoded feature columns differ")
    return train, valid, x_train, x_valid


def run(
    data_dir: Path, f0_path: Path, e1_summary_path: Path,
    e1_predictions_path: Path, e3_summary_path: Path, output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    e1_summary = load_control_summary(e1_summary_path)
    fixed_rule, e3_summary = load_fixed_rule(e3_summary_path, E1_CONFIG_ID)
    e3_recorded = next(row for row in e3_summary["comparisons"] if row["config_id"] == E3_RULE_ID)
    saved_e1 = load_saved_e1_predictions(e1_predictions_path, e1_summary)
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    if len(f0) != len(market) or not f0.loc[:, KEYS].equals(market.loc[:, KEYS]):
        raise ValueError("E6 requires the original historical F0 and market row keys")

    arms = ("f0_control__taiex_rule", "f0_control__threshold_0500", *(
        f"{config['config_id']}__{rule}" for config in CONFIGS for rule in RULES
    ))
    scores = {arm: {seed: {} for seed in E1_SEEDS} for arm in arms}
    metrics = []
    round_rows = []
    candidate_predictions = {config["config_id"]: [] for config in CONFIGS}
    control_predictions = []

    for outer in FOLDS:
        inner = checked_inner_split(outer)
        inner_train_rows, inner_valid_rows = split_market(market, target_ids, inner)
        outer_train_rows, outer_valid_rows = split_market(market, target_ids, outer)
        if set(inner_train_rows.stock_id) != set(all_ids) or set(outer_train_rows.stock_id) != set(all_ids):
            raise ValueError(f"Inner or outer training fold {outer.name} lacks some of the 100 stocks")
        if inner_valid_rows.date.max() >= outer_valid_rows.date.min():
            raise ValueError("Inner validation reaches the outer validation period")
        inner_train, inner_valid, x_inner_train, x_inner_valid = _prepare_fold(inner_train_rows, inner_valid_rows, f0)
        outer_train, outer_valid, x_outer_train, x_outer_valid = _prepare_fold(outer_train_rows, outer_valid_rows, f0)
        d_inner_train = xgb.DMatrix(x_inner_train, label=inner_train["y"].to_numpy())
        d_inner_valid = xgb.DMatrix(x_inner_valid, label=inner_valid_rows["y"].to_numpy())
        d_outer_train = xgb.DMatrix(x_outer_train, label=outer_train["y"].to_numpy())
        d_outer_valid = xgb.DMatrix(x_outer_valid)
        target_mask = outer_train.stock_id.isin(target_ids).to_numpy()
        d_outer_target_train = xgb.DMatrix(x_outer_train.loc[target_mask])
        outer_target_truth = outer_train.loc[target_mask, "y"].to_numpy(dtype=int)
        outer_truth = outer_valid_rows["y"].to_numpy(dtype=int)
        taiex = outer_valid["taiex_return"].to_numpy(dtype=float)

        for seed in E1_SEEDS:
            control_p = align_saved_probabilities(outer_valid_rows, saved_e1, outer.name, seed)
            row = outer_valid_rows.loc[:, KEYS].copy()
            row.insert(0, "seed", seed)
            row.insert(0, "fold", outer.name)
            row["p_up"] = control_p
            for rule in RULES:
                arm = f"f0_control__{rule}"
                labels = _labels(control_p, taiex, rule, fixed_rule)
                result = _score(outer_valid_rows, labels)
                scores[arm][seed][outer.name] = result["official_score"]
                metrics.append({
                    "member": e1_summary["member"], "model": "XGBoost",
                    "experiment": "E1" if rule == "threshold_0500" else "E2",
                    "config_id": arm, "fold": outer.name, "seed": seed,
                    **result, "train_seconds": None, "predict_seconds": None,
                    "training_target_auc": None,
                    "validation_auc": _auc(outer_truth, control_p),
                    "predicted_up_rate": float(labels.mean()),
                })
                recorded = e3_recorded if rule == "taiex_rule" else e1_summary
                checked_source_score(recorded["fold_official_scores_by_seed"], seed, outer.name,
                                     result["official_score"], "E3" if rule == "taiex_rule" else "E1")
                row[f"predicted_y_{rule}"] = labels
            control_predictions.append(row)

        for config in CONFIGS:
            config_id = config["config_id"]
            for seed in E1_SEEDS:
                params = {**e1_summary["params"], **config["overrides"], "seed": seed}
                started = time.perf_counter()
                inner_booster = xgb.train(
                    params, d_inner_train, num_boost_round=MAX_ROUNDS,
                    evals=[(d_inner_valid, "inner_target")],
                    early_stopping_rounds=PATIENCE, verbose_eval=False,
                )
                if not hasattr(inner_booster, "best_iteration") or not np.isfinite(inner_booster.best_score):
                    raise ValueError(f"Inner early stopping failed for {config_id}, {outer.name}, seed {seed}")
                selected_rounds = int(inner_booster.best_iteration) + 1
                if not 1 <= selected_rounds <= MAX_ROUNDS:
                    raise ValueError("Selected boosting round count is out of range")
                inner_seconds = time.perf_counter() - started
                started = time.perf_counter()
                booster = xgb.train(params, d_outer_train, num_boost_round=selected_rounds)
                refit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_valid = booster.predict(d_outer_valid)
                p_train_target = booster.predict(d_outer_target_train)
                predict_seconds = time.perf_counter() - started
                for name, p, expected in (("validation", p_valid, len(outer_valid_rows)),
                                           ("training", p_train_target, int(target_mask.sum()))):
                    if len(p) != expected or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise ValueError(f"Invalid {name} probabilities for {config_id}, {outer.name}, seed {seed}")
                train_auc = _auc(outer_target_truth, p_train_target)
                valid_auc = _auc(outer_truth, p_valid)
                round_rows.append({
                    "config_id": config_id, "fold": outer.name, "seed": seed,
                    "inner_validation_year": inner.name, "selected_rounds": selected_rounds,
                    "inner_best_logloss": float(inner_booster.best_score),
                    "inner_train_seconds": inner_seconds, "outer_refit_seconds": refit_seconds,
                })
                row = outer_valid_rows.loc[:, KEYS].copy()
                row.insert(0, "seed", seed)
                row.insert(0, "fold", outer.name)
                row["p_up"] = p_valid
                for rule in RULES:
                    arm = f"{config_id}__{rule}"
                    labels = _labels(p_valid, taiex, rule, fixed_rule)
                    result = _score(outer_valid_rows, labels)
                    scores[arm][seed][outer.name] = result["official_score"]
                    metrics.append({
                        "member": e1_summary["member"], "model": "XGBoost", "experiment": "E2",
                        "config_id": arm, "fold": outer.name, "seed": seed,
                        **result, "train_seconds": inner_seconds + refit_seconds,
                        "predict_seconds": predict_seconds,
                        "training_target_auc": train_auc, "validation_auc": valid_auc,
                        "predicted_up_rate": float(labels.mean()),
                    })
                    row[f"predicted_y_{rule}"] = labels
                candidate_predictions[config_id].append(row)
            print(f"{config_id}, outer {outer.name}, inner {inner.name}: completed five seeds", flush=True)

    comparisons = []
    for arm in arms:
        config_id, rule = arm.split("__", 1)
        comparisons.append({
            "arm": arm, "config_id": config_id, "rule": rule,
            "fold_official_scores_by_seed": scores[arm], **e1_seed_summary(scores[arm]),
        })
    selected = max(comparisons, key=lambda row: row["mean_cv_official_score"])
    incumbent = comparisons[0]
    selected_frames = control_predictions if selected["config_id"] == "f0_control" else candidate_predictions[selected["config_id"]]
    selected_predictions = pd.concat(selected_frames, ignore_index=True).loc[
        :, [*KEYS, "fold", "seed", "p_up", f"predicted_y_{selected['rule']}"]
    ].rename(columns={f"predicted_y_{selected['rule']}": "predicted_y"})
    selected_predictions.insert(0, "config_id", selected["arm"])
    selected_predictions.insert(0, "experiment", "E1" if selected["arm"] == "f0_control__threshold_0500" else "E2")
    selected_predictions.insert(0, "model", "XGBoost")
    selected_predictions.insert(0, "member", e1_summary["member"])

    summary = {
        "experiment_label": "E6 chronological inner early stopping",
        "experiment": "E2", "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "preprocessing": "shared BaselinePreprocessor fitted on inner training rows for round selection, then separately on full outer training rows for refit",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "inner_splits": {outer.name: vars(checked_inner_split(outer)) for outer in FOLDS},
        "round_selection": f"inner target-stock logloss, max {MAX_ROUNDS} rounds, patience {PATIENCE}; refit selected rounds on full outer training set",
        "configs": CONFIGS, "fixed_e3_rule": fixed_rule,
        "source_e1_summary": str(e1_summary_path), "source_e1_summary_sha256": _sha256(e1_summary_path),
        "source_e1_predictions": str(e1_predictions_path), "source_e1_predictions_sha256": _sha256(e1_predictions_path),
        "source_e3_summary": str(e3_summary_path), "source_e3_summary_sha256": _sha256(e3_summary_path),
        "source_f0": str(f0_path), "source_f0_sha256": _sha256(f0_path),
        "n_inner_fits": len(CONFIGS) * len(FOLDS) * len(E1_SEEDS),
        "n_outer_refits": len(CONFIGS) * len(FOLDS) * len(E1_SEEDS),
        "comparisons": comparisons,
        "incumbent_arm": incumbent["arm"],
        "incumbent_mean_cv_official_score": incumbent["mean_cv_official_score"],
        "selected_arm_by_cv": selected["arm"],
        "selected_mean_cv_official_score": selected["mean_cv_official_score"],
        "improvement_over_e3": selected["mean_cv_official_score"] - incumbent["mean_cv_official_score"],
        "e6_beats_e3": selected["config_id"] != "f0_control" and selected["mean_cv_official_score"] > incumbent["mean_cv_official_score"],
        "selection_method": "Mean of three target-stock outer-year official scores within each seed, then mean of five seeds; predeclared arm order breaks ties",
        "auc_note": "Training target-stock AUC is in-sample and diagnostic; only outer validation official score selects an arm",
        "status": "validation only; no deployment model or submission",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics).loc[:, [*METRIC_FIELDS, "training_target_auc", "validation_auc", "predicted_up_rate"]].to_csv(
        output_dir / "fold_metrics.csv", index=False,
    )
    pd.DataFrame(round_rows).to_csv(output_dir / "selected_rounds.csv", index=False)
    selected_predictions.loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "selected_validation_predictions.csv", index=False,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    for comparison in comparisons:
        print(f"{comparison['arm']}: mean CV official score {comparison['mean_cv_official_score']:.8f}")
    print(f"Selected {selected['arm']}; E6 beats E3: {summary['e6_beats_e3']}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_E1_SUMMARY)
    parser.add_argument("--e1-predictions", type=Path, default=DEFAULT_E1_PREDICTIONS)
    parser.add_argument("--e3-summary", type=Path, default=DEFAULT_E3_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.e1_summary, args.e1_predictions, args.e3_summary, args.output_dir)
