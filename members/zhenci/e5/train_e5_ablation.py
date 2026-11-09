"""Compare prespecified F0 feature removals with the existing E3 control.

Run from the repository root with no flags:
    uv run python -m members.zhenci.e5.train_e5_ablation

This is validation only. It does not use the 2026 holdout, updated market
files, submission outcomes, or train a deployment model.
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
    E1_CONFIG_ID,
    E3_RULE_ID,
    checked_source_score,
    load_control_summary,
    load_fixed_rule,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import CATEGORICAL_FEATURES, F0_FEATURES, NUMERIC_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, FOLDS, split_market

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_E1_SUMMARY = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
DEFAULT_E3_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_xgboost_e5_f0_ablation_v1"

# These were chosen from the read-only F0 drift, redundancy, and model-gain
# audit before this experiment. Keep this list fixed when interpreting results.
ABLATIONS = (
    ("f0_control", ()),
    ("drop_relative_amount", ("relative_amount_20d",)),
    ("drop_rsi", ("rsi_normalized",)),
    ("drop_sma_distance", ("sma_distance",)),
    ("drop_nasdaq", ("nasdaq_return",)),
    ("drop_sp500", ("sp500_return",)),
    ("drop_atr", ("atr_normalized",)),
    ("drop_volatility_group", ("price_range_1d", "atr_normalized", "bb_width")),
)
RULES = ("taiex_rule", "threshold_0500")


def retained_columns(encoded_columns: list[str], omitted: tuple[str, ...]) -> list[str]:
    """Remove whole F0 logical fields after shared fold-fitted preprocessing."""
    if len(omitted) != len(set(omitted)) or set(omitted) - set(NUMERIC_FEATURES):
        raise ValueError("Ablations may remove only distinct numeric F0 fields")
    if "taiex_return" in omitted:
        raise ValueError("The fixed E3 decision rule needs taiex_return")
    if set(F0_FEATURES) - set(encoded_columns) - set(CATEGORICAL_FEATURES):
        raise ValueError("Preprocessed F0 numeric columns are incomplete")
    kept = [name for name in encoded_columns if name not in omitted]
    if len(kept) != len(encoded_columns) - len(omitted):
        raise ValueError("An ablated numeric field was absent from preprocessing")
    return kept


def _labels(probabilities: np.ndarray, taiex: np.ndarray, rule: str, fixed_rule: dict) -> np.ndarray:
    if rule == "taiex_rule":
        return regime_predictions(probabilities, taiex, fixed_rule)
    if rule == "threshold_0500":
        return (probabilities > 0.5).astype(int)
    raise ValueError(f"Unknown prediction rule: {rule}")


def _score(market_rows: pd.DataFrame, labels: np.ndarray) -> dict:
    predictions = market_rows.loc[:, KEYS].copy()
    predictions["predicted_y"] = labels
    return evaluate_predictions(market_rows, predictions)


def _auc(truth: np.ndarray, probabilities: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied probabilities."""
    positive = np.asarray(truth) == 1
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC needs both up and down outcomes")
    ranks = pd.Series(probabilities).rank(method="average").to_numpy(dtype=float)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _logical_gain(booster: xgb.Booster, retained: list[str]) -> list[dict]:
    gains = booster.get_score(importance_type="total_gain")
    if set(gains) - set(retained):
        raise ValueError("XGBoost reported an unexpected feature name")
    total = float(sum(gains.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Model has no finite feature gain")
    logical = {name: 0.0 for name in F0_FEATURES}
    for name, gain in gains.items():
        field = name.split("=", 1)[0] if name.startswith(("stock_id=", "day_of_week=")) else name
        if field not in logical or not np.isfinite(gain) or gain < 0:
            raise ValueError("Invalid XGBoost feature gain")
        logical[field] += float(gain) / total
    return [{"logical_feature": name, "normalized_total_gain": value} for name, value in logical.items()]


def run(
    data_dir: Path,
    f0_path: Path,
    e1_summary_path: Path,
    e3_summary_path: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    e1_summary = load_control_summary(e1_summary_path)
    fixed_rule, e3_summary = load_fixed_rule(e3_summary_path, E1_CONFIG_ID)
    e3_recorded = next(row for row in e3_summary["comparisons"] if row["config_id"] == E3_RULE_ID)
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    if len(f0) != len(market) or not f0.loc[:, KEYS].equals(market.loc[:, KEYS]):
        raise ValueError("E5 requires the original historical F0 with exactly the market row keys")

    comparisons = []
    metrics = []
    gain_rows = []
    selected_score = -np.inf
    selected_arm = None
    selected_predictions = None

    for config_id, omitted in ABLATIONS:
        scores = {rule: {seed: {} for seed in E1_SEEDS} for rule in RULES}
        prediction_frames = []
        for fold in FOLDS:
            train_rows, valid_rows = split_market(market, target_ids, fold)
            if set(train_rows["stock_id"]) != set(all_ids):
                raise ValueError(f"Training fold {fold.name} does not cover all 100 stocks")
            train = join_f0(train_rows, f0, include_label=True)
            valid = join_f0(valid_rows, f0, include_label=False)
            if not train.loc[:, KEYS].equals(train_rows.loc[:, KEYS]) or not valid.loc[:, KEYS].equals(valid_rows.loc[:, KEYS]):
                raise ValueError(f"F0 join changed row order in fold {fold.name}")
            processor = BaselinePreprocessor()
            x_train_full = processor.fit_transform(train.loc[:, list(F0_FEATURES)])
            x_valid_full = processor.transform(valid.loc[:, list(F0_FEATURES)])
            if not x_train_full.columns.equals(x_valid_full.columns):
                raise ValueError(f"Encoded feature columns differ in {fold.name}")
            kept = retained_columns(x_train_full.columns.tolist(), omitted)
            x_train = x_train_full.loc[:, kept]
            x_valid = x_valid_full.loc[:, kept]
            target_mask = train["stock_id"].isin(target_ids).to_numpy()
            train_target_rows = train_rows.loc[target_mask].reset_index(drop=True)
            if len(train_target_rows) == 0:
                raise ValueError("No target-stock training rows")
            dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy())
            dvalid = xgb.DMatrix(x_valid)
            dtrain_target = xgb.DMatrix(x_train.loc[target_mask])
            valid_taiex = valid["taiex_return"].to_numpy(dtype=float)
            train_taiex = train.loc[target_mask, "taiex_return"].to_numpy(dtype=float)
            train_truth = train.loc[target_mask, "y"].to_numpy(dtype=int)
            valid_truth = valid_rows["y"].to_numpy(dtype=int)

            for seed in E1_SEEDS:
                started = time.perf_counter()
                booster = xgb.train(
                    {**e1_summary["params"], "seed": seed}, dtrain,
                    num_boost_round=e1_summary["num_boost_round"],
                )
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_valid = booster.predict(dvalid)
                p_train_target = booster.predict(dtrain_target)
                predict_seconds = time.perf_counter() - started
                for name, values, expected in (
                    ("validation", p_valid, len(valid_rows)),
                    ("training", p_train_target, len(train_target_rows)),
                ):
                    if len(values) != expected or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                        raise ValueError(f"Invalid {name} probabilities for {config_id}, {fold.name}, seed {seed}")
                training_auc = _auc(train_truth, p_train_target)
                validation_auc = _auc(valid_truth, p_valid)
                for item in _logical_gain(booster, kept):
                    gain_rows.append({"config_id": config_id, "fold": fold.name, "seed": seed, **item})

                one_run = valid_rows.loc[:, KEYS].copy()
                one_run.insert(0, "seed", seed)
                one_run.insert(0, "fold", fold.name)
                one_run["p_up"] = p_valid
                for rule in RULES:
                    arm = f"{config_id}__{rule}"
                    labels = _labels(p_valid, valid_taiex, rule, fixed_rule)
                    training_labels = _labels(p_train_target, train_taiex, rule, fixed_rule)
                    result = _score(valid_rows, labels)
                    training_result = _score(train_target_rows, training_labels)
                    scores[rule][seed][fold.name] = result["official_score"]
                    metrics.append({
                        "member": e1_summary["member"], "model": "XGBoost",
                        "experiment": "E1" if arm == "f0_control__threshold_0500" else "E2",
                        "config_id": arm, "fold": fold.name, "seed": seed,
                        **result, "train_seconds": train_seconds, "predict_seconds": predict_seconds,
                        "training_target_official_score": training_result["official_score"],
                        "training_target_auc": training_auc, "validation_auc": validation_auc,
                        "predicted_up_rate": float(labels.mean()),
                    })
                    one_run[f"predicted_y_{rule}"] = labels
                    if config_id == "f0_control":
                        recorded = (e3_recorded if rule == "taiex_rule" else e1_summary)
                        checked_source_score(
                            recorded["fold_official_scores_by_seed"], seed, fold.name,
                            result["official_score"], "E3" if rule == "taiex_rule" else "E1",
                        )
                prediction_frames.append(one_run)
            print(f"{config_id}, {fold.name}: completed {len(E1_SEEDS)} seeds", flush=True)

        variant_predictions = pd.concat(prediction_frames, ignore_index=True)
        for rule in RULES:
            arm = f"{config_id}__{rule}"
            selection = e1_seed_summary(scores[rule])
            value = selection["mean_cv_official_score"]
            comparisons.append({
                "arm": arm, "config_id": config_id, "rule": rule,
                "omitted_features": list(omitted),
                "retained_logical_features": [name for name in F0_FEATURES if name not in omitted],
                "fold_official_scores_by_seed": scores[rule], **selection,
            })
            if value > selected_score:
                selected_score = value
                selected_arm = arm
                selected_predictions = variant_predictions.loc[:, [*KEYS, "fold", "seed", "p_up", f"predicted_y_{rule}"]].rename(
                    columns={f"predicted_y_{rule}": "predicted_y"}
                ).copy()
            print(f"  {arm}: mean CV official score {value:.8f}", flush=True)

    incumbent = next(row for row in comparisons if row["arm"] == "f0_control__taiex_rule")
    if selected_predictions is None or selected_arm is None:
        raise RuntimeError("No ablation comparison completed")
    selected_config = selected_arm.split("__", 1)[0]
    selected_predictions.insert(0, "config_id", selected_arm)
    selected_predictions.insert(0, "experiment", "E1" if selected_arm == "f0_control__threshold_0500" else "E2")
    selected_predictions.insert(0, "model", "XGBoost")
    selected_predictions.insert(0, "member", e1_summary["member"])

    gain_frame = pd.DataFrame(gain_rows)
    gain_summary = gain_frame.groupby(["config_id", "logical_feature"], as_index=False).agg(
        mean_normalized_total_gain=("normalized_total_gain", "mean"),
        std_normalized_total_gain=("normalized_total_gain", "std"),
    )
    gain_summary["std_normalized_total_gain"] = gain_summary["std_normalized_total_gain"].fillna(0.0)
    omitted_by_config = {config_id: set(omitted) for config_id, omitted in ABLATIONS}
    gain_summary["omitted_from_model"] = [
        feature in omitted_by_config[config_id]
        for config_id, feature in zip(gain_summary["config_id"], gain_summary["logical_feature"])
    ]
    summary = {
        "experiment_label": "E5 prespecified F0 feature ablation",
        "experiment": "E2", "source_feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "preprocessing": "shared BaselinePreprocessor refitted on each fold's training rows for each variant; omitted fields excluded only from XGBoost input",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "source_e1_summary": str(e1_summary_path), "source_e1_summary_sha256": _sha256(e1_summary_path),
        "source_e3_summary": str(e3_summary_path), "source_e3_summary_sha256": _sha256(e3_summary_path),
        "source_f0": str(f0_path), "source_f0_sha256": _sha256(f0_path),
        "params": e1_summary["params"], "num_boost_round": e1_summary["num_boost_round"],
        "fixed_e3_rule": fixed_rule,
        "model_fit_budget": len(ABLATIONS) * len(FOLDS) * len(E1_SEEDS),
        "comparisons": comparisons,
        "incumbent_arm": incumbent["arm"],
        "incumbent_mean_cv_official_score": incumbent["mean_cv_official_score"],
        "selected_arm_by_cv": selected_arm,
        "selected_mean_cv_official_score": selected_score,
        "selected_model_omitted_features": list(next(omitted for config_id, omitted in ABLATIONS if config_id == selected_config)),
        "improvement_over_e3": selected_score - incumbent["mean_cv_official_score"],
        "ablation_beats_e3": selected_score > incumbent["mean_cv_official_score"],
        "selection_method": "Mean of three target-stock year scores within each seed, then mean of five seeds; predeclared comparison order breaks ties",
        "training_metric_note": "Training target-stock scores and AUC are in-sample diagnostics, not selection criteria; validation uses future years",
        "importance_note": "Gain measures split usage, not causal value; compare candidate official CV scores before removing features",
        "status": "validation only; no saved deployment model or submission",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics).loc[:, [*METRIC_FIELDS, "training_target_official_score", "training_target_auc", "validation_auc", "predicted_up_rate"]].to_csv(
        output_dir / "fold_metrics.csv", index=False,
    )
    gain_summary.to_csv(output_dir / "logical_feature_importance.csv", index=False)
    selected_predictions.loc[:, [*PREDICTION_FIELDS, "p_up"]].to_csv(
        output_dir / "selected_validation_predictions.csv", index=False,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Selected {selected_arm}: {selected_score:.8f}; E3 control {incumbent['mean_cv_official_score']:.8f}")
    print(f"Ablation beats E3: {summary['ablation_beats_e3']}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_E1_SUMMARY)
    parser.add_argument("--e3-summary", type=Path, default=DEFAULT_E3_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.e1_summary, args.e3_summary, args.output_dir)
