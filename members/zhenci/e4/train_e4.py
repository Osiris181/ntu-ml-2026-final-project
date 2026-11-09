"""Compare the E4 F2 XGBoost model with its saved E1 control on the CV folds.

From the repository root, after building F2:
    uv run python -m members.zhenci.e4.train_e4

The four predeclared arms share validation keys and five seeds. This script
does not inspect the independent 2026 holdout, refit a deployment model,
generate a submission, or use competition feedback.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.save_e1 import _read_cv_summary, _sha256
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from members.zhenci.e4.features_e4 import F2_DATASET_COLUMNS, F2_EXTRA_FEATURES, load_f2_dataset
from members.zhenci.e4.preprocessing_e4 import F2Preprocessor
from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import METRIC_FIELDS, PREDICTION_FIELDS, e1_seed_summary, evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, NUMERIC_FEATURES
from shared.splits import E1_SEEDS, FOLDS, split_market

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_F2_PATH = PROJECT_ROOT / "data/processed/f2.csv"
DEFAULT_E1_SUMMARY = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
DEFAULT_E1_PREDICTIONS = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"
DEFAULT_E3_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_xgboost_e4_f2_v1"

E1_CONFIG_ID = "xgboost_initial_v1"
E3_RULE_ID = "e3_taiex_negative_0525_nonnegative_0475"
E4_CONFIG_ID = "xgboost_f2_original_e1_params_v1"
ARMS = (
    "e1_f0_0500",
    "e1_f0_taiex_rule",
    "e4_f2_0500",
    "e4_f2_taiex_rule",
)
ARM_CONFIG_IDS = {
    "e1_f0_0500": E1_CONFIG_ID,
    "e1_f0_taiex_rule": E3_RULE_ID,
    "e4_f2_0500": E4_CONFIG_ID,
    "e4_f2_taiex_rule": f"{E4_CONFIG_ID}__{E3_RULE_ID}",
}
EXPECTED_E1_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "nthread": 4,
}


def load_control_summary(path: Path) -> dict:
    """Pin the archived, unsampled E1 configuration and its shared protocol."""
    summary = _read_cv_summary(path)
    if summary["config_id"] != E1_CONFIG_ID:
        raise ValueError(f"E4 requires the original full-sample E1 configuration {E1_CONFIG_ID}")
    if summary["params"] != EXPECTED_E1_PARAMS or summary["num_boost_round"] != 100:
        raise ValueError("E1 parameters differ from the predeclared full-sample E4 control")
    if not isinstance(summary.get("fold_official_scores_by_seed"), dict):
        raise ValueError("E1 summary lacks fold/seed scores")
    return summary


def load_fixed_rule(path: Path, e1_config_id: str) -> tuple[dict, dict]:
    """Use the already selected E3 rule, with no E4 threshold search."""
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    rule = selected_rule(summary, e1_config_id)
    expected = {
        "config_id": E3_RULE_ID,
        "negative": 0.525,
        "nonnegative": 0.475,
        "missing": 0.5,
    }
    if rule != expected:
        raise ValueError("The E3 selected TAIEX rule differs from the fixed E4 comparison")
    if summary.get("seeds") != list(E1_SEEDS):
        raise ValueError("E3 threshold study used a different seed schedule")
    return rule, summary


def load_saved_e1_predictions(path: Path, e1_summary: dict) -> pd.DataFrame:
    """Check the saved E1 probabilities before reusing them as paired controls."""
    source = pd.read_csv(path, dtype={"stock_id": str, "fold": str}, parse_dates=["date"])
    required = {*KEYS, "member", "model", "experiment", "config_id", "fold", "seed", "predicted_y", "p_up"}
    absent = required - set(source.columns)
    if absent:
        raise ValueError(f"Saved E1 predictions lack: {sorted(absent)}")
    if source.empty or source[KEYS + ["fold", "seed"]].isna().any().any():
        raise ValueError("Saved E1 predictions contain missing keys or are empty")
    if not pd.api.types.is_datetime64_any_dtype(source["date"]):
        raise ValueError("Saved E1 prediction dates must be parsed calendar dates")
    if not source["member"].eq(e1_summary["member"]).all() or not source["model"].eq("XGBoost").all():
        raise ValueError("Saved E1 predictions belong to a different model or member")
    if not source["experiment"].eq("E1").all() or not source["config_id"].eq(E1_CONFIG_ID).all():
        raise ValueError("Saved E1 predictions use the wrong experiment or configuration")
    if set(source["fold"]) != {fold.name for fold in FOLDS}:
        raise ValueError("Saved E1 predictions must cover the three agreed folds")
    numeric_seeds = pd.to_numeric(source["seed"], errors="raise")
    if not np.isfinite(numeric_seeds.to_numpy(dtype=float)).all() or not numeric_seeds.isin(E1_SEEDS).all():
        raise ValueError("Saved E1 predictions use an invalid seed")
    source["seed"] = numeric_seeds.astype(int)
    if set(source["seed"]) != set(E1_SEEDS):
        raise ValueError("Saved E1 predictions must cover all five seeds")
    if source.duplicated(["fold", "seed", *KEYS]).any():
        raise ValueError("Saved E1 predictions contain duplicate fold/seed/stock/date keys")
    probabilities = pd.to_numeric(source["p_up"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Saved E1 probabilities must be finite and in [0, 1]")
    if source["predicted_y"].isna().any() or not source["predicted_y"].isin([0, 1]).all():
        raise ValueError("Saved E1 predictions must be binary")
    if not np.array_equal(source["predicted_y"].to_numpy(dtype=int), (probabilities > 0.5).astype(int)):
        raise ValueError("Saved E1 labels disagree with their recorded 0.5 probabilities")
    source["p_up"] = probabilities
    return source


def align_saved_probabilities(
    valid_rows: pd.DataFrame, source: pd.DataFrame, fold_name: str, seed: int,
) -> np.ndarray:
    """Require exact target-key coverage and return probabilities in split order."""
    subset = source.loc[source["fold"].eq(fold_name) & source["seed"].eq(seed), [*KEYS, "p_up"]]
    validate_keys(subset)
    aligned = valid_rows.loc[:, KEYS].merge(
        subset, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if len(subset) != len(valid_rows) or not aligned["_merge"].eq("both").all():
        raise ValueError(f"Saved E1 predictions do not exactly cover fold {fold_name}, seed {seed}")
    return aligned["p_up"].to_numpy(dtype=float)


def join_f2(rows: pd.DataFrame, f2: pd.DataFrame, *, include_label: bool) -> pd.DataFrame:
    columns = [*KEYS, *(["y"] if include_label else [])]
    joined = rows.loc[:, columns].merge(
        f2, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Missing F2 rows for {int(joined['_merge'].ne('both').sum())} split stock/date keys")
    return joined.drop(columns="_merge")


def check_f2_matches_f0(f2: pd.DataFrame, f0: pd.DataFrame) -> None:
    """Refuse an F2 file that changed shared F0 keys or feature values."""
    if len(f2) != len(f0):
        raise ValueError("F2 and shared F0 have different row counts")
    left = f2.sort_values(["date", "stock_id"]).reset_index(drop=True)
    right = f0.sort_values(["date", "stock_id"]).reset_index(drop=True)
    if not left.loc[:, KEYS].equals(right.loc[:, KEYS]):
        raise ValueError("F2 and shared F0 have different stock/date keys")
    if not left["day_of_week"].equals(right["day_of_week"]):
        raise ValueError("F2 changed the shared F0 weekday feature")
    for field in NUMERIC_FEATURES:
        f2_values = pd.to_numeric(left[field], errors="raise").to_numpy(dtype=float)
        f0_values = pd.to_numeric(right[field], errors="raise").to_numpy(dtype=float)
        if not np.allclose(f2_values, f0_values, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"F2 changed the shared F0 feature {field}")


def checked_source_score(recorded: dict, seed: int, year: str, actual: float, source_name: str) -> None:
    """Stop if changed raw data or saved predictions no longer reproduce the control."""
    try:
        expected = float(recorded[str(seed)][year])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} summary lacks fold {year}, seed {seed}") from exc
    if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"{source_name} score no longer matches fold {year}, seed {seed}")


def _predictions(keys: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    output = keys.loc[:, KEYS].copy()
    output["predicted_y"] = labels
    return output


def _metric(valid_rows: pd.DataFrame, keys: pd.DataFrame, labels: np.ndarray) -> dict:
    return evaluate_predictions(valid_rows, _predictions(keys, labels))


def run(
    data_dir: Path,
    f0_path: Path,
    f2_path: Path,
    e1_summary_path: Path,
    e1_predictions_path: Path,
    e3_summary_path: Path,
    output_dir: Path,
) -> Path:
    """Run 15 F2 fits and compare all four arms on the same target keys."""
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E4 output directory: {output_dir}")
    e1_summary = load_control_summary(e1_summary_path)
    fixed_rule, e3_summary = load_fixed_rule(e3_summary_path, E1_CONFIG_ID)
    saved_e1 = load_saved_e1_predictions(e1_predictions_path, e1_summary)

    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    f2 = load_f2_dataset(f2_path)
    if f2.columns.tolist() != list(F2_DATASET_COLUMNS):
        raise ValueError("F2 dataset has a different column order")
    if len(f2) != len(market) or set(map(tuple, f2[KEYS].to_numpy())) != set(map(tuple, market[KEYS].to_numpy())):
        raise ValueError("Historical F2 must contain exactly the original market stock/date keys")
    check_f2_matches_f0(f2, f0)

    scores = {arm: {seed: {} for seed in E1_SEEDS} for arm in ARMS}
    metrics = []
    changes = []
    f2_predictions = []
    model_fit_rows = []
    e3_recorded = next(row for row in e3_summary["comparisons"] if row["config_id"] == E3_RULE_ID)
    feature_order = [*F0_FEATURES, *F2_EXTRA_FEATURES]

    for fold in FOLDS:
        train_rows, valid_rows = split_market(market, target_ids, fold)
        if set(train_rows["stock_id"]) != set(all_ids):
            raise ValueError(f"F2 training fold {fold.name} must contain all 100 stocks")
        train = join_f2(train_rows, f2, include_label=True)
        valid = join_f2(valid_rows, f2, include_label=False)
        keys = valid_rows.loc[:, KEYS]
        taiex = valid["taiex_return"].to_numpy(dtype=float)
        preprocessor = F2Preprocessor()
        x_train = preprocessor.fit_transform(train.loc[:, feature_order])
        x_valid = preprocessor.transform(valid.loc[:, feature_order])
        if not x_train.columns.equals(x_valid.columns):
            raise ValueError(f"Training and validation encoded columns differ in {fold.name}")
        if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(x_valid.to_numpy(dtype=float)).all():
            raise ValueError(f"F2 preprocessing left missing or nonfinite values in {fold.name}")
        dtrain = xgb.DMatrix(x_train, label=train["y"].to_numpy())
        dvalid = xgb.DMatrix(x_valid)

        for seed in E1_SEEDS:
            control_p_up = align_saved_probabilities(valid_rows, saved_e1, fold.name, seed)
            fit_start = time.perf_counter()
            booster = xgb.train(
                {**e1_summary["params"], "seed": seed}, dtrain,
                num_boost_round=e1_summary["num_boost_round"],
            )
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            f2_p_up = booster.predict(dvalid)
            predict_seconds = time.perf_counter() - predict_start
            if len(f2_p_up) != len(valid_rows) or not np.isfinite(f2_p_up).all() or ((f2_p_up < 0) | (f2_p_up > 1)).any():
                raise ValueError(f"Invalid E4 probabilities for fold {fold.name}, seed {seed}")

            labels = {
                "e1_f0_0500": (control_p_up > 0.5).astype(int),
                "e1_f0_taiex_rule": regime_predictions(control_p_up, taiex, fixed_rule),
                "e4_f2_0500": (f2_p_up > 0.5).astype(int),
                "e4_f2_taiex_rule": regime_predictions(f2_p_up, taiex, fixed_rule),
            }
            for arm in ARMS:
                result = _metric(valid_rows, keys, labels[arm])
                scores[arm][seed][fold.name] = result["official_score"]
                metrics.append({
                    "member": e1_summary["member"], "model": "XGBoost",
                    "experiment": "E1" if arm == "e1_f0_0500" else "E2",
                    "config_id": ARM_CONFIG_IDS[arm],
                    "arm": arm, "fold": fold.name, "seed": seed, **result,
                    "train_seconds": fit_seconds if arm.startswith("e4") else None,
                    "predict_seconds": predict_seconds if arm.startswith("e4") else None,
                })
            checked_source_score(e1_summary["fold_official_scores_by_seed"], seed, fold.name,
                                 scores["e1_f0_0500"][seed][fold.name], "E1")
            checked_source_score(e3_recorded["fold_official_scores_by_seed"], seed, fold.name,
                                 scores["e1_f0_taiex_rule"][seed][fold.name], "E3")
            for left, right in combinations(ARMS, 2):
                n_changed = int(np.count_nonzero(labels[left] != labels[right]))
                changes.append({
                    "fold": fold.name, "seed": seed, "left_arm": left, "right_arm": right,
                    "n_changed": n_changed, "fraction_changed": n_changed / len(valid_rows),
                })
            one_run = keys.copy()
            one_run.insert(0, "seed", seed)
            one_run.insert(0, "fold", fold.name)
            one_run.insert(0, "config_id", E4_CONFIG_ID)
            one_run.insert(0, "experiment", "E2")
            one_run.insert(0, "model", "XGBoost")
            one_run.insert(0, "member", e1_summary["member"])
            one_run["p_up"] = f2_p_up
            one_run["predicted_y"] = labels["e4_f2_0500"]
            one_run["predicted_y_taiex_rule"] = labels["e4_f2_taiex_rule"]
            f2_predictions.append(one_run)
            model_fit_rows.append({"fold": fold.name, "seed": seed, "train_seconds": fit_seconds,
                                   "predict_seconds": predict_seconds, "n_train": len(train), "n_eval": len(valid_rows)})
            print(f"{fold.name}, seed {seed}: E1 {scores['e1_f0_0500'][seed][fold.name]:.8f}; "
                  f"E1+rule {scores['e1_f0_taiex_rule'][seed][fold.name]:.8f}; "
                  f"E4 {scores['e4_f2_0500'][seed][fold.name]:.8f}; "
                  f"E4+rule {scores['e4_f2_taiex_rule'][seed][fold.name]:.8f}")

    comparisons = [{
        "arm": arm,
        "fold_official_scores_by_seed": scores[arm],
        "mean_official_score_by_fold": {
            fold.name: float(np.mean([scores[arm][seed][fold.name] for seed in E1_SEEDS]))
            for fold in FOLDS
        },
        **e1_seed_summary(scores[arm]),
    } for arm in ARMS]
    difference_summary = []
    total_evaluated = sum(row["n_eval"] for row in model_fit_rows)
    for left, right in combinations(ARMS, 2):
        n_changed = sum(row["n_changed"] for row in changes
                        if row["left_arm"] == left and row["right_arm"] == right)
        difference_summary.append({
            "left_arm": left, "right_arm": right,
            "n_changed_over_fold_seed_rows": n_changed,
            "n_compared_over_fold_seed_rows": total_evaluated,
            "fraction_changed": n_changed / total_evaluated,
        })
    # Tuple order prefers the already established E1 control on an exact tie.
    selected = max(comparisons, key=lambda row: row["mean_cv_official_score"])
    summary = {
        "experiment_label": "E4 F2 intraday/overnight history comparison",
        "experiment": "E2",
        "feature_set": "F2",
        "f0_feature_order": list(F0_FEATURES),
        "f2_extra_feature_order": list(F2_EXTRA_FEATURES),
        "feature_order": feature_order,
        "preprocessing": "F0 BaselinePreprocessor plus fold-fitted median/standard scaling for five E4 numeric fields",
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y", "split_version": "1.2",
        "seeds": list(E1_SEEDS),
        "source_e1_config_id": E1_CONFIG_ID,
        "source_e1_summary": str(e1_summary_path), "source_e1_summary_sha256": _sha256(e1_summary_path),
        "source_e1_predictions": str(e1_predictions_path), "source_e1_predictions_sha256": _sha256(e1_predictions_path),
        "source_e3_summary": str(e3_summary_path), "source_e3_summary_sha256": _sha256(e3_summary_path),
        "source_f2": str(f2_path), "source_f2_sha256": _sha256(f2_path),
        "source_f0": str(f0_path), "source_f0_sha256": _sha256(f0_path),
        "params": e1_summary["params"], "num_boost_round": e1_summary["num_boost_round"],
        "n_f2_model_fits": len(model_fit_rows),
        "fixed_e3_rule": fixed_rule,
        "arm_config_ids": ARM_CONFIG_IDS,
        "comparisons": comparisons,
        "selected_arm_by_cv": selected["arm"],
        "selected_mean_cv_official_score": selected["mean_cv_official_score"],
        "e4_arm_wins": selected["arm"].startswith("e4_"),
        "selection_method": "Mean of 2023, 2024, and 2025 target-stock official scores within each seed; mean of five seeds; prespecified arm order breaks exact ties",
        "prediction_difference_file": "prediction_differences.csv",
        "prediction_differences_over_fold_seed_rows": difference_summary,
        "runtime_note": "The same E4 model fit and probability inference serve the 0.5 and TAIEX-rule arms; E1 probabilities were reused. E4 runtime is repeated across those two metric rows.",
        "status": "validation only; no E4 deployment artifact or submission created",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "xgboost": xgb.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metrics).loc[:, [*METRIC_FIELDS, "arm"]].to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    pd.DataFrame(changes).to_csv(output_dir / "prediction_differences.csv", index=False)
    pd.DataFrame(model_fit_rows).to_csv(output_dir / "model_fit_runtime.csv", index=False)
    pd.concat(f2_predictions, ignore_index=True).loc[
        :, [*PREDICTION_FIELDS, "p_up", "predicted_y_taiex_rule"]
    ].to_csv(output_dir / "validation_predictions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    for row in comparisons:
        print(f"{row['arm']}: five-seed mean CV official score {row['mean_cv_official_score']:.8f}; "
              f"seed standard deviation {row['std_cv_official_score']:.8f}")
    print(f"Selected CV arm: {selected['arm']}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--f2-path", type=Path, default=DEFAULT_F2_PATH)
    parser.add_argument("--e1-summary", type=Path, default=DEFAULT_E1_SUMMARY)
    parser.add_argument("--e1-predictions", type=Path, default=DEFAULT_E1_PREDICTIONS)
    parser.add_argument("--e3-summary", type=Path, default=DEFAULT_E3_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.f2_path, args.e1_summary, args.e1_predictions,
        args.e3_summary, args.output_dir)
