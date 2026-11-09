"""One-time Jan-May 2026 holdout check of the CV-selected E7 decision rule.

Run from the repository root:
    uv run python -m members.zhenci.e7.evaluate_holdout

This uses the already refitted 2020-2025 E1 model. It never trains XGBoost,
refits preprocessing, or uses holdout outcomes to choose a rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.save_e1 import _sha256
from members.zhenci.e4.train_e4 import E1_CONFIG_ID, load_fixed_rule
from members.zhenci.e7.evaluate_e7 import (
    PSEUDO_COUNT_PER_CLASS, apply_payoff_cutoffs, fit_payoff_cutoffs,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.evaluation import evaluate_predictions
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT, split_market

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_E7_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e7_payoff_v1/summary.json"
DEFAULT_ARTIFACT = PROJECT_ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_xgboost_e7_holdout_v1"
SELECTED_ARM = "stock_payoff_shrunk_100"
CONTROL_ARMS = ("e1_0500_control", "e3_taiex_control")


def load_locked_inputs(summary_path: Path, artifact_dir: Path, f0_path: Path) -> tuple[dict, dict]:
    """Reject a changed CV decision, F0 input, or frozen probability model."""
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    with (artifact_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if (summary.get("selected_arm_by_cv") != SELECTED_ARM
            or summary.get("incumbent_arm") != "e3_taiex_control"
            or summary.get("payoff_rule_beats_e3") is not True
            or summary.get("stock_shrinkage")
            != f"{PSEUDO_COUNT_PER_CLASS} pooled pseudo-observations for each stock's up and non-up payoff mean"):
        raise ValueError("E7 CV selection is missing or differs from the frozen stock-payoff rule")
    if (summary.get("feature_set") != "F0"
            or summary.get("source_model_config_id") != "xgboost_initial_v1"
            or summary.get("source_f0_sha256") != _sha256(f0_path)
            or summary.get("source_e1_summary_sha256") != manifest.get("cv_summary_sha256")):
        raise ValueError("E7 summary does not match the frozen E1 model or original F0")
    for field, filename in (
        ("source_e1_predictions_sha256", "source_e1_predictions"),
        ("source_e3_summary_sha256", "source_e3_summary"),
    ):
        if _sha256(Path(summary[filename])) != summary[field]:
            raise ValueError(f"E7 source changed: {filename}")
    e3_rule, _ = load_fixed_rule(Path(summary["source_e3_summary"]), E1_CONFIG_ID)
    if summary.get("fixed_e3_rule") != e3_rule:
        raise ValueError("E7 incumbent rule differs from the recorded E3 rule")
    comparisons = {row["arm"]: row for row in summary.get("comparisons", [])}
    if (set(comparisons) != {SELECTED_ARM, *CONTROL_ARMS, "pooled_payoff", "regime_payoff"}
            or max(comparisons.values(), key=lambda row: row["mean_cv_official_score"])["arm"] != SELECTED_ARM):
        raise ValueError("E7 comparisons no longer support the selected rule")

    if (manifest.get("artifact_version") != 1
            or manifest.get("status") != "provisional_e1_pre_holdout"
            or manifest.get("training_split") != HOLDOUT.name
            or manifest.get("train_start") != "2020-01-01"
            or manifest.get("train_end") != HOLDOUT.train_end
            or manifest.get("training_stocks") != 100
            or manifest.get("seed") != 42
            or manifest.get("config_id") != summary["source_model_config_id"]
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("feature_set") != "F0"
            or manifest.get("f0_csv_sha256") != _sha256(f0_path)
            or manifest.get("threshold_rule") != "predict 1 when p_up > 0.5, else 0"
            or manifest.get("threshold") != 0.5):
        raise ValueError("Expected the frozen 2020-2025 E1 artifact")
    for name, hash_field in (
        ("model.json", "model_sha256"),
        ("preprocessor.json", "preprocessor_sha256"),
        ("feature_definitions.py", "feature_definitions_sha256"),
        ("preprocessing_definitions.py", "preprocessing_definitions_sha256"),
    ):
        if _sha256(artifact_dir / name) != manifest.get(hash_field):
            raise ValueError(f"Frozen artifact changed: {name}")
    if (_sha256(PROJECT_ROOT / "shared/features.py") != manifest["feature_definitions_sha256"]
            or _sha256(PROJECT_ROOT / "shared/preprocessing.py") != manifest["preprocessing_definitions_sha256"]):
        raise ValueError("Shared feature or preprocessing definitions changed")
    return summary, manifest


def run(data_dir: Path, f0_path: Path, summary_path: Path,
        artifact_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Holdout output already exists; do not rerun or overwrite: {output_dir}")
    summary, manifest = load_locked_inputs(summary_path, artifact_dir, f0_path)
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path)
    if len(f0) != len(market) or not f0.loc[:, KEYS].equals(market.loc[:, KEYS]):
        raise ValueError("Expected original historical F0 and market rows")
    train_rows, holdout_rows = split_market(market, target_ids, HOLDOUT)
    if set(train_rows.stock_id) != set(all_ids):
        raise ValueError("Holdout training window lacks some of the 100 stocks")
    if len(train_rows) != manifest["training_rows"]:
        raise ValueError("Training rows differ from the frozen model's training window")
    train_f0 = train_rows.loc[:, KEYS].merge(
        f0.loc[:, [*KEYS, "taiex_return"]], on=KEYS,
        how="left", validate="one_to_one", sort=False, indicator=True,
    )
    eval_f0 = holdout_rows.loc[:, KEYS].merge(
        f0, on=KEYS, how="left", validate="one_to_one", sort=False, indicator=True,
    )
    for joined, original in ((train_f0, train_rows), (eval_f0, holdout_rows)):
        if not joined["_merge"].eq("both").all() or not joined.loc[:, KEYS].equals(original.loc[:, KEYS]):
            raise ValueError("F0 join lost or reordered holdout split keys")
    fitted, cutoff_details = fit_payoff_cutoffs(train_rows, train_f0["taiex_return"], HOLDOUT.name)
    with (artifact_dir / "preprocessor.json").open("r", encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    transformed = preprocessor.transform(eval_f0.loc[:, list(F0_FEATURES)])
    if transformed.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("Frozen transformed feature order changed")
    booster = xgb.Booster(model_file=str(artifact_dir / "model.json"))
    p_up = booster.predict(xgb.DMatrix(transformed))
    if len(p_up) != len(holdout_rows) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
        raise ValueError("Frozen model returned invalid probabilities")

    predictions = holdout_rows.loc[:, KEYS].copy()
    predictions["p_up"] = p_up
    results = {}
    for arm in (*CONTROL_ARMS, SELECTED_ARM):
        labels = apply_payoff_cutoffs(
            p_up, eval_f0["stock_id"], eval_f0["taiex_return"],
            fitted, arm, summary["fixed_e3_rule"],
        )
        predictions[f"predicted_y_{arm}"] = labels
        scored = holdout_rows.loc[:, KEYS].copy()
        scored["predicted_y"] = labels
        results[arm] = {**evaluate_predictions(holdout_rows, scored),
                        "predicted_up_rate": float(labels.mean())}

    selected = predictions[f"predicted_y_{SELECTED_ARM}"].to_numpy()
    incumbent = predictions["predicted_y_e3_taiex_control"].to_numpy()
    report = {
        "experiment": "E7 one-time independent holdout evaluation",
        "selected_arm_frozen_before_holdout": SELECTED_ARM,
        "holdout_split": HOLDOUT.name,
        "holdout_dates": [HOLDOUT.eval_start, HOLDOUT.eval_end],
        "model_training_dates": ["2020-01-01", HOLDOUT.train_end],
        "model_seed": manifest["seed"],
        "cv_selected_mean_official_score": summary["selected_mean_cv_official_score"],
        "source_e7_summary_sha256": _sha256(summary_path),
        "source_artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "source_f0_sha256": _sha256(f0_path),
        "results": results,
        "selected_minus_e3_official_score": results[SELECTED_ARM]["official_score"] - results["e3_taiex_control"]["official_score"],
        "selected_vs_e3_changed_predictions": int((selected != incumbent).sum()),
        "interpretation": "Independent assessment only; do not use this holdout to change the rule, model, or training settings",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    cutoff_details.to_csv(output_dir / "payoff_cutoffs.csv", index=False)
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    for arm, result in results.items():
        print(f"{arm}: holdout official score {result['official_score']:.8f}")
    print(f"E7 minus E3: {report['selected_minus_e3_official_score']:+.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_E7_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.summary, args.artifact_dir, args.output_dir)
