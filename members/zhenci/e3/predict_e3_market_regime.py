"""Export Phase 1 predictions using the saved E1 model and selected E3 rule.

Run from the repository root:
    uv run python -m members.zhenci.e3.predict_e3_market_regime

This script only loads a frozen model/preprocessor and updated F0 rows. It
does not refit, select a rule from submission data, or upload the CSV.
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
import xgboost as xgb

from members.zhenci.e1.predict_e1 import require_template_features
from members.zhenci.e1.save_e1 import DEFAULT_OUTPUT_DIR as DEFAULT_ARTIFACT, _sha256
from members.zhenci.e3.evaluate_e3_market_regime import CANDIDATES
from members.zhenci.e3.evaluate_e3_market_regime import regime_predictions
from shared.data_loader import load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.submission import load_submission_template, write_submission

DEFAULT_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2/summary.json"
DEFAULT_F0 = PROJECT_ROOT / "data/processed/f0_with_updates.csv"
DEFAULT_TEMPLATE = PROJECT_ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "submissions/zhenci_e3_taiex_phase1.csv"


def selected_rule(summary: dict, artifact_config_id: str) -> dict:
    """Accept only a recorded, compatible CV-selected rule from the fixed grid."""
    if summary.get("source_model_configuration") != artifact_config_id:
        raise ValueError("Threshold study and frozen model use different E1 configurations")
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(CANDIDATES):
        raise ValueError("Threshold study has an unexpected candidate set")
    for recorded, expected in zip(comparisons, CANDIDATES):
        if any(recorded.get(field) != expected[field] for field in expected):
            raise ValueError("Threshold study rule definitions differ from this exporter")
    winner = summary.get("selected_same_period_config_id")
    if winner not in {candidate["config_id"] for candidate in CANDIDATES}:
        raise ValueError("Threshold study has no supported selected rule")
    best = max(comparisons, key=lambda row: row["mean_cv_official_score"])
    if winner != best["config_id"]:
        raise ValueError("Selected rule does not match the recorded CV winner")
    return next(candidate for candidate in CANDIDATES if candidate["config_id"] == winner)


def export_predictions(
    artifact_dir: Path,
    summary_path: Path,
    f0_path: Path,
    template_path: Path,
    output_path: Path,
    data_dir: Path,
) -> Path:
    """Apply the selected postprocessing rule to frozen E1 probabilities."""
    if output_path.exists():
        raise FileExistsError(output_path)
    with (artifact_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("artifact_version") != 1 or manifest.get("feature_order") != list(F0_FEATURES):
        raise ValueError("Unsupported artifact or changed F0 feature order")
    if manifest.get("threshold_rule") != "predict 1 when p_up > 0.5, else 0" or manifest.get("threshold") != 0.5:
        raise ValueError("Expected the frozen full-sample E1 probability model")
    for name, hash_field in (
        ("model.json", "model_sha256"),
        ("preprocessor.json", "preprocessor_sha256"),
        ("feature_definitions.py", "feature_definitions_sha256"),
        ("preprocessing_definitions.py", "preprocessing_definitions_sha256"),
    ):
        if _sha256(artifact_dir / name) != manifest.get(hash_field):
            raise ValueError(f"Artifact file changed: {name}")
    if _sha256(PROJECT_ROOT / "shared/features.py") != manifest["feature_definitions_sha256"]:
        raise ValueError("Current F0 construction differs from the frozen artifact")
    if _sha256(PROJECT_ROOT / "shared/preprocessing.py") != manifest["preprocessing_definitions_sha256"]:
        raise ValueError("Current preprocessing differs from the frozen artifact")

    with summary_path.open("r", encoding="utf-8") as file:
        rule = selected_rule(json.load(file), manifest["config_id"])
    target_ids, _ = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    rows = require_template_features(load_f0_dataset(f0_path), ids, dates)
    with (artifact_dir / "preprocessor.json").open("r", encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    features = preprocessor.transform(rows.loc[:, list(F0_FEATURES)])
    if features.columns.tolist() != manifest.get("transformed_feature_order"):
        raise ValueError("Frozen transformed feature columns do not match artifact")
    booster = xgb.Booster(model_file=str(artifact_dir / "model.json"))
    p_up = booster.predict(xgb.DMatrix(features))
    if len(p_up) != len(rows) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
        raise ValueError("Model returned invalid prediction probabilities")
    predictions = rows.loc[:, ["stock_id", "date"]].copy()
    predictions["predicted_y"] = regime_predictions(p_up, rows["taiex_return"].to_numpy(), rule)
    return write_submission(predictions, template_path, output_path, target_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    print(export_predictions(args.artifact_dir, args.summary, args.f0_path,
                             args.template, args.output, args.data_dir))
