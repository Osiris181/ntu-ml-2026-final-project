"""Export template-aligned E1 predictions from a frozen local artifact.

Requires a complete F0 row for every template stock/date. Does not fetch data,
refit preprocessing, retrain the model, or upload a submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.save_e1 import _sha256
from shared.data_loader import load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.submission import load_submission_template, write_submission


def require_template_features(f0: pd.DataFrame, ids: list[str], dates: list[str]) -> pd.DataFrame:
    """Preserve template order and reject absent stock/date feature rows."""
    keys = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=["stock_id", "date"])
    required = keys.to_frame(index=False)
    matched = required.merge(f0, on=["stock_id", "date"], how="left", validate="one_to_one", indicator=True)
    missing = matched.loc[matched["_merge"] != "both", ["stock_id", "date"]]
    if not missing.empty:
        first = missing.iloc[0]
        raise ValueError(
            f"F0 lacks {len(missing)} template stock/date rows; first missing: "
            f"{first['stock_id']} on {first['date'].date()}"
        )
    return matched.drop(columns="_merge")


def export_predictions(
    artifact_dir: Path,
    f0_path: Path,
    template_path: Path,
    output_path: Path,
    data_dir: Path,
) -> Path:
    """Use only frozen training state and current eligible F0 inputs."""
    manifest_path = artifact_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("artifact_version") != 1 or manifest.get("feature_order") != list(F0_FEATURES):
        raise ValueError("Unsupported artifact or changed F0 feature order")
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
    if manifest.get("threshold_rule") != "predict 1 when p_up > 0.5, else 0" or manifest.get("threshold") != 0.5:
        raise ValueError("Unsupported frozen E1 threshold rule")

    target_ids, _ = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    f0 = load_f0_dataset(f0_path)
    rows = require_template_features(f0, ids, dates)
    with (artifact_dir / "preprocessor.json").open("r", encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    features = preprocessor.transform(rows.loc[:, list(F0_FEATURES)])
    if features.columns.tolist() != manifest.get("transformed_feature_order"):
        raise ValueError("Frozen transformed feature columns do not match artifact")
    booster = xgb.Booster(model_file=str(artifact_dir / "model.json"))
    p_up = booster.predict(xgb.DMatrix(features))
    if len(p_up) != len(rows) or not np.isfinite(p_up).all() or not np.logical_and(p_up >= 0, p_up <= 1).all():
        raise ValueError("Model returned invalid prediction probabilities")
    predictions = rows.loc[:, ["stock_id", "date"]].copy()
    predictions["predicted_y"] = (p_up > manifest["threshold"]).astype(int)
    return write_submission(predictions, template_path, output_path, target_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    print(export_predictions(args.artifact_dir, args.f0_path, args.template, args.output, args.data_dir))
