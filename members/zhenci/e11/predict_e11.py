"""Export an exploratory E11 Phase 1 CSV from a frozen E11 artifact.

Run from final_project after save_e11:
    uv run python -m members.zhenci.e11.predict_e11

This reads updated F0 feature values and the official template. It does not
train, refit preprocessing, use submission-date labels, or upload the CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.predict_e1 import require_template_features
from members.zhenci.e1.save_e1 import _sha256
from members.zhenci.e11.save_e11 import DEFAULT_ARTIFACT
from members.zhenci.e11.train_e11 import CAP_QUANTILE, expected_return
from shared.data_loader import load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.submission import load_submission_template, write_submission

DEFAULT_F0 = ROOT / "data/processed/f0_with_updates.csv"
DEFAULT_TEMPLATE = ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "submissions/zhenci_e11_expected_return_phase1.csv"
MODEL_FILES = ("model_up.json", "model_flat.json", "model_gain.json", "model_loss.json",
               "preprocessor.json", "feature_definitions.py", "preprocessing_definitions.py")


def load_artifact(artifact_dir: Path) -> tuple[dict, BaselinePreprocessor]:
    with (artifact_dir/"manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("artifact_version") != 1
            or manifest.get("status") != "exploratory_e11_pre_phase1_requested"
            or manifest.get("train_end") != "2025-12-31"
            or manifest.get("training_stocks") != 100
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("cap_quantile_training_only") != CAP_QUANTILE
            or manifest.get("formula") != "p_up*mu_gain - (1-p_up)*(1-q_flat_given_nonup)*mu_loss > 0"
            or set(manifest.get("file_sha256", {})) != set(MODEL_FILES)):
        raise ValueError("Unsupported E11 artifact or changed frozen recipe")
    for name in MODEL_FILES:
        if _sha256(artifact_dir/name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen E11 artifact changed: {name}")
    if (_sha256(ROOT/"shared/features.py") != manifest["file_sha256"]["feature_definitions.py"]
            or _sha256(ROOT/"shared/preprocessing.py")
            != manifest["file_sha256"]["preprocessing_definitions.py"]):
        raise ValueError("Shared F0 or preprocessing definitions changed after E11 training")
    if (not np.isfinite(manifest.get("gain_cap", np.nan))
            or not np.isfinite(manifest.get("loss_cap", np.nan))
            or manifest["gain_cap"] <= 0 or manifest["loss_cap"] <= 0):
        raise ValueError("Invalid frozen conditional-return caps")
    with (artifact_dir/"preprocessor.json").open(encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    if list(preprocessor.feature_names_) != manifest.get("transformed_feature_order"):
        raise ValueError("Frozen transformed feature order differs")
    return manifest, preprocessor


def export_predictions(artifact_dir: Path, f0_path: Path, template_path: Path,
                       output_path: Path, data_dir: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Submission already exists: {output_path}")
    manifest, preprocessor = load_artifact(artifact_dir)
    target_ids, _ = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    if min(pd.to_datetime(dates)) <= pd.Timestamp(manifest["train_end"]):
        raise ValueError("Template dates must be after E11's frozen training window")
    rows = require_template_features(load_f0_dataset(f0_path), ids, dates)
    if rows[["stock_id", "date"]].duplicated().any():
        raise ValueError("Duplicate template features")
    features = preprocessor.transform(rows[list(F0_FEATURES)])
    if features.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("E11 transformed feature order changed")
    matrix = xgb.DMatrix(features)
    raw = {name:xgb.Booster(model_file=str(artifact_dir/f"model_{name}.json")).predict(matrix)
           for name in ("up", "flat", "gain", "loss")}
    if any(values.shape != (len(rows),) or not np.isfinite(values).all()
           for values in raw.values()):
        raise ValueError("E11 model returned invalid component predictions")
    if ((raw["up"] < 0).any() or (raw["up"] > 1).any()
            or (raw["flat"] < 0).any() or (raw["flat"] > 1).any()):
        raise ValueError("E11 probability component is outside [0,1]")
    p_up = raw["up"].astype(float)
    q_flat = raw["flat"].astype(float)
    gain = np.clip(raw["gain"].astype(float), 0, manifest["gain_cap"])
    loss = np.clip(raw["loss"].astype(float), 0, manifest["loss_cap"])
    expected = expected_return(p_up, q_flat, gain, loss)
    predictions = rows[["stock_id", "date"]].copy()
    predictions["predicted_y"] = (expected > 0).astype(int)
    result = write_submission(predictions, template_path, output_path, target_ids)
    print(f"E11 predicted up for {int(predictions['predicted_y'].sum())} "
          f"of {len(predictions)} template rows ({predictions['predicted_y'].mean():.1%})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir",type=Path,default=DEFAULT_ARTIFACT)
    parser.add_argument("--f0-path",type=Path,default=DEFAULT_F0)
    parser.add_argument("--template",type=Path,default=DEFAULT_TEMPLATE)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    args = parser.parse_args()
    print(export_predictions(args.artifact_dir,args.f0_path,args.template,args.output,args.data_dir))
