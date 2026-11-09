"""Export Phase 1 E12 predictions using a frozen final-refit artifact.

Run after save_e12: uv run --extra e12 python -m members.zhenci.e12.predict_e12
Reads updated F0 values but no 2026 labels; does not train or upload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from members.zhenci.e1.predict_e1 import require_template_features
from members.zhenci.e12.save_e12 import DEFAULT_ARTIFACT, FROZEN_SOURCES
from members.zhenci.e12.sequences import WINDOW, sequence_batch
from members.zhenci.e12.train_e12 import (
    BATCH_SIZE, EPOCHS, HEAD_DROPOUT, HIDDEN, LEARNING_RATE,
    StockLSTM, _probabilities, set_seed, sha256,
)
from shared.data_loader import KEYS, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import SEED
from shared.submission import load_submission_template, write_submission

DEFAULT_F0 = ROOT / "data/processed/f0_with_updates.csv"
DEFAULT_TEMPLATE = ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "submissions/zhenci_e12_lstm_phase1.csv"


def load_artifact(artifact_dir: Path) -> tuple[dict, BaselinePreprocessor, StockLSTM]:
    with (artifact_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("artifact_version") != 1
            or manifest.get("experiment") != "E12 fixed one-layer LSTM sequence model"
            or manifest.get("train_end") != "2025-12-31"
            or manifest.get("training_stocks") != 100
            or manifest.get("seed") != SEED
            or manifest.get("window") != WINDOW
            or manifest.get("feature_order") != list(F0_FEATURES)
            or manifest.get("numeric_feature_order") != list(NUMERIC_FEATURES)
            or manifest.get("model") != {"layers": 1, "hidden_size": HIDDEN,
                                         "head_dropout": HEAD_DROPOUT}
            or any(manifest.get("training", {}).get(field) != value for field, value in {
                "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "loss": "BCEWithLogitsLoss", "optimizer": "Adam",
                "learning_rate": LEARNING_RATE, "gradient_clip_norm": 1.0,
                "device": "cpu"}.items())
            or manifest.get("decision") != "predict 1 when p_up > 0.5, else 0"
            or manifest.get("threshold") != 0.5
            or set(manifest.get("file_sha256", {}))
            != {"model.pt", "preprocessor.json", *FROZEN_SOURCES}):
        raise ValueError("Unsupported or changed E12 frozen artifact")
    for name, expected in manifest["file_sha256"].items():
        if sha256(artifact_dir / name) != expected:
            raise ValueError(f"Frozen E12 artifact file changed: {name}")
    for name, source in FROZEN_SOURCES.items():
        if sha256(source) != manifest["file_sha256"][name]:
            raise ValueError(f"Current E12 or shared definitions differ: {source}")
    with (artifact_dir / "preprocessor.json").open(encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    if (list(preprocessor.feature_names_) != manifest.get("transformed_feature_order")
            or list(preprocessor.feature_names_[len(NUMERIC_FEATURES):])
            != manifest.get("context_names")):
        raise ValueError("Frozen E12 feature or context order changed")
    set_seed()
    model = StockLSTM(len(manifest["context_names"]))
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu",
                                     weights_only=True), strict=True)
    model.eval()
    return manifest, preprocessor, model


def export_predictions(artifact_dir: Path, f0_path: Path, template_path: Path,
                       output_path: Path, data_dir: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Submission already exists: {output_path}")
    manifest, preprocessor, model = load_artifact(artifact_dir)
    target_ids, _ = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    if min(pd.to_datetime(dates)) <= pd.Timestamp(manifest["train_end"]):
        raise ValueError("Template dates must follow the frozen E12 training window")
    f0 = load_f0_dataset(f0_path)
    matched = require_template_features(f0, ids, dates)
    keys = matched[KEYS]
    batch, context_names = sequence_batch(f0, keys, preprocessor)
    if (not batch["keys"].equals(keys.reset_index(drop=True))
            or context_names != manifest["context_names"]):
        raise ValueError("E12 submission sequence keys or context order changed")
    probabilities = _probabilities(model, batch)
    predictions = keys.copy()
    predictions["predicted_y"] = (probabilities > manifest["threshold"]).astype(int)
    output = write_submission(predictions, template_path, output_path, target_ids)
    print(f"E12 predicted up for {int(predictions['predicted_y'].sum())} "
          f"of {len(predictions)} template rows "
          f"({predictions['predicted_y'].mean():.1%})", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(export_predictions(args.artifact_dir, args.f0_path, args.template,
                             args.output, args.data_dir))
