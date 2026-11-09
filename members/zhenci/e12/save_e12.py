"""Refit the fixed E12 LSTM on 2020-2025 history and freeze its artifact.

Run from final_project: uv run --extra e12 python -m members.zhenci.e12.save_e12
This does not use 2026 labels, create a submission, or upload anything.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from members.zhenci.e1.save_e1 import holdout_training_rows
from members.zhenci.e1.train_e1 import join_f0
from members.zhenci.e12.sequences import WINDOW, complete_training_keys, sequence_batch
from members.zhenci.e12.train_e12 import (
    BATCH_SIZE, EPOCHS, HEAD_DROPOUT, HIDDEN, LEARNING_RATE,
    StockLSTM, _bce, _probabilities, set_seed, sha256,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import HOLDOUT, SEED

DEFAULT_CV = ROOT / "members/zhenci/e12/outputs/summary.json"
DEFAULT_ARTIFACT = ROOT / "results/zhenci_e12_lstm_pre_phase1_artifact"
FROZEN_SOURCES = {
    "feature_definitions.py": ROOT / "shared/features.py",
    "preprocessing_definitions.py": ROOT / "shared/preprocessing.py",
    "lstm_definition.py": Path(__file__).resolve().parent / "train_e12.py",
    "sequence_definitions.py": Path(__file__).resolve().parent / "sequences.py",
}


def verify_cv_recipe(path: Path, f0_path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        summary = json.load(file)
    architecture = summary.get("architecture", {})
    training = summary.get("training", {})
    if (summary.get("experiment") != "E12 fixed one-layer LSTM sequence model"
            or architecture.get("window") != WINDOW
            or architecture.get("numeric_features") != list(NUMERIC_FEATURES)
            or architecture.get("hidden_size") != HIDDEN
            or architecture.get("layers") != 1
            or architecture.get("head_dropout") != HEAD_DROPOUT
            or training.get("epochs_fixed") != EPOCHS
            or training.get("batch_size") != BATCH_SIZE
            or training.get("optimizer") != "Adam"
            or training.get("loss") != "BCEWithLogitsLoss"
            or training.get("initial_learning_rate") != LEARNING_RATE
            or training.get("seed") != SEED
            or training.get("device") != "cpu"
            or training.get("gradient_clip_norm") != 1.0
            or summary.get("input_sha256", {}).get("f0") != sha256(f0_path)):
        raise ValueError("E12 CV summary, fixed recipe, or original F0 input changed")
    return summary


def save_e12(cv_path: Path, data_dir: Path, f0_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new E12 artifact directory: {output_dir}")
    verify_cv_recipe(cv_path, f0_path)
    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    train_rows = holdout_training_rows(market, target_ids, all_ids)
    if (train_rows["date"].max() > pd.Timestamp(HOLDOUT.train_end)
            or train_rows["stock_id"].nunique() != 100
            or not train_rows["y"].isin([0, 1]).all()):
        raise ValueError("Invalid 2020-2025 training window or labels")
    f0 = load_f0_dataset(f0_path)
    train = join_f0(train_rows, f0, include_label=True)
    preprocessor = BaselinePreprocessor().fit(train[list(F0_FEATURES)])
    keys = complete_training_keys(train_rows)
    batch, context_names = sequence_batch(f0.loc[f0["date"].le(HOLDOUT.train_end)],
                                          keys, preprocessor)
    labeled = batch["keys"].merge(train_rows[KEYS + ["y"]], on=KEYS, how="left",
                                  validate="one_to_one", indicator=True)
    if (not labeled["_merge"].eq("both").all()
            or not labeled[KEYS].equals(batch["keys"])
            or not labeled["y"].isin([0, 1]).all()):
        raise ValueError("Training labels do not align to E12 sequences")
    labels = labeled["y"].to_numpy(dtype=np.float32)
    print(f"Fitting E12 on {len(batch['keys'])} sequences from "
          f"{train_rows['stock_id'].nunique()} stocks through {HOLDOUT.train_end}", flush=True)
    set_seed()
    model = StockLSTM(len(context_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(batch["sequences"]),
                            torch.from_numpy(batch["context"]),
                            torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    started = time.perf_counter()
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for sequence, context, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(sequence, context), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_bce = _bce(_probabilities(model, batch), labels)
        losses.append(train_bce)
        print(f"E12 final refit epoch {epoch}/{EPOCHS}: training BCE {train_bce:.5f}",
              flush=True)
    train_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    with (output_dir / "preprocessor.json").open("w", encoding="utf-8") as file:
        json.dump(preprocessor.to_state(), file, indent=2, allow_nan=False)
        file.write("\n")
    for name, path in FROZEN_SOURCES.items():
        shutil.copyfile(path, output_dir / name)
    restored = StockLSTM(len(context_names))
    restored.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=True)
    probe = {"keys": batch["keys"].iloc[:5], "sequences": batch["sequences"][:5],
             "context": batch["context"][:5]}
    if not np.array_equal(_probabilities(model, probe), _probabilities(restored, probe)):
        raise ValueError("Saved E12 model changed predictions after reload")
    frozen_files = ["model.pt", "preprocessor.json", *FROZEN_SOURCES]
    manifest = {
        "artifact_version": 1,
        "experiment": "E12 fixed one-layer LSTM sequence model",
        "status": "exploratory_e12_pre_phase1_requested",
        "train_start": "2020-01-01", "train_end": HOLDOUT.train_end,
        "training_rows": len(train_rows), "training_sequences": len(batch["keys"]),
        "training_stocks": train_rows["stock_id"].nunique(),
        "seed": SEED, "window": WINDOW,
        "feature_order": list(F0_FEATURES),
        "numeric_feature_order": list(NUMERIC_FEATURES),
        "transformed_feature_order": list(preprocessor.feature_names_),
        "context_names": context_names,
        "model": {"layers": 1, "hidden_size": HIDDEN, "head_dropout": HEAD_DROPOUT},
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "loss": "BCEWithLogitsLoss", "optimizer": "Adam",
                     "learning_rate": LEARNING_RATE, "gradient_clip_norm": 1.0,
                     "device": "cpu", "seconds": train_seconds,
                     "training_bce_by_epoch": losses},
        "decision": "predict 1 when p_up > 0.5, else 0",
        "threshold": 0.5,
        "f0_csv_sha256": sha256(f0_path),
        "cv_summary_sha256": sha256(cv_path),
        "stock_metadata_sha256": sha256(data_dir / "info.json"),
        "file_sha256": {name: sha256(output_dir / name) for name in frozen_files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Exploratory E12 artifact saved to {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=ROOT / "data/processed/f0.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    save_e12(args.cv_summary, args.data_dir, args.f0_path, args.output_dir)
