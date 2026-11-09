"""Refit a selected XGBoost E1 configuration and save a provisional artifact.

This script does not use the Jan-May 2026 holdout labels for training or
selection, predict submission dates, or upload a file. Defaults use the
selected full-sample E1 summary, a new artifact directory, the shared
holdout training window, and reference seed 42.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CV_SUMMARY = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_e1_full_pre_holdout_artifact"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from members.zhenci.e1.train_e1 import join_f0
from shared.data_loader import load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, HOLDOUT, SEED, split_market


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cv_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    required = {
        "model": "XGBoost",
        "experiment": "E1",
        "feature_set": "F0",
        "training_universe": "Top100@Y",
        "evaluation_universe": "Top50@Y",
        "split_version": "1.2",
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
    }
    if any(summary.get(field) != value for field, value in required.items()):
        raise ValueError("CV summary is incompatible with the shared XGBoost E1 protocol")
    if summary.get("feature_order") != list(F0_FEATURES) or summary.get("seeds") != list(E1_SEEDS):
        raise ValueError("CV summary has incompatible F0 features or seed schedule")
    params = summary.get("params")
    rounds = summary.get("num_boost_round")
    if not isinstance(params, dict) or params.get("objective") != "binary:logistic":
        raise ValueError("CV summary has no compatible XGBoost classification parameters")
    if type(rounds) is not int or rounds < 1 or not summary.get("config_id"):
        raise ValueError("CV summary has no valid configuration ID or boosting rounds")
    return summary


def holdout_training_rows(market: pd.DataFrame, target_ids: list[str], all_ids: list[str]) -> pd.DataFrame:
    """Use the shared holdout split without exposing its evaluation rows to training."""
    rows, _ = split_market(market, target_ids, HOLDOUT)
    if set(rows["stock_id"]) != set(all_ids):
        raise ValueError("Holdout training window must contain all 100 supplied stocks")
    return rows


def save_e1(
    cv_summary_path: Path,
    seed: int,
    output_dir: Path,
    data_dir: Path,
    f0_path: Path,
) -> Path:
    """Train on the shared holdout's training window, then save the artifact."""
    if output_dir.exists():
        raise FileExistsError(f"Choose a new artifact directory: {output_dir}")
    if seed not in E1_SEEDS:
        raise ValueError(f"Seed must be one of {E1_SEEDS}")
    summary = _read_cv_summary(cv_summary_path)

    target_ids, all_ids = load_metadata(data_dir)
    market = load_market(data_dir)
    rows = holdout_training_rows(market, target_ids, all_ids)
    f0 = load_f0_dataset(f0_path)
    train = join_f0(rows, f0, include_label=True)
    preprocessor = BaselinePreprocessor()
    x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
    params = {key: value for key, value in summary["params"].items() if key != "seed"}
    params["seed"] = seed
    train_start = time.perf_counter()
    booster = xgb.train(
        params,
        xgb.DMatrix(x_train, label=train["y"].to_numpy()),
        num_boost_round=summary["num_boost_round"],
    )
    train_seconds = time.perf_counter() - train_start

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.json"
    preprocessing_path = output_dir / "preprocessor.json"
    booster.save_model(str(model_path))
    with preprocessing_path.open("w", encoding="utf-8") as file:
        json.dump(preprocessor.to_state(), file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    feature_source = PROJECT_ROOT / "shared/features.py"
    preprocessing_source = PROJECT_ROOT / "shared/preprocessing.py"
    shutil.copyfile(feature_source, output_dir / "feature_definitions.py")
    shutil.copyfile(preprocessing_source, output_dir / "preprocessing_definitions.py")

    restored = BaselinePreprocessor.from_state(json.loads(preprocessing_path.read_text(encoding="utf-8")))
    probe = train.loc[:, list(F0_FEATURES)].head(3)
    if not np.allclose(
        booster.predict(xgb.DMatrix(preprocessor.transform(probe))),
        xgb.Booster(model_file=str(model_path)).predict(xgb.DMatrix(restored.transform(probe))),
        rtol=0, atol=0,
    ):
        raise ValueError("Saved model and preprocessing did not reproduce training predictions")

    manifest = {
        "artifact_version": 1,
        "status": "provisional_e1_pre_holdout",
        "model": "XGBoost",
        "config_id": summary["config_id"],
        "cv_summary_path": str(cv_summary_path),
        "cv_summary_sha256": _sha256(cv_summary_path),
        "training_split": HOLDOUT.name,
        "train_start": "2020-01-01",
        "train_end": HOLDOUT.train_end,
        "training_rows": len(train),
        "training_stocks": train["stock_id"].nunique(),
        "seed": seed,
        "params": params,
        "num_boost_round": summary["num_boost_round"],
        "feature_set": "F0",
        "feature_order": list(F0_FEATURES),
        "transformed_feature_order": list(preprocessor.feature_names_),
        "threshold": 0.5,
        "threshold_rule": summary["threshold_rule"],
        "f0_csv_sha256": _sha256(f0_path),
        "feature_definitions_sha256": _sha256(feature_source),
        "preprocessing_definitions_sha256": _sha256(preprocessing_source),
        "model_sha256": _sha256(model_path),
        "preprocessor_sha256": _sha256(preprocessing_path),
        "train_seconds": train_seconds,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Provisional E1 artifact saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV_SUMMARY,
                        help="CV summary for the selected configuration (default: full-sample E1 run)")
    parser.add_argument("--seed", type=int, choices=E1_SEEDS, default=SEED,
                        help=f"Prespecified reference seed (default: {SEED})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="New artifact directory (default: results/zhenci_e1_full_pre_holdout_artifact)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    args = parser.parse_args()
    save_e1(args.cv_summary, args.seed, args.output_dir, args.data_dir, args.f0_path)
