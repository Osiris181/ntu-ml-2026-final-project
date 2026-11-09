"""Refit a CV-winning E3 regime configuration on pre-holdout history.

Run only after train_e3_regime has completed and selected an alternative
that beats its E1 control. This does not inspect the holdout or submit.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
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
from members.zhenci.e3.train_e3_regime import CANDIDATES, NUM_BOOST_ROUND, PARAMS, recency_weights, select_training_rows
from shared.data_loader import load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import E1_SEEDS, HOLDOUT, SEED, split_market

DEFAULT_CV_SUMMARY = PROJECT_ROOT / "results/zhenci_xgboost_e3_regime_v1/summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/zhenci_e3_regime_pre_holdout_artifact"


def chosen_configuration(summary: dict) -> dict:
    """Reject a mismatched summary or an E3 run that did not beat E1."""
    expected = {"model": "XGBoost", "experiment": "E2", "feature_set": "F0",
                "evaluation_universe": "Top50@Y", "split_version": "1.2",
                "threshold_rule": "predict 1 when p_up > 0.5, else 0"}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("CV summary does not match the E3 regime protocol")
    if summary.get("feature_order") != list(F0_FEATURES) or summary.get("seeds") != list(E1_SEEDS):
        raise ValueError("CV summary has incompatible F0 features or seeds")
    if len(summary.get("configurations", [])) != len(CANDIDATES):
        raise ValueError("CV summary is missing candidate results")
    for recorded, expected_candidate in zip(summary["configurations"], CANDIDATES):
        if any(recorded.get(key) != value for key, value in expected_candidate.items()):
            raise ValueError("CV candidate definition changed")
        if recorded.get("params") != PARAMS or recorded.get("num_boost_round") != NUM_BOOST_ROUND:
            raise ValueError("CV XGBoost training settings changed")
    selected_id = summary.get("selected_config_id")
    if selected_id == CANDIDATES[0]["config_id"] or summary.get("improvement_over_e1", 0) <= 0:
        baseline = summary["configurations"][0]["mean_cv_official_score"]
        best_alternative = max(
            summary["configurations"][1:], key=lambda item: item["mean_cv_official_score"]
        )
        raise ValueError(
            "save_e3_regime cannot create an E3 artifact: validation selected the E1 control "
            f"({baseline:.8f}); the best E3 alternative, {best_alternative['config_id']}, "
            f"scored {best_alternative['mean_cv_official_score']:.8f}."
        )
    selected = next((candidate for candidate in CANDIDATES if candidate["config_id"] == selected_id), None)
    if selected is None:
        raise ValueError("Unknown selected E3 configuration")
    scores = {item["config_id"]: item["mean_cv_official_score"] for item in summary["configurations"]}
    if scores[selected_id] != max(scores.values()) or summary["baseline_cv_score"] != scores[CANDIDATES[0]["config_id"]]:
        raise ValueError("CV selection and recorded scores disagree")
    return selected


def save_e3(cv_summary_path: Path, output_dir: Path, data_dir: Path, f0_path: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new artifact directory: {output_dir}")
    with cv_summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    selected = chosen_configuration(summary)

    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    all_train_rows, _ = split_market(market, target_ids, HOLDOUT)
    train_rows = select_training_rows(all_train_rows, target_ids, selected["training_universe"])
    f0 = load_f0_dataset(f0_path)
    train = join_f0(train_rows, f0, include_label=True)
    preprocessor = BaselinePreprocessor()
    x_train = preprocessor.fit_transform(train.loc[:, list(F0_FEATURES)])
    weights = recency_weights(train["date"], HOLDOUT.train_end, selected["half_life_days"])
    params = {**PARAMS, "seed": SEED}
    started = time.perf_counter()
    booster = xgb.train(params, xgb.DMatrix(x_train, label=train["y"].to_numpy(), weight=weights),
                        num_boost_round=NUM_BOOST_ROUND)
    train_seconds = time.perf_counter() - started

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
    original = booster.predict(xgb.DMatrix(preprocessor.transform(probe)))
    reloaded = xgb.Booster(model_file=str(model_path)).predict(xgb.DMatrix(restored.transform(probe)))
    if not np.allclose(original, reloaded, rtol=0, atol=0):
        raise ValueError("Saved E3 model and preprocessing did not reproduce predictions")

    manifest = {
        "artifact_version": 1, "status": "provisional_e3_pre_holdout",
        "model": "XGBoost", "config_id": selected["config_id"],
        "cv_summary_path": str(cv_summary_path), "cv_summary_sha256": _sha256(cv_summary_path),
        "training_split": HOLDOUT.name, "train_start": "2020-01-01", "train_end": HOLDOUT.train_end,
        "training_rows": len(train), "training_stocks": train["stock_id"].nunique(),
        "training_universe": selected["training_universe"], "half_life_days": selected["half_life_days"],
        "seed": SEED, "params": params, "num_boost_round": NUM_BOOST_ROUND,
        "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "transformed_feature_order": list(preprocessor.feature_names_),
        "threshold": 0.5, "threshold_rule": summary["threshold_rule"],
        "f0_csv_sha256": _sha256(f0_path),
        "feature_definitions_sha256": _sha256(feature_source),
        "preprocessing_definitions_sha256": _sha256(preprocessing_source),
        "model_sha256": _sha256(model_path), "preprocessor_sha256": _sha256(preprocessing_path),
        "train_seconds": train_seconds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "xgboost": xgb.__version__},
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Provisional E3 artifact saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-summary", type=Path, default=DEFAULT_CV_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=PROJECT_ROOT / "data/processed/f0.csv")
    args = parser.parse_args()
    save_e3(args.cv_summary, args.output_dir, args.data_dir, args.f0_path)
