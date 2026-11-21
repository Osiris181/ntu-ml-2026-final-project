"""Prepare an exploratory E16 Phase 1 CSV from a frozen pre-2026 refit.

From final_project: uv run python -m members.zhenci.xgboost.e16.export_e16_phase1
Use --variant B for daily ranking with the same frozen artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xgboost as xgb

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci._archived_sources import archived_source_hash
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from members.zhenci.xgboost.e1.save_e1 import _sha256, holdout_training_rows
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e4.train_e4 import DEFAULT_E1_SUMMARY, load_control_summary
from members.zhenci.xgboost.e16.train_e16 import DEFAULT_OUTPUT, balanced_calls, relative_labels
from shared.data_loader import KEYS, load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT, SEED
from shared.submission import load_submission_template, write_submission

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data"
ORIGINAL_F0 = DATA / "processed/f0.csv"
UPDATED_F0 = DATA / "processed/f0_with_updates.csv"
TEMPLATE = ROOT / "sample_submission.csv"
ARTIFACT = ROOT / "results/zhenci_e16_a_pre_phase1_artifact"
SUBMISSION = ROOT / "submissions/zhenci_e16_a_exploratory_phase1.csv"
RANKED_SUBMISSION = ROOT / "submissions/zhenci_e16_b_exploratory_phase1.csv"
CUTOFF = .5


def freeze(artifact: Path, recipe: dict, expected: dict,
           target_ids: list[str], all_ids: list[str]) -> None:
    train = holdout_training_rows(load_market(DATA), target_ids, all_ids)
    features = join_f0(train, load_f0_dataset(ORIGINAL_F0), include_label=False)
    labels = relative_labels(train)["relative_up"].to_numpy()
    preprocessor = BaselinePreprocessor().fit(features[list(F0_FEATURES)])
    x_train = preprocessor.transform(features[list(F0_FEATURES)])
    model = xgb.train(expected["params"], xgb.DMatrix(x_train, label=labels),
                      num_boost_round=recipe["num_boost_round"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="e16-", dir=artifact.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        model.save_model(str(staging / "model.json"))
        (staging / "preprocessor.json").write_text(
            json.dumps(preprocessor.to_state(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {**expected, "training_rows": len(train), "training_stocks": len(all_ids),
                    "relative_positive_labels": int(labels.sum()),
                    "transformed_feature_order": list(x_train.columns),
                    "xgboost_version": xgb.__version__, "file_sha256": {
                        name: _sha256(staging / name) for name in ("model.json", "preprocessor.json")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(artifact)


def export(artifact: Path, updated_f0: Path, output: Path, *, variant: str = "A") -> Path:
    if variant not in {"A", "B"}:
        raise ValueError("Expected variant A or B")
    if output.exists():
        raise FileExistsError(output)
    target_ids, all_ids = load_metadata(DATA)
    ids, days = load_submission_template(TEMPLATE, target_ids)
    if len(days) != 84 or min(days) <= HOLDOUT.train_end:
        raise ValueError("Expected the official Phase 1 template with 84 dates after training")
    features = require_template_features(load_f0_dataset(updated_f0), ids, days)
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    cv_path = DEFAULT_OUTPUT / "summary.json"
    history_hash = _sha256(ORIGINAL_F0)
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    if (cv.get("params") != recipe["params"] or cv.get("rounds") != recipe["num_boost_round"]
            or cv.get("seed") != SEED or cv.get("input_sha256", {}).get("processed/f0.csv") != history_hash):
        raise ValueError("E16 validation recipe or training F0 changed")
    training_source = Path(__file__).with_name("train_e16.py")
    expected = {"experiment": "E16-A", "train_start": "2020-01-01", "train_end": HOLDOUT.train_end,
                "target": "adjusted intraday return strictly above the training date median",
                "threshold": CUTOFF, "seed": SEED, "feature_order": list(F0_FEATURES),
                "params": {**recipe["params"], "seed": SEED}, "rounds": recipe["num_boost_round"],
                "cv_sha256": _sha256(cv_path), "original_f0_sha256": history_hash,
                "feature_definitions_sha256": _sha256(ROOT / "shared/features.py"),
                "preprocessing_definitions_sha256": _sha256(ROOT / "shared/preprocessing.py"),
                "target_definition_sha256": archived_source_hash(training_source, _sha256(training_source))}
    if not artifact.exists():
        if variant == "B":
            raise FileNotFoundError(f"E16-B requires the frozen E16 model used for A: {artifact}")
        freeze(artifact, recipe, expected, target_ids, all_ids)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen E16-A manifest differs from the expected recipe")
    for name in ("model.json", "preprocessor.json"):
        if _sha256(artifact / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen E16-A file changed: {name}")
    state = json.loads((artifact / "preprocessor.json").read_text(encoding="utf-8"))
    x_valid = BaselinePreprocessor.from_state(state).transform(features[list(F0_FEATURES)])
    if list(x_valid.columns) != manifest["transformed_feature_order"]:
        raise ValueError("Frozen E16-A encoded feature order changed")
    model = xgb.Booster(model_file=str(artifact / "model.json"))
    probability = model.predict(xgb.DMatrix(x_valid))
    if (probability.shape != (4200,) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Expected 4,200 valid relative-performance probabilities")
    if variant == "B":
        calls = balanced_calls(features[KEYS], probability)
    else:
        calls = (probability > CUTOFF).astype(int)
    predictions = features[KEYS].assign(predicted_y=calls)
    if variant == "B":
        daily = predictions.groupby("date")["predicted_y"].agg(["size", "sum"])
        if len(daily) != 84 or not daily["size"].eq(50).all() or not daily["sum"].eq(25).all():
            raise ValueError("Expected 25 up and 25 down calls on all 84 Phase 1 dates")
    saved = write_submission(predictions, TEMPLATE, output, target_ids)
    print(f"Saved {len(predictions):,} predictions with {int(predictions['predicted_y'].sum()):,} up calls")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_output = RANKED_SUBMISSION if args.variant == "B" else SUBMISSION
    print(export(args.artifact, args.f0_path, args.output or default_output, variant=args.variant))
