"""Refit the requested E17 variants through 2025 and export four Phase 1 CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xgboost as xgb

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from members.zhenci._archived_sources import archived_source_hash
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from members.zhenci.xgboost.e1.save_e1 import _sha256, holdout_training_rows
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY, load_control_summary, load_fixed_rule
from members.zhenci.xgboost.e17.train_e17 import DATA, F0, OUTPUT, ROOT, STRENGTHS, fit_classifier, sample_weights
from shared.data_loader import KEYS, load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT, SEED
from shared.submission import load_submission_template, write_submission

ARTIFACT = ROOT / "results/zhenci_e17_pre_phase1_artifact"
TEMPLATE = ROOT / "sample_submission.csv"
UPDATED_F0 = DATA / "processed/f0_with_updates.csv"
ARMS = ("E17-A", "E17-B")
FILES = ("preprocessor.json", *(f"{arm}.json" for arm in ARMS))


def freeze(artifact: Path, recipe: dict, expected: dict, target_ids: list[str], all_ids: list[str]) -> None:
    rows = holdout_training_rows(load_market(DATA), target_ids, all_ids)
    train = join_f0(rows, load_f0_dataset(F0), include_label=False)
    preprocessor = BaselinePreprocessor().fit(train[list(F0_FEATURES)])
    features = preprocessor.transform(train[list(F0_FEATURES)])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="e17-", dir=artifact.parent) as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        for arm in ARMS:
            model = fit_classifier(recipe, features, rows, sample_weights(rows, STRENGTHS[arm]))
            model.get_booster().save_model(str(staged / f"{arm}.json"))
        (staged / "preprocessor.json").write_text(json.dumps(preprocessor.to_state(), indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
        manifest = {**expected, "training_rows": len(rows), "training_stocks": len(all_ids),
                    "transformed_feature_order": list(features.columns),
                    "file_sha256": {name: _sha256(staged / name) for name in FILES}}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(artifact)


def export(artifact: Path, f0_path: Path, submission_dir: Path, cv_dir: Path = OUTPUT) -> list[Path]:
    destinations = {(arm, decision): submission_dir / f"zhenci_{arm.lower().replace('-', '_')}_{decision}_phase1.csv"
                    for arm in ARMS for decision in ("taiex", "050")}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(path)
    target_ids, all_ids = load_metadata(DATA)
    ids, days = load_submission_template(TEMPLATE, target_ids)
    if len(days) != 84 or min(days) <= HOLDOUT.train_end:
        raise ValueError("Expected the official Phase 1 template with 84 dates after training")
    features = require_template_features(load_f0_dataset(f0_path), ids, days)
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, _ = load_fixed_rule(DEFAULT_E3_SUMMARY, recipe["config_id"])
    cv_path = cv_dir / "summary.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    history_hash = _sha256(F0)
    cv_hash = cv.get("input_sha256", {}).get(str(F0.relative_to(ROOT)))
    if (cv.get("lambda") != STRENGTHS or cv.get("params") != recipe["params"] or cv.get("seed") != SEED
            or cv.get("rounds") != recipe["num_boost_round"] or cv.get("taiex_rule") != rule or cv_hash != history_hash):
        raise ValueError("E17 historical experiment inputs or recipe changed")
    training_source = Path(__file__).with_name("train_e17.py")
    expected = {"experiment": "E17", "train_start": "2020-01-01", "train_end": HOLDOUT.train_end,
                "seed": SEED, "feature_order": list(F0_FEATURES), "params": recipe["params"],
                "rounds": recipe["num_boost_round"], "lambda": {arm: STRENGTHS[arm] for arm in ARMS},
                "taiex_rule": rule, "fixed_threshold": .5, "cv_sha256": _sha256(cv_path), "f0_sha256": history_hash,
                "features_sha256": _sha256(ROOT / "shared/features.py"),
                "preprocessing_sha256": _sha256(ROOT / "shared/preprocessing.py"),
                "training_code_sha256": archived_source_hash(training_source, _sha256(training_source))}
    if not artifact.exists():
        freeze(artifact, recipe, expected, target_ids, all_ids)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen E17 manifest differs from experiment recipe")
    for name in FILES:
        if _sha256(artifact / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen E17 artifact changed: {name}")
    state = json.loads((artifact / "preprocessor.json").read_text(encoding="utf-8"))
    transformed = BaselinePreprocessor.from_state(state).transform(features[list(F0_FEATURES)])
    if list(transformed.columns) != manifest["transformed_feature_order"]:
        raise ValueError("Frozen E17 encoded feature order changed")
    matrix = xgb.DMatrix(transformed)
    saved = []
    for arm in ARMS:
        model = xgb.Booster(model_file=str(artifact / f"{arm}.json"))
        probability = model.predict(matrix)
        if probability.shape != (4200,) or not np.isfinite(probability).all():
            raise ValueError("Expected 4,200 valid probabilities")
        for decision, calls in (("taiex", regime_predictions(probability, features["taiex_return"].to_numpy(), rule)),
                                ("050", (probability > .5).astype(int))):
            path = write_submission(features[KEYS].assign(predicted_y=calls), TEMPLATE,
                                    destinations[arm, decision], target_ids)
            print(f"{path.name}: {int(calls.sum()):,} up calls", flush=True)
            saved.append(path)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--cv-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.artifact, args.f0_path, args.submission_dir, args.cv_dir)
