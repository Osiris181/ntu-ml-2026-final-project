"""Refit frozen E18-E through 2025 and export the requested exploratory Phase 1 CSV."""

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
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import DEFAULT_E1_SUMMARY, DEFAULT_E3_SUMMARY, load_control_summary, load_fixed_rule
from members.zhenci.xgboost.e17.train_e17 import fit_classifier
from members.zhenci.xgboost.e18.features_e18 import CATALOGUE, EXTRAS, ExpandedPreprocessor, build_features
from members.zhenci.xgboost.e18.train_e18 import DATA, F0, HERE, OUTPUT, ROOT, check_catalogue
from shared.data_loader import KEYS, load_global, load_market, load_metadata
from shared.features import build_f0_dataset, load_f0_dataset
from shared.preprocessing import F0_FEATURES
from shared.splits import HOLDOUT, SEED
from shared.submission import load_submission_template, write_submission
from shared.updated_features import DEFAULT_UPDATE, add_template_keys, combine_feature_history, load_updated_market

TEMPLATE = ROOT / "sample_submission.csv"
ARTIFACT = ROOT / "results/zhenci_e18_e_pre_phase1_artifact"
SUBMISSION = ROOT / "submissions/zhenci_e18_e_exploratory_phase1.csv"
FILES = ("model.json", "preprocessing.json")


def freeze(artifact: Path, metadata: dict, market: pd.DataFrame, target_ids: list[str],
           all_ids: list[str], recipe: dict) -> None:
    rows = holdout_training_rows(market, target_ids, all_ids)
    expanded, _, _ = build_features(rows, rows[KEYS], all_ids)
    f0 = load_f0_dataset(F0)
    inputs = join_f0(rows, f0, include_label=False).merge(
        expanded[KEYS + list(EXTRAS)], on=KEYS, validate="one_to_one", sort=False)
    if not inputs[KEYS].equals(rows[KEYS]):
        raise ValueError("Training features changed row alignment")
    processor = ExpandedPreprocessor(EXTRAS).fit(inputs)
    transformed = processor.transform(inputs)
    model = fit_classifier(recipe, transformed, rows, pd.Series(1.0, index=rows.index))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="e18-", dir=artifact.parent) as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        model.get_booster().save_model(str(staged / "model.json"))
        (staged / "preprocessing.json").write_text(
            json.dumps(processor.state(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {**metadata, "training_rows": len(rows), "training_stocks": len(all_ids),
                    "encoded_feature_order": list(transformed.columns),
                    "file_sha256": {name: _sha256(staged / name) for name in FILES}}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(artifact)


def export(artifact: Path, update: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    check_catalogue()
    target_ids, all_ids = load_metadata(DATA)
    ids, dates = load_submission_template(TEMPLATE, target_ids)
    if len(dates) != 84 or min(dates) <= HOLDOUT.train_end:
        raise ValueError("Expected the 84-date Phase 1 template after the training cutoff")
    recipe = load_control_summary(DEFAULT_E1_SUMMARY)
    rule, _ = load_fixed_rule(DEFAULT_E3_SUMMARY, recipe["config_id"])
    cv_path = OUTPUT / "summary.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    inputs = (F0, CATALOGUE, HERE / "features_e18.py", ROOT / "shared/features.py", ROOT / "shared/preprocessing.py")
    recorded = cv["input_sha256"]
    for path in inputs:
        current = _sha256(path)
        key = path.relative_to(ROOT).as_posix()
        expected = recorded.get(key)
        if expected is None and path.parent == HERE:
            expected = recorded.get(f"members/zhenci/e18/{path.name}")
            current = archived_source_hash(path, current)
        if expected != current:
            raise ValueError(f"Input differs from the E18 historical run: {path.name}")
    if (cv["arm_extras"]["E18-E"] != list(EXTRAS) or cv["params"] != recipe["params"]
            or cv["rounds"] != recipe["num_boost_round"] or cv["seed"] != SEED or cv["taiex_rule"] != rule):
        raise ValueError("E18-E recipe changed since the historical run")
    for stock in all_ids:
        if cv["market_sha256"].get(stock) != _sha256(DATA / "market" / f"{stock}.csv"):
            raise ValueError(f"Original market data changed for {stock}")

    market = load_market(DATA)
    feature_market, global_data = combine_feature_history(
        market, load_updated_market(update, all_ids), load_global(DATA), load_global(update))
    last = pd.Timestamp(max(dates))
    feature_market = feature_market.loc[feature_market["date"].lt(last)]
    global_data = global_data.loc[global_data["date"].lt(last)]
    # The final date is inserted only as a key. Earlier dates use strictly lagged inputs.
    f0_market = add_template_keys(feature_market, TEMPLATE, target_ids)
    aligned = require_template_features(build_f0_dataset(f0_market, global_data), ids, dates)
    expanded, _, _ = build_features(feature_market, aligned[KEYS], all_ids)
    prediction_inputs = aligned.merge(expanded[KEYS + list(EXTRAS)], on=KEYS,
                                      validate="one_to_one", sort=False)
    if len(prediction_inputs) != 4200 or not prediction_inputs[KEYS].equals(aligned[KEYS]):
        raise ValueError("Expanded features do not preserve Phase 1 keys")

    metadata = {"experiment": "E18-E", "status": "user_requested_exploratory_phase1",
                "train_start": "2020-01-01", "train_end": HOLDOUT.train_end, "seed": SEED,
                "params": recipe["params"], "rounds": recipe["num_boost_round"], "taiex_rule": rule,
                "features": [*F0_FEATURES, *EXTRAS], "historical_summary_sha256": _sha256(cv_path),
                "xgboost_version": xgb.__version__}
    if not artifact.exists():
        freeze(artifact, metadata, market, target_ids, all_ids, recipe)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError("Frozen E18-E manifest differs from the requested recipe")
    for name in FILES:
        if _sha256(artifact / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen artifact changed: {name}")
    state = json.loads((artifact / "preprocessing.json").read_text(encoding="utf-8"))
    transformed = ExpandedPreprocessor.from_state(state).transform(prediction_inputs)
    if list(transformed.columns) != manifest["encoded_feature_order"] or not np.isfinite(transformed.to_numpy()).all():
        raise ValueError("Invalid Phase 1 encoded features")
    model = xgb.Booster(model_file=str(artifact / "model.json"))
    p_up = model.predict(xgb.DMatrix(transformed))
    calls = regime_predictions(p_up, prediction_inputs["taiex_return"].to_numpy(), rule)
    path = write_submission(prediction_inputs[KEYS].assign(predicted_y=calls), TEMPLATE, output, target_ids)
    print(f"{len(calls):,} predictions, {int(calls.sum()):,} up calls")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--update", type=Path, default=DEFAULT_UPDATE)
    parser.add_argument("--output", type=Path, default=SUBMISSION)
    args = parser.parse_args()
    print(export(args.artifact, args.update, args.output))
