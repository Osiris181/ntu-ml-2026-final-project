"""Freeze the historical L3 candidate and export an exploratory Phase 1 submission."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn

from members.zhenci.logistic_regression.l1.model import digest, read_json, write_json
from members.zhenci.logistic_regression.l2.diagnostics import realized_returns
from members.zhenci.ridge_regression.l3.diagnostics import REFERENCE, SELECTION_POLICY, summarize
from members.zhenci.ridge_regression.l3.train_l3 import (
    DATA, DELTA_MULTIPLIER, HERE, OUTPUT, PARAMS, fit_ridge, load_history, taiex_thresholds,
)
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from members.zhenci.xgboost.e1.train_e1 import join_f0
from shared.data_loader import KEYS, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES
from shared.splits import HOLDOUT, SEED
from shared.submission import load_submission_template, write_submission
from shared.updated_features import DEFAULT_OUTPUT as UPDATED_F0, DEFAULT_TEMPLATE as TEMPLATE

PHASE1 = HERE / "phase1"
FILES = ("model.joblib", "preprocessor.json")
TRAIN_START = "2020-01-01"


def selected_recipe(history: Path) -> dict:
    manifest = history / "summary.json"
    summary = read_json(manifest)
    if (summary["params"] != PARAMS or summary["selection_policy"] != SELECTION_POLICY
            or summary["reference"] != REFERENCE or summary["sample_weighting"] != "uniform"
            or summary["features"] != list(F0_FEATURES) or summary["seed"] != SEED
            or summary["delta_multiplier"] != DELTA_MULTIPLIER
            or summary["versions"]["scikit_learn"] != sklearn.__version__):
        raise ValueError("L3 historical recipe no longer matches the fixed configuration")
    for name, expected in summary["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"L3 historical input changed: {name}")
    tables = [history / f"{name}.csv" for name in ("fold_metrics", "four_month_scores", "paired_uncertainty")]
    _, screen, candidate = summarize(*(pd.read_csv(path, dtype={"fold": str}) for path in tables))
    if candidate is None or candidate != summary["candidate"]:
        raise ValueError("The historical L3 screen did not select a Phase 1 candidate")
    return {
        "candidate": candidate, "reference": REFERENCE, "selection_policy": SELECTION_POLICY,
        "historical_evidence": screen.loc[screen["arm"].eq(candidate)].iloc[0].to_dict(),
        "status": "exploratory_phase1", "params": PARAMS, "features": list(F0_FEATURES),
        "seed": SEED, "scikit_learn": sklearn.__version__, "target": summary["target"],
        "train_start": TRAIN_START, "train_end": HOLDOUT.train_end,
        "decision": summary["thresholds"][candidate], "delta_multiplier": DELTA_MULTIPLIER,
        "historical_result_sha256": {path.name: digest(path) for path in [manifest, *tables]},
    }


def freeze(artifact: Path, recipe: dict) -> None:
    _, universe, market, f0 = load_history()
    train = market.loc[market["date"].between(TRAIN_START, HOLDOUT.train_end)]
    if set(train["stock_id"]) != set(universe):
        raise ValueError("L3 final training requires the entire original stock universe")
    raw = join_f0(train, f0, include_label=False)
    target = realized_returns(train)
    model, pre, fit_time = fit_ridge(raw, target)
    mean_abs_return = float(np.abs(target).mean())
    artifact.mkdir()
    joblib.dump(model, artifact / FILES[0])
    write_json(artifact / FILES[1], pre.to_state())
    write_json(artifact / "manifest.json", {
        **recipe, "observed_solver": getattr(model, "solver_", "unavailable"),
        "training_rows": len(train), "training_stocks": len(universe),
        "last_training_date": train["date"].max().date().isoformat(), "fit_seconds": fit_time,
        "mean_abs_training_return": mean_abs_return, "delta": DELTA_MULTIPLIER * mean_abs_return,
        "encoded_feature_order": list(pre.feature_names_),
        "file_sha256": {name: digest(artifact / name) for name in FILES},
    })


def predict(raw: pd.DataFrame, artifact: Path, recipe: dict, *, expected_rows: int = 4200) -> pd.DataFrame:
    manifest = read_json(artifact / "manifest.json")
    if any(manifest.get(key) != value for key, value in recipe.items()):
        raise ValueError("Frozen L3 artifact does not match the selected recipe")
    for name in FILES:
        if digest(artifact / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen L3 file changed: {name}")
    if manifest["delta"] != DELTA_MULTIPLIER * manifest["mean_abs_training_return"]:
        raise ValueError("Frozen L3 threshold differs from the training-only scale")
    features = BaselinePreprocessor.from_state(read_json(artifact / FILES[1])).transform(raw)
    if features.columns.tolist() != manifest["encoded_feature_order"]:
        raise ValueError("Frozen L3 encoded feature order changed")
    predicted = joblib.load(artifact / FILES[0]).predict(features)
    if predicted.shape != (expected_rows,) or not np.isfinite(predicted).all():
        raise ValueError(f"Expected {expected_rows:,} finite Ridge return predictions")
    cutoff = (0 if recipe["candidate"] == "L3-A"
              else taiex_thresholds(raw["taiex_return"].to_numpy(), manifest["delta"]))
    return raw[KEYS].assign(predicted_return=predicted, decision_threshold=cutoff,
                            predicted_y=(predicted > cutoff).astype(int))


def export(history: Path, output: Path, f0_path: Path, template: Path,
           artifact: Path | None, submission: Path | None) -> Path:
    recipe = selected_recipe(history)
    name = recipe["candidate"].lower().replace("-", "_")
    submission = submission or ROOT / "submissions" / f"zhenci_{name}_exploratory_phase1.csv"
    if output.exists() or submission.exists():
        raise FileExistsError(f"Choose a new L3 output and submission path: {output}, {submission}")
    resolved = submission.resolve()
    if resolved.is_relative_to(output.resolve()):
        raise ValueError("Place the submission CSV outside the artifact output directory")
    targets, _ = load_metadata(DATA)
    ids, dates = load_submission_template(template, targets)
    if len(dates) != 84 or min(dates) <= HOLDOUT.train_end:
        raise ValueError("Expected the 84 Phase 1 dates after the training cutoff")
    raw = require_template_features(load_f0_dataset(f0_path), ids, dates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l3-phase1-", dir=output.parent) as temp:
        staged = Path(temp) / "phase1"
        staged.mkdir()
        frozen = artifact if artifact is not None else staged / "artifact"
        if artifact is None:
            freeze(frozen, recipe)
        predictions = predict(raw, frozen, recipe)
        csv = write_submission(predictions, template, Path(temp) / submission.name, targets)
        restored = pd.read_csv(csv, dtype={"id": str})
        if (restored.columns.tolist() != ["id", *dates] or restored["id"].tolist() != ids
                or not np.array_equal(restored[dates].to_numpy().ravel(), predictions["predicted_y"])):
            raise ValueError("L3 export did not round-trip in the official order")
        predictions.to_csv(staged / "phase1_predictions.csv", index=False)
        write_json(staged / "export_manifest.json", {
            "candidate": recipe["candidate"], "artifact": str(artifact.resolve()) if artifact else "artifact",
            "artifact_manifest_sha256": digest(frozen / "manifest.json"),
            "template_sha256": digest(template), "inference_f0_sha256": digest(f0_path),
            "submission": resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved),
            "submission_sha256": digest(csv), "predictions": len(predictions),
            "predicted_up": int(predictions["predicted_y"].sum()),
        })
        submission.parent.mkdir(parents=True, exist_ok=True)
        # Create the final CSV under its destination ACL, including on Windows.
        stream = submission.open("xb")
        try:
            with stream:
                stream.write(csv.read_bytes())
            if output.exists():
                raise FileExistsError(output)
            staged.rename(output)
        except BaseException:
            submission.unlink()
            raise
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=PHASE1)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--artifact", type=Path, help="Reuse the frozen Ridge fit")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    print(export(args.history, args.output_dir, args.f0_path, args.template, args.artifact, args.submission))
