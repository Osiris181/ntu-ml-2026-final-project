"""Refit or reuse the historical-screened L2 candidate and export Phase 1 predictions."""

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

from members.zhenci.logistic_regression.l1.export_phase1 import (
    DATA, E1_CONFIG_ID, DEFAULT_E3_SUMMARY, TEMPLATE, TRAIN_START, UPDATED_F0, load_fixed_rule,
)
from members.zhenci.logistic_regression.l1.model import PARAMS, digest, read_json, write_json
from members.zhenci.logistic_regression.l2.diagnostics import SELECTION_POLICY, return_weights, summarize
from members.zhenci.logistic_regression.l2.train_l2 import (
    F0, HERE, OUTPUT, checked_l1_summary, fit_weighted, load_history,
)
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from shared.data_loader import KEYS, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT, SEED
from shared.submission import load_submission_template, write_submission

PHASE1 = HERE / "phase1"
ARTIFACT_FILES = ("model.joblib", "preprocessor.json")


def selected_recipe(history: Path) -> dict:
    summary_path = history / "summary.json"
    summary = read_json(summary_path)
    rule, _ = load_fixed_rule(DEFAULT_E3_SUMMARY, E1_CONFIG_ID)
    l1 = checked_l1_summary(rule)
    if (summary["selection_policy"] != SELECTION_POLICY or summary["params"] != PARAMS
            or summary["features"] != list(F0_FEATURES) or summary["seed"] != SEED
            or summary["thresholds"] != {"L2-A": .5, "L2-B": rule}
            or summary["versions"]["scikit_learn"] != sklearn.__version__):
        raise ValueError("Historical L2 settings differ from the controlled recipe")
    for name, expected in summary["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"L2 historical input changed: {name}")
    tables = [history / f"{name}.csv" for name in ("fold_metrics", "four_month_scores", "paired_uncertainty")]
    _, screen, selected = summarize(*(pd.read_csv(path, dtype={"fold": str}) for path in tables))
    if selected is None or selected != summary["candidate"]:
        raise ValueError("The locked historical screen did not select an L2 Phase 1 candidate")
    evidence = screen.loc[screen["arm"].eq(selected)].iloc[0].to_dict()
    return dict(candidate=selected, selection_policy=SELECTION_POLICY, historical_evidence=evidence,
                status="exploratory_phase1", threshold=summary["thresholds"][selected],
                params=PARAMS, features=list(F0_FEATURES), seed=SEED,
                versions={"scikit_learn": sklearn.__version__}, train_start=TRAIN_START,
                train_end=HOLDOUT.train_end, historical_result_sha256={
                    path.name: digest(path) for path in [summary_path, *tables]},
                training_input_sha256={name: value for name, value in l1["input_sha256"].items()
                                       if name.startswith("data/") or name.startswith("shared/")})


def freeze(artifact: Path, recipe: dict) -> None:
    for name, expected in recipe["training_input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Archived training data or shared input changed: {name}")
    _, universe, market, f0 = load_history()
    train = market.loc[market["date"].between(TRAIN_START, HOLDOUT.train_end)]
    if set(train["stock_id"]) != set(universe):
        raise ValueError("L2 final refit requires the complete 100-stock training universe")
    raw = join_f0(train, f0, include_label=False)
    weights, normalizer = return_weights(train)
    model, pre, features, fit_seconds = fit_weighted(raw, train["y"].to_numpy(), weights)
    artifact.mkdir()
    joblib.dump(model, artifact / ARTIFACT_FILES[0])
    write_json(artifact / ARTIFACT_FILES[1], pre.to_state())
    write_json(artifact / "manifest.json", {
        **recipe, "weight_normalizer": normalizer, "training_rows": len(train),
        "training_stocks": len(universe), "last_training_date": train["date"].max().date().isoformat(),
        "optimizer_iterations": int(model.n_iter_[0]), "fit_seconds": fit_seconds,
        "encoded_feature_order": list(features.columns),
        "file_sha256": {name: digest(artifact / name) for name in ARTIFACT_FILES},
    })


def predict(raw: pd.DataFrame, artifact: Path, recipe: dict) -> pd.DataFrame:
    manifest = read_json(artifact / "manifest.json")
    if any(manifest.get(name) != value for name, value in recipe.items()):
        raise ValueError("Frozen L2 recipe does not match the selected candidate")
    for name in ARTIFACT_FILES:
        if digest(artifact / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Frozen L2 file changed: {name}")
    pre = BaselinePreprocessor.from_state(read_json(artifact / ARTIFACT_FILES[1]))
    features = pre.transform(raw)
    if list(features.columns) != manifest["encoded_feature_order"]:
        raise ValueError("Frozen L2 encoded feature order changed")
    score = joblib.load(artifact / ARTIFACT_FILES[0]).predict_proba(features)[:, 1]
    if score.shape != (4200,) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Expected 4,200 valid L2 scores")
    threshold = recipe["threshold"]
    calls = (regime_predictions(score, raw["taiex_return"].to_numpy(), threshold)
             if isinstance(threshold, dict) else (score > threshold).astype(int))
    return raw[KEYS].assign(weighted_score=score, predicted_y=calls)


def export(history: Path, output: Path, f0_path: Path, template: Path,
           artifact: Path | None, submission: Path | None) -> Path:
    recipe = selected_recipe(history)
    name = recipe["candidate"].lower().replace("-", "_")
    submission = submission or ROOT / "submissions" / f"zhenci_{name}_exploratory_phase1.csv"
    if output.exists() or submission.exists():
        raise FileExistsError(f"Choose new L2 export paths: {output}, {submission}")
    submission_path = submission.resolve()
    if submission_path.is_relative_to(output.resolve()):
        raise ValueError("Place the submission CSV outside the artifact output directory")
    target_ids, _ = load_metadata(DATA)
    ids, dates = load_submission_template(template, target_ids)
    if len(dates) != 84 or min(dates) <= HOLDOUT.train_end:
        raise ValueError("Expected the 84 Phase 1 dates after the training cutoff")
    raw = require_template_features(load_f0_dataset(f0_path), ids, dates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l2-phase1-", dir=output.parent) as temporary:
        staged = Path(temporary) / "phase1"
        staged.mkdir()
        frozen = artifact if artifact is not None else staged / "artifact"
        if artifact is None:
            freeze(frozen, recipe)
        predictions = predict(raw, frozen, recipe)
        csv = write_submission(predictions, template, Path(temporary) / submission.name, target_ids)
        restored = pd.read_csv(csv, dtype={"id": str})
        if (restored.columns.tolist() != ["id", *dates] or restored["id"].tolist() != ids
                or not np.array_equal(restored[dates].to_numpy().ravel(), predictions["predicted_y"])):
            raise ValueError("L2 export did not round-trip in official template order")
        predictions.to_csv(staged / "phase1_predictions.csv", index=False)
        write_json(staged / "export_manifest.json", {
            "candidate": recipe["candidate"], "artifact": str(artifact.resolve()) if artifact else "artifact",
            "artifact_manifest_sha256": digest(frozen / "manifest.json"),
            "template_sha256": digest(template), "inference_f0_sha256": digest(f0_path),
            "submission": (submission_path.relative_to(ROOT).as_posix()
                           if submission_path.is_relative_to(ROOT) else str(submission_path)),
            "submission_sha256": digest(csv), "predictions": len(predictions),
            "predicted_up": int(predictions["predicted_y"].sum()),
        })
        submission.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--artifact", type=Path, help="Reuse the frozen L2 fit")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    print(export(args.history, args.output_dir, args.f0_path, args.template, args.artifact, args.submission))
