"""Export exploratory Phase 1 predictions with either predefined L1 rule."""

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

from members.zhenci.logistic_regression.l1.model import PARAMS, digest, fit_l1, read_json, write_json
from members.zhenci.logistic_regression.l1.train_l1 import DATA, F0, OUTPUT, load_history
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from members.zhenci.xgboost.e1.train_e1 import join_f0
from members.zhenci.xgboost.e3.evaluate_e3_market_regime import regime_predictions
from members.zhenci.xgboost.e4.train_e4 import DEFAULT_E3_SUMMARY, E1_CONFIG_ID, load_fixed_rule
from shared.data_loader import KEYS, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT
from shared.submission import load_submission_template, write_submission
from shared.updated_features import DEFAULT_OUTPUT as UPDATED_F0, DEFAULT_TEMPLATE as TEMPLATE

PHASE1 = Path(__file__).resolve().with_name("phase1")
SUBMISSION = "zhenci_l1_a_exploratory_phase1.csv"
DEFAULT_SUBMISSION = ROOT / "submissions" / SUBMISSION
PHASE1_B = PHASE1.with_name("phase1_b")
SUBMISSION_B = DEFAULT_SUBMISSION.with_name("zhenci_l1_b_exploratory_phase1.csv")
FILES = ("model.joblib", "preprocessor.json")
THRESHOLD = 0.5
TRAIN_START = "2020-01-01"


def recipe(cv_summary: Path) -> dict:
    cv = read_json(cv_summary)
    _, universe = load_metadata(DATA)
    sources = [F0, ROOT / "shared/features.py", ROOT / "shared/preprocessing.py",
               Path(__file__).with_name("model.py"), DATA / "info.json",
               *(DATA / "market" / f"{stock}.csv" for stock in universe)]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    if (cv["model"] != "LogisticRegression" or cv["params"] != PARAMS
            or cv["features"] != list(F0_FEATURES) or cv["thresholds"]["L1-A"] != THRESHOLD
            or any(cv["input_sha256"].get(name) != value for name, value in hashes.items())):
        raise ValueError("L1-A recipe or training inputs differ from historical evaluation")
    return {
        "experiment": "L1-A", "status": "user_requested_exploratory_phase1",
        "cv_recommended_candidate": cv["candidate"], "cv_summary_sha256": digest(cv_summary),
        "train_start": TRAIN_START, "train_end": HOLDOUT.train_end,
        "target": "adjusted close strictly greater than adjusted open",
        "threshold": THRESHOLD, "params": PARAMS, "scikit_learn": sklearn.__version__,
        "source_sha256": hashes,
    }


def freeze(artifact: Path, expected: dict) -> None:
    _, universe, market, f0 = load_history()
    train = market.loc[market["date"].between(TRAIN_START, HOLDOUT.train_end)]
    if set(train["stock_id"]) != set(universe):
        raise ValueError("L1-A refit requires the original 100-stock universe")
    raw = join_f0(train, f0, include_label=False)
    model, pre, features, fit_seconds = fit_l1(raw, train["y"].to_numpy())
    artifact.mkdir()
    joblib.dump(model, artifact / FILES[0])
    write_json(artifact / FILES[1], pre.to_state())
    write_json(artifact / "manifest.json", {
        **expected, "training_rows": len(train), "training_stocks": len(universe),
        "last_training_date": train["date"].max().date().isoformat(),
        "optimizer_iterations": int(model.n_iter_[0]), "fit_seconds": fit_seconds,
        "encoded_feature_order": list(features.columns),
        "file_sha256": {name: digest(artifact / name) for name in FILES},
    })


def predict(artifact: Path, expected: dict, raw: pd.DataFrame, taiex_rule: dict | None = None) -> pd.DataFrame:
    saved = read_json(artifact / "manifest.json")
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen L1-A model does not match the fixed Phase 1 recipe")
    for name in FILES:
        if digest(artifact / name) != saved["file_sha256"][name]:
            raise ValueError(f"Frozen L1-A file changed: {name}")
    pre = BaselinePreprocessor.from_state(read_json(artifact / FILES[1]))
    features = pre.transform(raw)
    if list(features.columns) != saved["encoded_feature_order"]:
        raise ValueError("L1-A encoded feature order changed")
    model = joblib.load(artifact / FILES[0])
    p_up = model.predict_proba(features)[:, 1]
    if p_up.shape != (4200,) or not np.isfinite(p_up).all() or np.any((p_up < 0) | (p_up > 1)):
        raise ValueError("Expected 4,200 valid L1-A upward probabilities")
    calls = ((p_up > saved["threshold"]).astype(int) if taiex_rule is None
             else regime_predictions(p_up, raw["taiex_return"].to_numpy(), taiex_rule))
    return raw[KEYS].assign(p_up=p_up, predicted_y=calls)


def export(output: Path, f0_path: Path, template: Path, cv_summary: Path,
           artifact: Path | None, submission: Path | None = None, *, variant: str = "A") -> Path:
    if variant not in {"A", "B"}:
        raise ValueError("Expected L1 variant A or B")
    submission = submission or (DEFAULT_SUBMISSION if variant == "A" else SUBMISSION_B)
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    if submission.exists():
        raise FileExistsError(f"Choose a new submission path: {submission}")
    submission_path = submission.resolve()
    if submission_path.is_relative_to(output.resolve()):
        raise ValueError("Place the submission CSV outside the artifact output directory")
    submission_location = (submission_path.relative_to(ROOT).as_posix()
                           if submission_path.is_relative_to(ROOT) else str(submission_path))
    target_ids, _ = load_metadata(DATA)
    ids, dates = load_submission_template(template, target_ids)
    if len(dates) != 84 or min(dates) <= HOLDOUT.train_end:
        raise ValueError("Expected 84 Phase 1 dates after the L1-A training cutoff")
    raw = require_template_features(load_f0_dataset(f0_path), ids, dates)
    expected = recipe(cv_summary)
    taiex_rule = None
    if variant == "B":
        artifact = artifact if artifact is not None else PHASE1 / "artifact"
        if not artifact.is_dir():
            raise FileNotFoundError(f"L1-B requires the existing L1-A frozen model: {artifact}")
        taiex_rule, _ = load_fixed_rule(DEFAULT_E3_SUMMARY, E1_CONFIG_ID)
        cv = read_json(cv_summary)
        if (cv["thresholds"]["L1-B"] != taiex_rule
                or cv["input_sha256"][DEFAULT_E3_SUMMARY.relative_to(ROOT).as_posix()] != digest(DEFAULT_E3_SUMMARY)):
            raise ValueError("L1-B must use the TAIEX rule from the historical L1 evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="l1-phase1-", dir=output.parent) as temporary:
        staged = Path(temporary) / "phase1"
        staged.mkdir()
        frozen = artifact if artifact is not None else staged / "artifact"
        if artifact is None:
            freeze(frozen, expected)
        predictions = predict(frozen, expected, raw, taiex_rule)
        csv = write_submission(predictions, template, Path(temporary) / submission.name, target_ids)
        restored = pd.read_csv(csv, dtype={"id": str})
        if (restored.columns.tolist() != ["id", *dates] or restored["id"].tolist() != ids
                or not np.array_equal(restored[dates].to_numpy().ravel(), predictions["predicted_y"])):
            raise ValueError("L1 export did not round-trip in official template order")
        predictions.to_csv(staged / "phase1_predictions.csv", index=False)
        write_json(staged / "export_manifest.json", {
            "experiment": f"L1-{variant}",
            "decision_rule": THRESHOLD if taiex_rule is None else taiex_rule,
            "artifact": str(artifact.resolve()) if artifact is not None else "artifact",
            "artifact_manifest_sha256": digest(frozen / "manifest.json"),
            "template_sha256": digest(template), "inference_f0_sha256": digest(f0_path),
            "submission": submission_location, "submission_sha256": digest(csv),
            "first_date": min(dates), "last_date": max(dates),
            "predictions": len(predictions), "predicted_up": int(predictions["predicted_y"].sum()),
        })
        if output.exists():
            raise FileExistsError(output)
        submission.parent.mkdir(parents=True, exist_ok=True)
        stream = submission.open("xb")
        try:
            with stream:
                stream.write(csv.read_bytes())
            staged.rename(output)
        except BaseException:
            submission.unlink()
            raise
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--cv-summary", type=Path, default=OUTPUT / "summary.json")
    parser.add_argument("--artifact", type=Path, help="Reuse an existing frozen L1-A artifact")
    args = parser.parse_args()
    output = args.output_dir or (PHASE1 if args.variant == "A" else PHASE1_B)
    print(export(output, args.f0_path, args.template, args.cv_summary, args.artifact, args.submission,
                 variant=args.variant))
