"""Rebuild one session's shared F0 inputs and export predictions with frozen L3-B.

From the repository root, the defaults use the November 21 data bundle and
its November 23 template. For another session pass --update-dir, --template,
and --previous-session. The CSV is placed under submissions by default.
"""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[4]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn

from members.zhenci.logistic_regression.l1.model import digest, read_json
from members.zhenci.ridge_regression.l3.export_phase1 import DATA, OUTPUT, PHASE1, predict
from members.zhenci.xgboost.e1.predict_e1 import require_template_features
from shared.data_loader import load_global, load_market, load_metadata
from shared.evaluation import CLOSE, OPEN
from shared.features import build_f0_dataset
from shared.preprocessing import F0_FEATURES
from shared.splits import HOLDOUT
from shared.submission import load_submission_template, write_submission
from shared.updated_features import add_template_keys, combine_feature_history, load_updated_market

DEFAULT_UPDATE = ROOT / "updated_data_21_Nov"
DEFAULT_TEMPLATE = DEFAULT_UPDATE / "sample_submission (3).csv"
DEFAULT_DATE = "2026-11-23"
DEFAULT_PREVIOUS_SESSION = "2026-11-20"
ARTIFACT = PHASE1 / "artifact"


def frozen_recipe() -> dict:
    """Check the existing final Ridge fit and shared source definitions."""
    manifest_path = ARTIFACT / "manifest.json"
    provenance = read_json(PHASE1 / "export_manifest.json")
    if provenance["candidate"] != "L3-B" or provenance["artifact_manifest_sha256"] != digest(manifest_path):
        raise ValueError("L3-B artifact differs from the frozen Phase 1 export")
    manifest = read_json(manifest_path)
    expected = {"candidate": "L3-B", "train_end": HOLDOUT.train_end,
                "last_training_date": HOLDOUT.train_end, "training_stocks": 100,
                "features": list(F0_FEATURES), "scikit_learn": sklearn.__version__,
                "decision": "negative +delta, nonnegative -delta, missing 0"}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected frozen L3-B model, feature order, or scikit-learn version")
    history = OUTPUT / "summary.json"
    if digest(history) != manifest["historical_result_sha256"]["summary.json"]:
        raise ValueError("L3-B historical manifest has changed")
    source_hashes = read_json(history)["input_sha256"]
    for name in ("shared/features.py", "shared/preprocessing.py"):
        if digest(ROOT / name) != source_hashes[name]:
            raise ValueError(f"Frozen L3-B definition changed: {name}")
    return expected


def phase2_features(template: Path, update_dir: Path, previous_session: str | None):
    target_ids, universe = load_metadata(DATA)
    ids, dates = load_submission_template(template, target_ids)
    if len(dates) != 1:
        raise ValueError("Phase 2 requires a template with one prediction date")
    day = dates[0]
    prediction_date = pd.Timestamp(day)
    if previous_session is None:
        if day != DEFAULT_DATE:
            raise ValueError("Pass --previous-session for this template")
        previous_session = DEFAULT_PREVIOUS_SESSION
    prior = pd.Timestamp(previous_session)
    if not pd.Timestamp(HOLDOUT.train_end) < prior < prediction_date:
        raise ValueError("The prior stock session must be after training and before the prediction date")

    data_dir = update_dir / "data" if (update_dir / "data").is_dir() else update_dir
    updated_market = load_updated_market(data_dir, universe)
    updated_global = load_global(data_dir)
    if updated_market["date"].max() != prior or updated_global["date"].ge(prediction_date).any():
        raise ValueError("Stock history must end on the previous session and global rows must predate the template")
    recent = updated_market.loc[updated_market["stock_id"].isin(target_ids)]
    latest = recent.groupby("stock_id")["date"].max().reindex(target_ids)
    prior_prices = recent.loc[recent["date"].eq(prior), [OPEN, CLOSE]].to_numpy(dtype=float)
    if not latest.eq(prior).all() or not (np.isfinite(prior_prices) & (prior_prices > 0)).all():
        raise ValueError("The previous session needs completed opening and closing prices for all 50 target stocks")

    market, global_data = combine_feature_history(load_market(DATA), updated_market,
                                                   load_global(DATA), updated_global)
    market = add_template_keys(market, template, target_ids)
    f0 = build_f0_dataset(market, global_data)
    return require_template_features(f0, ids, dates), prior.date().isoformat(), day, target_ids, ids


def export_phase2(template: Path = DEFAULT_TEMPLATE, update_dir: Path = DEFAULT_UPDATE,
                  previous_session: str | None = None, output: Path | None = None) -> Path:
    recipe = frozen_recipe()
    raw, prior, day, target_ids, ids = phase2_features(template, update_dir, previous_session)
    output = output or ROOT / "submissions" / f"zhenci_l3_b_phase2_{day}.csv"
    if output.resolve() == template.resolve():
        raise ValueError("Submission output must not overwrite the template")
    predictions = predict(raw, ARTIFACT, recipe, expected_rows=50)
    with TemporaryDirectory(prefix="l3-phase2-") as temp:
        candidate = write_submission(predictions, template, Path(temp) / "submission.csv", target_ids)
        restored = pd.read_csv(candidate, dtype={"id": str})
        if (restored.columns.tolist() != ["id", day] or restored["id"].tolist() != ids
                or not np.array_equal(restored[day], predictions["predicted_y"])):
            raise ValueError("Phase 2 CSV differs from the template or predictions")
        content = candidate.read_bytes()
    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError(f"Existing submission differs. Choose another --output: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            file.write(content)
    up = int(predictions["predicted_y"].sum())
    print(f"{output}\nDate {day}, prior stock session {prior}, 50 predictions ({up} up, {50-up} down)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--update-dir", type=Path, default=DEFAULT_UPDATE)
    parser.add_argument("--previous-session")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export_phase2(args.template, args.update_dir, args.previous_session, args.output)
