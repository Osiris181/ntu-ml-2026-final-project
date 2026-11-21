"""Build fresh F0 and export one Phase 2 CSV with the fixed E3 approach.

Run from final_project for November 18:
    uv run python -m members.zhenci.xgboost.e3.export_phase2

For later sessions, supply their actual template, cumulative updated data folder,
and previous completed stock session. This script never trains or uploads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from members.zhenci.xgboost.e3.predict_e3_market_regime import (
    DEFAULT_ARTIFACT, DEFAULT_SUMMARY, export_predictions,
)
from shared.data_loader import load_global, load_metadata
from shared.submission import load_submission_template
from shared.updated_features import generate_updated_f0, load_updated_market

BASE_DATA = ROOT / "data"
DEFAULT_UPDATE = ROOT / "updated_data_17_Nov"
DEFAULT_TEMPLATE = ROOT / "sample_submission_2026-11-18.csv"
DEFAULT_PREVIOUS_SESSION = "2026-11-17"
EXPECTED_ARTIFACT_SHA256 = "7abbecfaabee2325c8d75c5f95a6b755f9be316162a9065937a5b5e2e5c28540"
EXPECTED_RULE_SHA256 = "24f06993203c719706e1ecb8050f018c023e197cd3efdb4f449a69fc1eeae2a9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(template: Path, update_dir: Path,
                    previous_session: str | None) -> str:
    """Require the declared prior stock session and only pre-target sources."""
    target_ids, universe = load_metadata(BASE_DATA)
    ids, dates = load_submission_template(template, target_ids)
    if len(ids) != 50 or len(dates) != 1:
        raise ValueError("Phase 2 needs exactly one date and 50 target stocks")
    prediction_date = pd.Timestamp(dates[0])
    if previous_session is None:
        if dates[0] != "2026-11-18":
            raise ValueError("Pass --previous-session for this session's template")
        previous_session = DEFAULT_PREVIOUS_SESSION
    prior_date = pd.Timestamp(previous_session)
    if prior_date >= prediction_date:
        raise ValueError("Specify the actual previous completed trading session")
    market = load_updated_market(update_dir, universe)
    if market["date"].max() >= prediction_date:
        raise ValueError("Updated stock inputs contain prediction-day or later observations")
    latest = market.loc[market["stock_id"].isin(target_ids)].groupby("stock_id")["date"].max()
    if (len(latest) != 50 or not latest.eq(prior_date).all()):
        missing = sorted(set(target_ids) - set(latest.index[latest.eq(prior_date)]))
        raise ValueError(f"Target stocks lack the declared previous session {previous_session}: {missing}")
    global_data = load_global(update_dir)
    if global_data["date"].max() >= prediction_date:
        raise ValueError("Updated global inputs contain prediction-day or later observations")
    manifest_path = DEFAULT_ARTIFACT / "manifest.json"
    if _sha256(manifest_path) != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("E3 source model artifact differs from the frozen reference")
    if _sha256(DEFAULT_SUMMARY) != EXPECTED_RULE_SHA256:
        raise ValueError("E3 TAIEX threshold rule differs from the frozen reference")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("train_end") != "2025-12-31"
            or manifest.get("config_id") != "xgboost_initial_v1"
            or manifest.get("seed") != 42):
        raise ValueError("Unexpected E3 model training recipe")
    return dates[0]


def export_phase2(template: Path = DEFAULT_TEMPLATE, update_dir: Path = DEFAULT_UPDATE,
                  previous_session: str | None = None,
                  output: Path | None = None) -> Path:
    day = validate_inputs(template, update_dir, previous_session)
    if output is None:
        output = ROOT / "submissions" / f"zhenci_e3_taiex_{day}.csv"
    if output.resolve() == template.resolve():
        raise ValueError("Submission output must not overwrite the template")
    with tempfile.TemporaryDirectory(prefix="e3_phase2_") as temporary:
        temporary = Path(temporary)
        f0_path = temporary / "f0.csv"
        candidate = temporary / "submission.csv"
        generate_updated_f0(BASE_DATA, update_dir, template, f0_path)
        export_predictions(DEFAULT_ARTIFACT, DEFAULT_SUMMARY, f0_path,
                           template, candidate, BASE_DATA)
        content = candidate.read_bytes()
        if output.exists():
            if output.read_bytes() != content:
                raise FileExistsError(f"Existing submission differs; choose a new --output: {output}")
            state = "Existing CSV verified"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as file:
                file.write(content)
            state = "CSV created"
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    up = sum(row[1] == "1" for row in rows[1:])
    print(f"{state}: {output}; date {day}; 50 predictions; {up} up, {50-up} down; "
          f"SHA256 {_sha256(output)}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--update-dir", type=Path, default=DEFAULT_UPDATE)
    parser.add_argument("--previous-session", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    export_phase2(args.template, args.update_dir, args.previous_session, args.output)
