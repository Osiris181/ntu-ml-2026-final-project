"""Explicit user-requested exploratory E12-ranking Phase 1 export.

The historical E14 selection remains `none`. This exporter does not retrain,
reselect a model from Phase 1 data, or upload a submission.

Run from final_project:
    uv run --extra e12 python -m members.zhenci.xgboost.e14.export_e14_exploratory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from members.zhenci.xgboost.e14.rank_e14 import (
    DEFAULT_OUTPUT, ROOT as PROJECT_ROOT, TEMPLATE, UPDATED_F0,
    phase1_probabilities, rank_daily_25_25, source_hashes,
)
from shared.submission import build_submission, write_submission

DEFAULT_SUBMISSION = PROJECT_ROOT / "submissions/zhenci_e14_e12_rank_exploratory_phase1.csv"
CANDIDATE = "e12_rank"


def export(selection_dir: Path, f0_path: Path, template_path: Path,
           data_dir: Path, output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Submission already exists: {output_path}")
    with (selection_dir / "selection.json").open(encoding="utf-8") as file:
        selection = json.load(file)
    if selection.get("input_sha256") != source_hashes():
        raise ValueError("Historical E14 inputs or code differ from the saved comparison")
    if selection.get("selected_candidate") is not None:
        raise ValueError("This exploratory override expects the archived E14 no-candidate result")
    scores = selection.get("equal_year_scores", {})
    if not (scores.get("e12_rank") is not None
            and scores.get("e3_rank") is not None
            and scores["e12_rank"] > scores["e3_rank"]):
        raise ValueError("E12 was not the stronger historical ranking variant")
    probabilities, ids, dates = phase1_probabilities(CANDIDATE, f0_path,
                                                     template_path, data_dir)
    predictions = rank_daily_25_25(probabilities, "p_up")
    counts = predictions.groupby("date")["predicted_y"].agg(["size", "sum"])
    if (len(predictions) != 4200 or len(counts) != 84
            or not counts["size"].eq(50).all()
            or not counts["sum"].eq(25).all()):
        raise ValueError("Expected 84 dates with exactly 25 up and 25 down predictions")
    submission = build_submission(predictions, template_path, ids)
    if (submission.columns.tolist() != ["id", *dates]
            or submission["id"].tolist() != ids
            or not submission[dates].sum(axis=0).eq(25).all()):
        raise ValueError("Official Phase 1 template layout or 25/25 balance changed")
    saved = write_submission(predictions, template_path, output_path, ids)
    restored = pd.read_csv(saved, dtype={"id": str})
    if not restored.equals(submission):
        raise ValueError("Saved submission does not match the validated CSV")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--f0-path", type=Path, default=UPDATED_F0)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION)
    args = parser.parse_args()
    print(export(args.selection_dir, args.f0_path, args.template,
                 args.data_dir, args.output))
