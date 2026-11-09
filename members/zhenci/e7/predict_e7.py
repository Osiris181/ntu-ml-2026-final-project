"""Export Phase 1 predictions with the frozen E1 model and CV-selected E7 rule.

Run from the repository root:
    uv run python -m members.zhenci.e7.predict_e7

The stock payoff cutoffs are fitted only on the original 2020-2025 training
returns. The model and preprocessing stay frozen. No submission is uploaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xgboost as xgb

from members.zhenci.e1.predict_e1 import require_template_features
from members.zhenci.e7.evaluate_e7 import apply_payoff_cutoffs, fit_payoff_cutoffs
from members.zhenci.e7.evaluate_holdout import (
    DEFAULT_ARTIFACT, DEFAULT_DATA_DIR, DEFAULT_E7_SUMMARY,
    DEFAULT_F0_PATH, SELECTED_ARM, load_locked_inputs,
)
from shared.data_loader import KEYS, load_market, load_metadata
from shared.features import load_f0_dataset
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor
from shared.splits import HOLDOUT
from shared.submission import load_submission_template, write_submission

DEFAULT_UPDATED_F0 = PROJECT_ROOT / "data/processed/f0_with_updates.csv"
DEFAULT_TEMPLATE = PROJECT_ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "submissions/zhenci_e7_stock_payoff_phase1.csv"


def export_predictions(
    data_dir: Path, original_f0_path: Path, updated_f0_path: Path,
    summary_path: Path, artifact_dir: Path, template_path: Path,
    output_path: Path,
) -> Path:
    """Apply the CV-frozen E7 policy to all keys in the official template."""
    if output_path.exists():
        raise FileExistsError(output_path)
    summary, manifest = load_locked_inputs(summary_path, artifact_dir, original_f0_path)
    target_ids, all_ids = load_metadata(data_dir)
    ids, dates = load_submission_template(template_path, target_ids)
    if len(ids) * len(dates) != 4200:
        raise ValueError("Expected the 4,200-key Phase 1 submission template")

    market = load_market(data_dir)
    original_f0 = load_f0_dataset(original_f0_path)
    if len(market) != len(original_f0) or not market.loc[:, KEYS].equals(original_f0.loc[:, KEYS]):
        raise ValueError("Original market and F0 row keys differ")
    train = market.loc[market["date"].between("2020-01-01", HOLDOUT.train_end)].copy()
    if len(train) != manifest["training_rows"] or set(train["stock_id"]) != set(all_ids):
        raise ValueError("Stock cutoff training rows differ from the frozen E1 training window")
    train_taiex = train.loc[:, KEYS].merge(
        original_f0.loc[:, [*KEYS, "taiex_return"]],
        on=KEYS, how="left", validate="one_to_one", sort=False, indicator=True,
    )
    if (not train_taiex["_merge"].eq("both").all()
            or not train_taiex.loc[:, KEYS].equals(train.loc[:, KEYS])):
        raise ValueError("Original F0 TAIEX rows are missing or reordered")
    fitted, _ = fit_payoff_cutoffs(train, train_taiex["taiex_return"], "phase1_2020_2025")

    updated_f0 = load_f0_dataset(updated_f0_path)
    rows = require_template_features(updated_f0, ids, dates)
    with (artifact_dir / "preprocessor.json").open("r", encoding="utf-8") as file:
        preprocessor = BaselinePreprocessor.from_state(json.load(file))
    transformed = preprocessor.transform(rows.loc[:, list(F0_FEATURES)])
    if transformed.columns.tolist() != manifest["transformed_feature_order"]:
        raise ValueError("Frozen transformed feature order changed")
    booster = xgb.Booster(model_file=str(artifact_dir / "model.json"))
    p_up = booster.predict(xgb.DMatrix(transformed))
    if len(p_up) != len(rows) or not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
        raise ValueError("Frozen model returned invalid probabilities")
    labels = apply_payoff_cutoffs(
        p_up, rows["stock_id"], rows["taiex_return"],
        fitted, SELECTED_ARM, summary["fixed_e3_rule"],
    )
    predictions = rows.loc[:, KEYS].copy()
    predictions["predicted_y"] = labels
    output = write_submission(predictions, template_path, output_path, target_ids)
    print(f"E7 predicted up on {int(labels.sum())} of {len(labels)} template keys")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--original-f0", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--updated-f0", type=Path, default=DEFAULT_UPDATED_F0)
    parser.add_argument("--summary", type=Path, default=DEFAULT_E7_SUMMARY)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(export_predictions(args.data_dir, args.original_f0, args.updated_f0,
                             args.summary, args.artifact_dir, args.template, args.output))
