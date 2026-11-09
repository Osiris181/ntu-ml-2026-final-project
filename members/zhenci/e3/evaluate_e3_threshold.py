"""Evaluate XGBoost decision thresholds from saved chronological CV predictions.

Run from the repository root:
    uv run python -m members.zhenci.e3.evaluate_e3_threshold

This does not train a model. The walk-forward check chooses each year's
threshold from strictly earlier validation years. The final threshold is
selected using all three CV years, never the holdout or submission feedback.
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
import pandas as pd

from shared.data_loader import KEYS, load_market, load_metadata, validate_keys
from shared.evaluation import e1_seed_summary, evaluate_predictions
from shared.splits import E1_SEEDS, FOLDS, split_market

THRESHOLDS = tuple(round(index * 0.025, 3) for index in range(14, 27))
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"


def choose_threshold(scores: dict[float, dict[int, dict[str, float]]], years: tuple[str, ...]) -> float:
    """Average each year's official score equally across five model seeds."""
    if not years or not set(years).issubset({fold.name for fold in FOLDS}):
        raise ValueError("Threshold training years must be completed CV years")
    means = {}
    for threshold in THRESHOLDS:
        means[threshold] = float(np.mean([
            np.mean([scores[threshold][seed][year] for year in years]) for seed in E1_SEEDS
        ]))
    if not np.isfinite(list(means.values())).all():
        raise ValueError("Nonfinite threshold score")
    # Prefer the original 0.5 cutoff on an exact tie.
    return min(THRESHOLDS, key=lambda value: (-means[value], abs(value - 0.5), value))


def run(data_dir: Path, predictions_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    frame = pd.read_csv(predictions_path, dtype={"stock_id": str}, parse_dates=["date"])
    required = {"stock_id", "date", "fold", "seed", "config_id", "p_up"}
    if required - set(frame.columns):
        raise ValueError(f"Validation predictions lack: {sorted(required - set(frame.columns))}")
    if frame["config_id"].nunique() != 1 or frame["config_id"].isna().any():
        raise ValueError("Supply predictions for exactly one model configuration")
    if frame["p_up"].isna().any() or not np.isfinite(frame["p_up"]).all() or not frame["p_up"].between(0, 1).all():
        raise ValueError("Invalid validation probabilities")
    if set(frame["seed"]) != set(E1_SEEDS) or set(frame["fold"].astype(str)) != {fold.name for fold in FOLDS}:
        raise ValueError("Predictions must contain all five seeds and three folds")
    if frame.duplicated(["fold", "seed", *KEYS]).any():
        raise ValueError("Duplicate seed/fold prediction keys")

    scores = {threshold: {seed: {} for seed in E1_SEEDS} for threshold in THRESHOLDS}
    for fold in FOLDS:
        _, valid_rows = split_market(market, target_ids, fold)
        for seed in E1_SEEDS:
            subset = frame.loc[frame["fold"].astype(str).eq(fold.name) & frame["seed"].eq(seed)]
            validate_keys(subset)
            for threshold in THRESHOLDS:
                predictions = subset.loc[:, KEYS].copy()
                predictions["predicted_y"] = (subset["p_up"].to_numpy() > threshold).astype(int)
                scores[threshold][seed][fold.name] = evaluate_predictions(valid_rows, predictions)["official_score"]

    baseline = e1_seed_summary(scores[0.5])
    selected = choose_threshold(scores, tuple(fold.name for fold in FOLDS))
    selected_cv = e1_seed_summary(scores[selected])
    forward_thresholds = {
        "2023": 0.5,
        "2024": choose_threshold(scores, ("2023",)),
        "2025": choose_threshold(scores, ("2023", "2024")),
    }
    forward_scores = {seed: {year: scores[forward_thresholds[year]][seed][year]
                             for year in forward_thresholds} for seed in E1_SEEDS}
    forward = e1_seed_summary(forward_scores)
    table = pd.DataFrame([
        {"threshold": threshold,
         "mean_cv_official_score": e1_seed_summary(scores[threshold])["mean_cv_official_score"],
         "std_cv_official_score": e1_seed_summary(scores[threshold])["std_cv_official_score"]}
        for threshold in THRESHOLDS
    ])
    summary = {
        "experiment_label": "E3 threshold study", "model": "XGBoost",
        "source_predictions": str(predictions_path), "config_id": frame["config_id"].iloc[0],
        "threshold_grid": list(THRESHOLDS), "seeds": list(E1_SEEDS),
        "baseline_threshold": 0.5, "baseline_cv": baseline,
        "selected_threshold_for_future_inference": selected,
        "selected_same_period_cv": selected_cv,
        "walk_forward_thresholds": forward_thresholds,
        "walk_forward_cv": forward,
        "selection_note": "Same-period selected CV is tuned on these folds; walk-forward thresholds use only prior validation years. Neither is a 2026 result.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(output_dir / "threshold_scores.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Baseline 0.5 CV: {baseline['mean_cv_official_score']:.8f}")
    print(f"Selected {selected:.3f} same-period CV: {selected_cv['mean_cv_official_score']:.8f}")
    print(f"Walk-forward CV: {forward['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/zhenci_xgboost_e3_threshold_v1")
    args = parser.parse_args()
    run(args.data_dir, args.predictions, args.output_dir)
