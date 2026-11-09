"""Test fixed TAIEX-regime threshold rules on saved E1 CV probabilities.

Run from the repository root:
    uv run python -m members.zhenci.e3.evaluate_e3_market_regime

This study does not train XGBoost or use the independent holdout. The regime
comes from the shared F0 ``taiex_return`` field, which is aligned strictly
before each prediction date by the shared feature builder.
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
from shared.features import load_f0_dataset
from shared.splits import E1_SEEDS, FOLDS, split_market

DEFAULT_PREDICTIONS = PROJECT_ROOT / "results/zheng_xgboost_e1_five_seeds_v1/validation_predictions.csv"
DEFAULT_F0 = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/zhenci_xgboost_e3_market_regime_v2"

# Predeclared small comparison set. The first three rows measure whether a
# regime rule adds value beyond an ordinary, fixed threshold change.
# "Negative" means the last eligible TAIEX observation in F0 fell; it never
# refers to the target day's completed market return.
CANDIDATES = (
    {"config_id": "e3_threshold_0500_control", "negative": 0.5, "nonnegative": 0.5, "missing": 0.5},
    {"config_id": "e3_threshold_0475_static", "negative": 0.475, "nonnegative": 0.475, "missing": 0.475},
    {"config_id": "e3_threshold_0525_static", "negative": 0.525, "nonnegative": 0.525, "missing": 0.525},
    {"config_id": "e3_taiex_negative_0525", "negative": 0.525, "nonnegative": 0.5, "missing": 0.5},
    {"config_id": "e3_taiex_negative_0550", "negative": 0.55, "nonnegative": 0.5, "missing": 0.5},
    {"config_id": "e3_taiex_negative_0525_nonnegative_0475", "negative": 0.525, "nonnegative": 0.475, "missing": 0.5},
)
def regime_predictions(p_up: np.ndarray, taiex_return: np.ndarray, candidate: dict) -> np.ndarray:
    """Use only the saved model output and the already lagged F0 regime value."""
    probabilities = np.asarray(p_up, dtype=float)
    regime = np.asarray(taiex_return, dtype=float)
    if probabilities.ndim != 1 or regime.shape != probabilities.shape:
        raise ValueError("Probabilities and TAIEX values must be aligned one-dimensional arrays")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid saved probabilities")
    if np.isinf(regime).any():
        raise ValueError("F0 TAIEX return contains infinity")
    threshold = np.where(
        np.isnan(regime), candidate["missing"],
        np.where(regime < 0, candidate["negative"], candidate["nonnegative"]),
    )
    return (probabilities > threshold).astype(int)


def choose_candidate(scores: dict[str, dict[int, dict[str, float]]], years: tuple[str, ...]) -> str:
    """Rank candidates by equal-year, equal-seed official score on prior years."""
    if not years or not set(years).issubset({fold.name for fold in FOLDS}):
        raise ValueError("Selection years must be completed CV years")
    ranked = []
    for candidate in CANDIDATES:
        config_id = candidate["config_id"]
        values = [scores[config_id][seed][year] for seed in E1_SEEDS for year in years]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite candidate score")
        ranked.append(float(np.mean(values)))
    # Candidate order prefers the E1 control on an exact tie.
    return CANDIDATES[int(np.argmax(ranked))]["config_id"]


def run(data_dir: Path, f0_path: Path, predictions_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    target_ids, _ = load_metadata(data_dir)
    market = load_market(data_dir)
    f0 = load_f0_dataset(f0_path).loc[:, [*KEYS, "taiex_return"]]
    source = pd.read_csv(predictions_path, dtype={"stock_id": str}, parse_dates=["date"])
    required = {*KEYS, "fold", "seed", "config_id", "p_up"}
    if required - set(source.columns):
        raise ValueError(f"Saved predictions lack: {sorted(required - set(source.columns))}")
    source["fold"] = source["fold"].astype(str)
    if source["config_id"].isna().any() or source["config_id"].nunique() != 1:
        raise ValueError("Supply one E1 model configuration")
    if set(source["seed"]) != set(E1_SEEDS) or set(source["fold"]) != {fold.name for fold in FOLDS}:
        raise ValueError("Saved predictions must cover all five seeds and three folds")
    if source.duplicated(["fold", "seed", *KEYS]).any():
        raise ValueError("Duplicate saved prediction keys")

    scores = {candidate["config_id"]: {seed: {} for seed in E1_SEEDS} for candidate in CANDIDATES}
    metric_rows = []
    missing_regime_rows = {}
    for fold in FOLDS:
        _, valid_rows = split_market(market, target_ids, fold)
        regime_rows = valid_rows.loc[:, KEYS].merge(
            f0, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True
        )
        if not regime_rows["_merge"].eq("both").all():
            raise ValueError(f"Missing F0 rows for {fold.name}")
        missing_regime_rows[fold.name] = int(regime_rows["taiex_return"].isna().sum())
        for seed in E1_SEEDS:
            subset = source.loc[source["fold"].eq(fold.name) & source["seed"].eq(seed)]
            validate_keys(subset)
            aligned = regime_rows.loc[:, [*KEYS, "taiex_return"]].merge(
                subset.loc[:, [*KEYS, "p_up"]], on=KEYS, how="left", sort=False,
                validate="one_to_one", indicator=True,
            )
            if len(subset) != len(valid_rows) or not aligned["_merge"].eq("both").all():
                raise ValueError(f"Incomplete validation predictions for {fold.name}, seed {seed}")
            for candidate in CANDIDATES:
                predicted = aligned.loc[:, KEYS].copy()
                predicted["predicted_y"] = regime_predictions(
                    aligned["p_up"].to_numpy(), aligned["taiex_return"].to_numpy(), candidate
                )
                result = evaluate_predictions(valid_rows, predicted)
                config_id = candidate["config_id"]
                scores[config_id][seed][fold.name] = result["official_score"]
                metric_rows.append({"config_id": config_id, "fold": fold.name, "seed": seed, **result})

    comparisons = []
    for candidate in CANDIDATES:
        config_id = candidate["config_id"]
        comparisons.append({**candidate, "fold_official_scores_by_seed": scores[config_id],
                            **e1_seed_summary(scores[config_id])})
    selected = choose_candidate(scores, tuple(fold.name for fold in FOLDS))
    forward_choices = {
        "2023": CANDIDATES[0]["config_id"],
        "2024": choose_candidate(scores, ("2023",)),
        "2025": choose_candidate(scores, ("2023", "2024")),
    }
    forward_scores = {seed: {year: scores[forward_choices[year]][seed][year]
                             for year in forward_choices} for seed in E1_SEEDS}
    summary = {
        "experiment_label": "E3 TAIEX-regime threshold study",
        "source_model_configuration": source["config_id"].iloc[0],
        "source_predictions": str(predictions_path),
        "source_f0": str(f0_path),
        "regime_feature": "F0 taiex_return, strictly prior date, five-calendar-day maximum age",
        "missing_regime_rows_by_fold": missing_regime_rows,
        "seeds": list(E1_SEEDS),
        "comparisons": comparisons,
        "selected_same_period_config_id": selected,
        "selected_same_period_cv": e1_seed_summary(scores[selected]),
        "walk_forward_choices": forward_choices,
        "walk_forward_cv": e1_seed_summary(forward_scores),
        "selection_note": "Same-period CV is used for tuning; walk-forward choices use only prior validation years. Neither is the 2026 holdout or a submission result.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"E1 control: {comparisons[0]['mean_cv_official_score']:.8f}")
    print(f"Selected {selected}: {summary['selected_same_period_cv']['mean_cv_official_score']:.8f}")
    print(f"Walk-forward CV: {summary['walk_forward_cv']['mean_cv_official_score']:.8f}")
    print(f"Results written to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.data_dir, args.f0_path, args.predictions, args.output_dir)
