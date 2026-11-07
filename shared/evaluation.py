"""Official score, target-key integrity, and equal-weight CV selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import KEYS, validate_keys
from .splits import E1_SEEDS

OPEN = "etl:adj_open"
CLOSE = "etl:adj_close"


def aligned_predictions(evaluate: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    validate_keys(evaluate)
    validate_keys(predictions)
    if not {OPEN, CLOSE}.issubset(evaluate.columns):
        raise ValueError("Evaluation requires adjusted open and close")
    if "predicted_y" not in predictions:
        raise ValueError("Missing predicted_y")
    values = predictions["predicted_y"]
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("Predictions must be binary 0/1 without missing values")
    if set(map(tuple, evaluate[KEYS].to_numpy())) != set(map(tuple, predictions[KEYS].to_numpy())):
        raise ValueError("Prediction keys must exactly match evaluation keys")
    merged = evaluate[KEYS + [OPEN, CLOSE, *(["y"] if "y" in evaluate else [])]].merge(
        predictions[KEYS + ["predicted_y"]], on=KEYS, validate="one_to_one", sort=False
    )
    if merged[[OPEN, CLOSE]].isna().any().any() or not np.isfinite(merged[[OPEN, CLOSE]].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite adjusted prices")
    if (merged[OPEN] <= 0).any():
        raise ValueError("Adjusted open must be positive")
    if "y" in merged and (merged["y"].isna().any() or not merged["y"].isin([0, 1]).all()):
        raise ValueError("Evaluation labels must be present and binary")
    return merged.sort_values(["date", "stock_id"]).reset_index(drop=True)


def evaluate_predictions(evaluate: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    aligned = aligned_predictions(evaluate, predictions)
    realized = (aligned[CLOSE] - aligned[OPEN]) / aligned[OPEN]
    score = ((2 * aligned["predicted_y"] - 1) * realized).mean()
    truth = (aligned[CLOSE] > aligned[OPEN]).astype(int)
    return {
        "official_score": float(score),
        "accuracy": float((aligned["predicted_y"] == truth).mean()),
        "n_eval": len(aligned),
    }


def cv_score(scores: dict[str, float]) -> float:
    if set(scores) != {"2023", "2024", "2025"}:
        raise ValueError("CV requires one official score for each of 2023, 2024, 2025")
    values = np.array([scores[year] for year in ("2023", "2024", "2025")], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CV scores must be finite")
    return float(values.mean())


def e1_seed_summary(scores_by_seed: dict[int, dict[str, float]]) -> dict:
    """Summarize five seed-level, equal-weight three-fold E1 CV scores."""
    if set(scores_by_seed) != set(E1_SEEDS):
        raise ValueError(f"E1 requires exactly these seeds: {E1_SEEDS}")
    per_seed = {seed: cv_score(scores_by_seed[seed]) for seed in E1_SEEDS}
    values = np.array(list(per_seed.values()), dtype=float)
    return {
        "per_seed_cv_official_score": per_seed,
        "mean_cv_official_score": float(values.mean()),
        "std_cv_official_score": float(values.std(ddof=0)),
        "min_cv_official_score": float(values.min()),
        "max_cv_official_score": float(values.max()),
    }


METRIC_FIELDS = (
    "member", "model", "experiment", "config_id", "fold", "seed",
    "official_score", "accuracy", "n_eval", "train_seconds", "predict_seconds",
)
PREDICTION_FIELDS = (
    "member", "model", "experiment", "config_id", "fold", "seed", "stock_id", "date", "predicted_y",
)
