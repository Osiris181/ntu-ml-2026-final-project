"""Build E12 20-session windows from frozen, training-fitted F0 preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_loader import KEYS, validate_keys
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES

WINDOW = 20


def complete_training_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only labeled dates with 19 earlier same-stock snapshots."""
    validate_keys(rows)
    parts = []
    for _, group in rows.sort_values(KEYS).groupby("stock_id", sort=True):
        if not group["date"].is_monotonic_increasing:
            raise ValueError("Stock sessions must be ordered")
        parts.append(group.iloc[WINDOW - 1:][KEYS])
    if not parts:
        raise ValueError("No training stocks")
    result = pd.concat(parts, ignore_index=True)
    if result.empty:
        raise ValueError("No complete training sequences")
    return result


def sequence_batch(f0: pd.DataFrame, keys: pd.DataFrame,
                   preprocessor: BaselinePreprocessor) -> tuple[dict, list[str]]:
    """Return windows ending on requested keys, in their original order.

    For each requested date, only that stock's preceding 19 sessions and the
    requested snapshot enter its window. Neither labels nor later snapshots
    enter the tensors.
    """
    validate_keys(f0)
    validate_keys(keys)
    if keys.empty:
        raise ValueError("No requested sequence keys")
    if not pd.api.types.is_datetime64_any_dtype(f0["date"]):
        raise TypeError("F0 dates must be datetime")
    if not pd.api.types.is_datetime64_any_dtype(keys["date"]):
        raise TypeError("Requested dates must be datetime")
    source = f0.loc[f0["stock_id"].isin(keys["stock_id"])
                    & f0["date"].le(keys["date"].max())].sort_values(KEYS).reset_index(drop=True)
    if source.empty:
        raise ValueError("No F0 history for requested stocks")
    transformed = preprocessor.transform(source[list(F0_FEATURES)])
    context_names = list(transformed.columns[len(NUMERIC_FEATURES):])
    if (list(transformed.columns[:len(NUMERIC_FEATURES)]) != list(NUMERIC_FEATURES)
            or not context_names
            or not all(name.startswith(("stock_id=", "day_of_week="))
                       for name in context_names)):
        raise ValueError("Frozen E12 feature order or categorical context changed")
    numeric = transformed[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float32)
    context = transformed[context_names].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all() or not np.isfinite(context).all():
        raise ValueError("Nonfinite transformed E12 inputs")
    sequences = np.empty((len(keys), WINDOW, len(NUMERIC_FEATURES)), dtype=np.float32)
    selected_context = np.empty((len(keys), len(context_names)), dtype=np.float32)
    requested = keys[KEYS].reset_index(drop=True)
    for stock_id, group in requested.groupby("stock_id", sort=False):
        history = source.loc[source["stock_id"].eq(stock_id)]
        history_dates = history["date"].to_numpy(dtype="datetime64[ns]")
        target_dates = group["date"].to_numpy(dtype="datetime64[ns]")
        ends = np.searchsorted(history_dates, target_dates)
        if (np.any(ends >= len(history_dates))
                or not np.array_equal(history_dates[ends], target_dates)):
            raise ValueError(f"Missing F0 snapshot for {stock_id}")
        if np.any(ends < WINDOW - 1):
            raise ValueError(f"Fewer than {WINDOW} stock sessions for {stock_id}")
        positions = history.index.to_numpy()
        window_positions = positions[ends[:, None] - np.arange(WINDOW - 1, -1, -1)]
        sequence = numeric[window_positions]
        if sequence.shape != (len(group), WINDOW, len(NUMERIC_FEATURES)):
            raise ValueError("Invalid E12 window dimensions")
        sequences[group.index.to_numpy()] = sequence
        selected_context[group.index.to_numpy()] = context[positions[ends]]
    return {"keys": requested, "sequences": sequences,
            "context": selected_context}, context_names
