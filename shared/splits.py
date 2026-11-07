"""Locked expanding-window splits; dates are trading dates present in the input."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_loader import KEYS, validate_keys

SEED = 42


@dataclass(frozen=True)
class Split:
    name: str
    train_end: str
    eval_start: str
    eval_end: str


FOLDS = (
    Split("2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    Split("2024", "2023-12-31", "2024-01-01", "2024-12-31"),
    Split("2025", "2024-12-31", "2025-01-01", "2025-12-31"),
)
HOLDOUT = Split("holdout_2026_jan_may", "2025-12-31", "2026-01-01", "2026-05-31")


def split_market(market: pd.DataFrame, target_ids: list[str], split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_keys(market)
    if not pd.api.types.is_datetime64_any_dtype(market["date"]):
        raise TypeError("date must be parsed as datetime")
    if len(target_ids) != 50 or len(set(target_ids)) != 50:
        raise ValueError("Expected exactly 50 distinct target stock IDs")
    if pd.Timestamp(split.train_end) >= pd.Timestamp(split.eval_start):
        raise ValueError("Training must end before evaluation starts")
    train = market.loc[market["date"].between("2020-01-01", split.train_end)]
    evaluate = market.loc[
        market["date"].between(split.eval_start, split.eval_end)
        & market["stock_id"].isin(target_ids)
    ]
    if train.empty or evaluate.empty:
        raise ValueError(f"No training or target evaluation rows for {split.name}")
    if set(target_ids) - set(evaluate["stock_id"]):
        raise ValueError(f"Missing target stocks in {split.name}")
    if train["date"].max() >= evaluate["date"].min():
        raise ValueError("Temporal inversion")
    return tuple(x.sort_values(KEYS[::-1]).reset_index(drop=True) for x in (train, evaluate))


def evaluation_keys(evaluate: pd.DataFrame) -> pd.DataFrame:
    validate_keys(evaluate)
    return evaluate[KEYS].sort_values(["date", "stock_id"]).reset_index(drop=True)
