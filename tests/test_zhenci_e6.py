"""Check that E6's inner dates cannot reach their outer validation years."""

import pandas as pd

from members.zhenci.e6.train_e6 import checked_inner_split
from shared.splits import FOLDS, split_market


def test_inner_validation_is_strictly_inside_outer_training_history() -> None:
    stock_ids = [f"S{index:03d}" for index in range(100)]
    target_ids = stock_ids[:50]
    years = range(2020, 2026)
    market = pd.DataFrame(
        [(stock_id, pd.Timestamp(f"{year}-06-01")) for year in years for stock_id in stock_ids],
        columns=["stock_id", "date"],
    )
    for outer in FOLDS:
        inner = checked_inner_split(outer)
        inner_train, inner_valid = split_market(market, target_ids, inner)
        outer_train, outer_valid = split_market(market, target_ids, outer)
        assert inner_train.date.max() < inner_valid.date.min()
        assert inner_valid.date.max() < outer_valid.date.min()
        assert set(map(tuple, inner_valid[["stock_id", "date"]].to_numpy())).issubset(
            set(map(tuple, outer_train[["stock_id", "date"]].to_numpy()))
        )
