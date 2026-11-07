"""Shared F0 construction for every stock/date row.

Stock fields use completed per-stock trading sessions. Global fields use the
latest nonmissing observation dated before the prediction date, separately for
each series, with a five-calendar-day maximum age. These are team-approved
conservative rules, not externally verified publication timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import KEYS, load_global, load_market, validate_keys
from .preprocessing import F0_FEATURES

GLOBAL_MAX_AGE_DAYS = 5
GLOBAL_SERIES = {
    "vix_normalized": "^VIX",
    "nasdaq_return": "^IXIC",
    "sp500_return": "^GSPC",
    "taiex_return": "^TWII",
}
COL = {
    "close": "etl:adj_close", "open": "etl:adj_open",
    "high": "etl:adj_high", "low": "etl:adj_low",
    "trades": "price:成交筆數", "amount": "price:成交金額",
    "volume": "price:成交股數",
    "ownership": "foreign_investors_shareholding:全體外資及陸資持股比率",
    "foreign": "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
    "trust": "institutional_investors_trading_summary:投信買賣超股數",
}
TECHNICAL_COLUMNS = ("RSI", "SMA", "MACD", "ATR", "BBANDS_upper", "BBANDS_mid", "BBANDS_lower")
DATASET_COLUMNS = ("stock_id", "date", *(field for field in F0_FEATURES if field != "stock_id"))


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _require_dates(frame: pd.DataFrame, name: str) -> None:
    if "date" not in frame or not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise TypeError(f"{name} date must be parsed as datetime")
    if frame["date"].isna().any():
        raise ValueError(f"{name} has missing dates")


def stock_history_candidates(market: pd.DataFrame) -> pd.DataFrame:
    """Calculate 15 lagged stock features and weekday without dropping rows.

    The name is retained for existing callers. Values are shared F0 inputs
    under the agreed lag policy; source dates are included for inspection.
    """
    validate_keys(market)
    _require_dates(market, "Market")
    absent = set((*COL.values(), *TECHNICAL_COLUMNS)) - set(market.columns)
    if absent:
        raise ValueError(f"Missing stock source columns: {sorted(absent)}")
    work = market.sort_values(["stock_id", "date"]).copy()
    group = work.groupby("stock_id", sort=False)

    def lag(column: str, sessions: int = 1) -> pd.Series:
        return group[column].shift(sessions)

    def mean20(column: str) -> pd.Series:
        return group[column].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).mean()
        )

    close1, open1, high1, low1 = (lag(COL[field]) for field in ("close", "open", "high", "low"))
    volume1, amount1 = lag(COL["volume"]), lag(COL["amount"])
    volume2 = lag(COL["volume"], 2)
    work["_trade_size"] = _divide(work[COL["volume"]], work[COL["trades"]])
    trade_size = work.groupby("stock_id", sort=False)["_trade_size"]
    size1 = trade_size.shift(1)
    size_mean20 = trade_size.transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )

    out = work[KEYS].copy()
    out["source_date"] = lag("date")
    out["institutional_source_date"] = lag("date", 2)
    out["intraday_return_1d"] = _divide(close1, open1) - 1
    out["daily_return_1d"] = _divide(close1, lag(COL["close"], 2)) - 1
    out["momentum_5d"] = _divide(close1, lag(COL["close"], 6)) - 1
    out["price_range_1d"] = _divide(high1 - low1, close1)
    out["relative_volume_20d"] = _divide(volume1, mean20(COL["volume"]))
    out["relative_amount_20d"] = _divide(amount1, mean20(COL["amount"]))
    out["trade_size_ratio_20d"] = _divide(size1, size_mean20)
    out["foreign_ownership"] = lag(COL["ownership"], 2) / 100
    out["foreign_net_buy_ratio"] = _divide(lag(COL["foreign"], 2), volume2)
    out["trust_net_buy_ratio"] = _divide(lag(COL["trust"], 2), volume2)
    out["rsi_normalized"] = lag("RSI") / 100
    out["sma_distance"] = _divide(close1, lag("SMA")) - 1
    out["macd_normalized"] = _divide(lag("MACD"), close1)
    out["atr_normalized"] = _divide(lag("ATR"), close1)
    out["bb_width"] = _divide(lag("BBANDS_upper") - lag("BBANDS_lower"), lag("BBANDS_mid"))
    out["day_of_week"] = out["date"].dt.day_name()
    return out.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _global_features(prediction_dates: pd.Series, global_data: pd.DataFrame) -> pd.DataFrame:
    _require_dates(global_data, "Global")
    required = set(GLOBAL_SERIES.values())
    if required - set(global_data.columns):
        raise ValueError(f"Missing global series: {sorted(required - set(global_data.columns))}")
    if global_data["date"].duplicated().any():
        raise ValueError("Duplicate global dates")
    dates = pd.DataFrame({"date": sorted(prediction_dates.unique())})
    source = global_data.sort_values("date")
    for feature, series in GLOBAL_SERIES.items():
        values = pd.to_numeric(source[series], errors="raise").replace([np.inf, -np.inf], np.nan)
        observations = pd.DataFrame({"source_date": source["date"], "value": values}).dropna(
            subset=["value"]
        )
        observations["feature_value"] = (
            observations["value"] / 100
            if feature == "vix_normalized"
            else _divide(observations["value"], observations["value"].shift(1)) - 1
        )
        aligned = pd.merge_asof(
            dates[["date"]],
            observations[["source_date", "feature_value"]],
            left_on="date",
            right_on="source_date",
            direction="backward",
            allow_exact_matches=False,
            tolerance=pd.Timedelta(days=GLOBAL_MAX_AGE_DAYS),
        )
        dates[feature] = aligned["feature_value"]
    return dates


def build_f0_dataset(market: pd.DataFrame, global_data: pd.DataFrame) -> pd.DataFrame:
    """Return one shared, unscaled F0 row for every input stock/date key.

    The output has only date plus the 21 logical F0 fields. In particular,
    target-day y, OHLC and realized returns are never included. Missing
    numerical values remain NaN for training-fold-only baseline imputation.
    """
    stock = stock_history_candidates(market)
    global_features = _global_features(stock["date"], global_data)
    merged = stock.merge(global_features, on="date", how="left", validate="many_to_one")
    output = merged.loc[:, DATASET_COLUMNS].replace([np.inf, -np.inf], np.nan)
    validate_keys(output)
    if len(output) != len(market) or set(map(tuple, output[KEYS].to_numpy())) != set(
        map(tuple, market[KEYS].to_numpy())
    ):
        raise ValueError("F0 construction changed stock/date row keys")
    return output.sort_values(["date", "stock_id"]).reset_index(drop=True)


def generate_f0_dataset(data_dir: str | Path, output_path: str | Path) -> Path:
    """Build the reusable CSV from unchanged raw files."""
    f0 = build_f0_dataset(load_market(data_dir), load_global(data_dir))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    f0.to_csv(output, index=False)
    return output


def load_f0_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a generated F0 CSV for a member's E1 run."""
    f0 = pd.read_csv(path, dtype={"stock_id": str}, parse_dates=["date"])
    if f0.columns.tolist() != list(DATASET_COLUMNS):
        raise ValueError("Unexpected F0 dataset columns")
    validate_keys(f0)
    _require_dates(f0, "F0")
    return f0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the shared F0 feature CSV")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/processed/f0.csv")
    args = parser.parse_args()
    path = generate_f0_dataset(args.data_dir, args.output)
    print(path)
