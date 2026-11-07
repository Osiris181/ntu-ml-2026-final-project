"""Load the supplied files without changing their source values."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

KEYS = ["stock_id", "date"]
GLOBAL_COLUMNS = ["^VIX", "^IXIC", "^GSPC", "^TWII", "^KS11", "TX_OI"]
MARKET_COLUMNS = [
    "etl:adj_close", "etl:adj_open", "etl:adj_high", "etl:adj_low",
    "price:成交筆數", "price:成交金額", "price:成交股數",
    "foreign_investors_shareholding:全體外資及陸資持股比率",
    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
    "institutional_investors_trading_summary:投信買賣超股數",
    "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)",
    "institutional_investors_trading_summary:自營商買賣超股數(避險)",
    "RSI", "SMA", "MACD", "ATR", "BBANDS_upper", "BBANDS_mid", "BBANDS_lower", "y",
]


def _dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("Missing dates are invalid row keys")
    return dates


def validate_keys(frame: pd.DataFrame) -> None:
    if not set(KEYS).issubset(frame.columns):
        raise ValueError(f"Required row keys: {KEYS}")
    if frame[KEYS].isna().any().any() or frame["stock_id"].astype(str).eq("").any():
        raise ValueError("Missing stock ID or date")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate stock/date keys")


def load_metadata(data_dir: str | Path) -> tuple[list[str], list[str]]:
    with (Path(data_dir) / "info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    target, universe = info["Top50@Y"], info["Top100@Y"]
    if (len(target), len(set(target)), len(universe), len(set(universe))) != (50, 50, 100, 100):
        raise ValueError("Expected 50 distinct target IDs and 100 distinct universe IDs")
    if not set(target).issubset(universe) or not all(isinstance(x, str) for x in universe):
        raise ValueError("Invalid stock universe metadata")
    return target, universe


def load_market(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir)
    _, universe = load_metadata(root)
    frames = []
    for stock_id in universe:
        path = root / "market" / f"{stock_id}.csv"
        frame = pd.read_csv(path)
        expected = ["date", *MARKET_COLUMNS]
        if frame.columns.tolist() != expected:
            raise ValueError(f"Unexpected market schema: {path}")
        frame["date"] = _dates(frame["date"])
        frame.insert(0, "stock_id", stock_id)
        frames.append(frame)
    market = pd.concat(frames, ignore_index=True)
    validate_keys(market)
    if not market["y"].isin([0, 1]).all():
        raise ValueError("Historical labels must be binary and present")
    return market.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_global(data_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(data_dir) / "global" / "global_data.csv")
    if frame.columns.tolist() != ["date", *GLOBAL_COLUMNS]:
        raise ValueError("Unexpected global data schema")
    frame["date"] = _dates(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate global dates")
    return frame.sort_values("date").reset_index(drop=True)
