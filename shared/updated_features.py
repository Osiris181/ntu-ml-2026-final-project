"""Build shared F0 from the original history and a later, separate data update.

Updated labels are deliberately ignored. The same frozen F0 formulas in
shared.features are used for historical and prediction-date rows.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import KEYS, MARKET_COLUMNS, load_global, load_market, load_metadata, validate_keys
from .features import build_f0_dataset
from .submission import load_submission_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data"
DEFAULT_UPDATE = PROJECT_ROOT / "updated_data_7_Nov/data"
DEFAULT_TEMPLATE = PROJECT_ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/f0_with_updates.csv"


def load_updated_market(update_dir: str | Path, stock_ids: list[str]) -> pd.DataFrame:
    """Read the later stock files as feature sources, never as training labels."""
    market_dir = Path(update_dir) / "market"
    actual = {path.stem for path in market_dir.glob("*.csv")}
    expected = set(stock_ids)
    if actual != expected:
        raise ValueError(
            f"Updated market files differ from the stock universe: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    frames = []
    for stock_id in stock_ids:
        path = market_dir / f"{stock_id}.csv"
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["date", *MARKET_COLUMNS]:
            raise ValueError(f"Unexpected updated market schema: {path}")
        if frame.empty:
            raise ValueError(f"Updated market file has no rows: {path}")
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        if frame["date"].isna().any():
            raise ValueError(f"Updated market file has missing dates: {path}")
        for column in MARKET_COLUMNS[:-1]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame.insert(0, "stock_id", stock_id)
        frames.append(frame.drop(columns="y"))
    market = pd.concat(frames, ignore_index=True)
    validate_keys(market)
    return market.sort_values(["date", "stock_id"]).reset_index(drop=True)


def combine_feature_history(
    original_market: pd.DataFrame,
    updated_market: pd.DataFrame,
    original_global: pd.DataFrame,
    updated_global: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append strictly later observations; never replace original source rows."""
    validate_keys(original_market)
    validate_keys(updated_market)
    if original_market.empty or updated_market.empty or original_global.empty or updated_global.empty:
        raise ValueError("Original and updated market/global sources must be nonempty")
    if updated_market["date"].min() <= original_market["date"].max():
        raise ValueError("Updated market dates must start after the original market endpoint")
    if updated_global["date"].min() <= original_global["date"].max():
        raise ValueError("Updated global dates must start after the original global endpoint")
    if original_global["date"].duplicated().any() or updated_global["date"].duplicated().any():
        raise ValueError("Duplicate global dates")
    market = pd.concat([original_market.drop(columns="y", errors="ignore"), updated_market], ignore_index=True)
    global_data = pd.concat([original_global, updated_global], ignore_index=True)
    validate_keys(market)
    return market, global_data


def add_template_keys(market: pd.DataFrame, template_path: str | Path, target_ids: list[str]) -> pd.DataFrame:
    """Add one future prediction-date key per stock without target-day values.

    An absent date inside observed history signals a data gap. Multiple future
    dates for one stock need intervening actual sessions before F0 can be built.
    """
    ids, dates = load_submission_template(template_path, target_ids)
    required = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS).to_frame(index=False)
    present = pd.MultiIndex.from_frame(market[KEYS])
    missing = required.loc[~pd.MultiIndex.from_frame(required).isin(present)].copy()
    if missing.empty:
        return market
    latest = market.groupby("stock_id")["date"].max()
    if (missing["date"] <= missing["stock_id"].map(latest)).any():
        raise ValueError("A template key is absent inside observed stock history")
    if missing["stock_id"].duplicated().any():
        raise ValueError("Multiple future dates for one stock require intervening actual market history")
    placeholders = pd.DataFrame({column: np.nan for column in market.columns}, index=missing.index)
    placeholders["stock_id"] = missing["stock_id"]
    placeholders["date"] = missing["date"]
    extended = pd.concat([market, placeholders], ignore_index=True)
    validate_keys(extended)
    return extended


def generate_updated_f0(
    base_dir: str | Path = DEFAULT_BASE,
    update_dir: str | Path = DEFAULT_UPDATE,
    template_path: str | Path = DEFAULT_TEMPLATE,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Build a separate F0 CSV that includes the validated later history."""
    target_ids, stock_ids = load_metadata(base_dir)
    original_market = load_market(base_dir)
    updated_market = load_updated_market(update_dir, stock_ids)
    market, global_data = combine_feature_history(
        original_market,
        updated_market,
        load_global(base_dir),
        load_global(update_dir),
    )
    market = add_template_keys(market, template_path, target_ids)
    f0 = build_f0_dataset(market, global_data)
    ids, dates = load_submission_template(template_path, target_ids)
    required = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS)
    if not required.isin(pd.MultiIndex.from_frame(f0[KEYS])).all():
        raise ValueError("Updated F0 still lacks template stock/date rows")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        f0.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"F0 rows: {len(f0)}; template keys: {len(required)}; output: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--update-dir", type=Path, default=DEFAULT_UPDATE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_updated_f0(args.base_dir, args.update_dir, args.template, args.output)
