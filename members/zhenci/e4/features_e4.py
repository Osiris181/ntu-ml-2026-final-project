"""Build member-owned F2 features from completed stock sessions.

F2 contains the shared 21-field F0 row plus five rolling intraday/overnight
history fields. Historical and updated builds use the same formulas. The
updated market files supply feature history only; their labels are ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from shared.data_loader import KEYS, load_global, load_market, load_metadata, validate_keys
from shared.features import COL, DATASET_COLUMNS, build_f0_dataset, load_f0_dataset
from shared.submission import load_submission_template
from shared.updated_features import add_template_keys, combine_feature_history, load_updated_market

F2_EXTRA_FEATURES = (
    "e4_intraday_mean_5",
    "e4_intraday_mean_20",
    "e4_overnight_mean_5",
    "e4_overnight_mean_20",
    "e4_overnight_minus_intraday_20",
)
F2_DATASET_COLUMNS = (*DATASET_COLUMNS, *F2_EXTRA_FEATURES)

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_UPDATE_DIR = PROJECT_ROOT / "updated_data_7_Nov/data"
DEFAULT_TEMPLATE = PROJECT_ROOT / "sample_submission.csv"
DEFAULT_F0_PATH = PROJECT_ROOT / "data/processed/f0.csv"
DEFAULT_HISTORICAL_OUTPUT = PROJECT_ROOT / "data/processed/f2.csv"
DEFAULT_UPDATED_OUTPUT = PROJECT_ROOT / "data/processed/f2_with_updates.csv"


def build_f2_dataset(market: pd.DataFrame, f0: pd.DataFrame) -> pd.DataFrame:
    """Add five lagged fields to exactly the supplied F0 stock/date keys.

For date t, rolling windows contain completed per-stock sessions t-n..t-1.
Undefined price ratios and incomplete windows remain missing until the
training-fold-only F2 preprocessor is fitted. No market label is copied.
"""
    validate_keys(market)
    validate_keys(f0)
    for name, frame in (("Market", market), ("F0", f0)):
        if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
            raise TypeError(f"{name} dates must be parsed as datetime")
    missing_f0 = set(DATASET_COLUMNS) - set(f0.columns)
    if missing_f0:
        raise ValueError(f"Missing F0 dataset columns: {sorted(missing_f0)}")
    missing_market = {COL["open"], COL["close"]} - set(market.columns)
    if missing_market:
        raise ValueError(f"Missing adjusted market prices: {sorted(missing_market)}")

    work = market.sort_values(KEYS).copy()
    groups = work.groupby("stock_id", sort=False)
    open_price = pd.to_numeric(work[COL["open"]], errors="raise")
    close_price = pd.to_numeric(work[COL["close"]], errors="raise")
    previous_close = groups[COL["close"]].shift(1)
    previous_close = pd.to_numeric(previous_close, errors="raise")
    work["_intraday"] = (close_price / open_price - 1).replace([np.inf, -np.inf], np.nan)
    work["_overnight"] = (open_price / previous_close - 1).replace([np.inf, -np.inf], np.nan)

    extra = work.loc[:, KEYS].copy()
    for window in (5, 20):
        extra[f"e4_intraday_mean_{window}"] = work.groupby("stock_id", sort=False)[
            "_intraday"
        ].transform(lambda values: values.shift(1).rolling(window, min_periods=window).mean())
        extra[f"e4_overnight_mean_{window}"] = work.groupby("stock_id", sort=False)[
            "_overnight"
        ].transform(lambda values: values.shift(1).rolling(window, min_periods=window).mean())
    extra["e4_overnight_minus_intraday_20"] = (
        extra["e4_overnight_mean_20"] - extra["e4_intraday_mean_20"]
    )
    extra.loc[:, list(F2_EXTRA_FEATURES)] = extra.loc[:, list(F2_EXTRA_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    )

    # Select the frozen F0 schema explicitly so labels or raw prices in a
    # caller's frame cannot enter the training feature matrix.
    combined = f0.loc[:, DATASET_COLUMNS].merge(
        extra, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True
    )
    if len(combined) != len(market) or not combined["_merge"].eq("both").all():
        raise ValueError("F2 requires exactly the same stock/date keys as market and F0")
    combined = combined.drop(columns="_merge").loc[:, F2_DATASET_COLUMNS]
    validate_keys(combined)
    return combined.sort_values(["date", "stock_id"]).reset_index(drop=True)


def load_f2_dataset(path: str | Path) -> pd.DataFrame:
    """Load generated F2 without accepting extra labels or reordered fields."""
    f2 = pd.read_csv(path, dtype={"stock_id": str}, parse_dates=["date"])
    if f2.columns.tolist() != list(F2_DATASET_COLUMNS):
        raise ValueError("Unexpected F2 dataset columns")
    validate_keys(f2)
    if not pd.api.types.is_datetime64_any_dtype(f2["date"]):
        raise TypeError("F2 dates must be parsed as datetime")
    return f2


def build_historical_and_updated_f2(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    update_dir: str | Path = DEFAULT_UPDATE_DIR,
    template_path: str | Path = DEFAULT_TEMPLATE,
    f0_path: str | Path = DEFAULT_F0_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare historical F2 and a separate version covering template keys."""
    target_ids, stock_ids = load_metadata(data_dir)
    original_market = load_market(data_dir)
    historical = build_f2_dataset(original_market, load_f0_dataset(f0_path))

    updated_market = load_updated_market(update_dir, stock_ids)
    combined_market, combined_global = combine_feature_history(
        original_market,
        updated_market,
        load_global(data_dir),
        load_global(update_dir),
    )
    combined_market = add_template_keys(combined_market, template_path, target_ids)
    updated_f0 = build_f0_dataset(combined_market, combined_global)
    updated = build_f2_dataset(combined_market, updated_f0)

    ids, dates = load_submission_template(template_path, target_ids)
    required = pd.MultiIndex.from_product([ids, pd.to_datetime(dates)], names=KEYS)
    if not required.isin(pd.MultiIndex.from_frame(updated.loc[:, KEYS])).all():
        raise ValueError("Updated F2 lacks official template stock/date keys")
    return historical, updated


def generate_f2_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    update_dir: str | Path = DEFAULT_UPDATE_DIR,
    template_path: str | Path = DEFAULT_TEMPLATE,
    f0_path: str | Path = DEFAULT_F0_PATH,
    historical_output: str | Path = DEFAULT_HISTORICAL_OUTPUT,
    updated_output: str | Path = DEFAULT_UPDATED_OUTPUT,
) -> tuple[Path, Path]:
    """Write the two generated datasets while leaving raw sources unchanged."""
    historical, updated = build_historical_and_updated_f2(
        data_dir, update_dir, template_path, f0_path
    )
    destinations = (Path(historical_output), Path(updated_output))
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("Historical and updated F2 need distinct output paths")
    for frame, destination in zip((historical, updated), destinations):
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"F2 rows: {len(frame)}; output: {destination}")
    return destinations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--update-dir", type=Path, default=DEFAULT_UPDATE_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--f0-path", type=Path, default=DEFAULT_F0_PATH)
    parser.add_argument("--historical-output", type=Path, default=DEFAULT_HISTORICAL_OUTPUT)
    parser.add_argument("--updated-output", type=Path, default=DEFAULT_UPDATED_OUTPUT)
    args = parser.parse_args()
    generate_f2_datasets(
        args.data_dir,
        args.update_dir,
        args.template,
        args.f0_path,
        args.historical_output,
        args.updated_output,
    )
