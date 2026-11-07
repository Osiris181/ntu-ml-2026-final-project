"""Member-owned F1 stock-history features and fold-fitted preprocessing.

Every added value for date t uses stock sessions no later than t-1. These
features are experimental; the shared 21-field F0 dataset is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_loader import KEYS, validate_keys
from shared.features import COL
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor


EXTRA_FEATURES = (
    "prior_overnight_gap",
    "momentum_10d",
    "close_volatility_5d",
)


def build_f1_dataset(market: pd.DataFrame, f0: pd.DataFrame) -> pd.DataFrame:
    """Add three prior-stock-session fields to each existing F0 row."""
    validate_keys(market)
    validate_keys(f0)
    if not pd.api.types.is_datetime64_any_dtype(market["date"]):
        raise TypeError("Market dates must be parsed as datetime")
    if not pd.api.types.is_datetime64_any_dtype(f0["date"]):
        raise TypeError("F0 dates must be parsed as datetime")
    if set(F0_FEATURES) - set(f0.columns):
        raise ValueError("F0 is missing required baseline features")
    if {COL["close"], COL["open"]} - set(market.columns):
        raise ValueError("Market is missing adjusted open or close")

    work = market.sort_values(["stock_id", "date"]).copy()
    group = work.groupby("stock_id", sort=False)
    close = group[COL["close"]]
    previous_open = group[COL["open"]].shift(1)
    previous_close = close.shift(1)
    two_sessions_ago_close = close.shift(2)
    eleven_sessions_ago_close = close.shift(11)

    extra = work.loc[:, KEYS].copy()
    extra["prior_overnight_gap"] = previous_open / two_sessions_ago_close - 1
    extra["momentum_10d"] = previous_close / eleven_sessions_ago_close - 1
    work["_close_return"] = group[COL["close"]].transform(lambda values: values / values.shift(1) - 1)
    extra["close_volatility_5d"] = work.groupby("stock_id", sort=False)["_close_return"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).std(ddof=0)
    )
    extra.loc[:, list(EXTRA_FEATURES)] = extra.loc[:, list(EXTRA_FEATURES)].replace([np.inf, -np.inf], np.nan)

    combined = f0.merge(extra, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if len(combined) != len(market) or not combined["_merge"].eq("both").all():
        raise ValueError("F1 requires exactly the same stock/date keys as market and F0")
    combined = combined.drop(columns="_merge")
    validate_keys(combined)
    return combined.sort_values(["date", "stock_id"]).reset_index(drop=True)


class F1Preprocessor:
    """Preserve baseline F0 transforms and fit added numeric fields per fold."""

    def fit(self, train: pd.DataFrame) -> "F1Preprocessor":
        self.base_ = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        numeric = self._extra_numeric(train)
        self.medians_ = numeric.median()
        if self.medians_.isna().any():
            missing = self.medians_.index[self.medians_.isna()].tolist()
            raise ValueError(f"All-missing F1 training feature(s): {missing}")
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0, 1)
        self.feature_names_ = (*self.base_.feature_names_, *EXTRA_FEATURES)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_"):
            raise RuntimeError("Fit the F1 preprocessor on training rows first")
        baseline = self.base_.transform(frame.loc[:, list(F0_FEATURES)])
        extra = (self._extra_numeric(frame).fillna(self.medians_) - self.means_) / self.scales_
        return pd.concat([baseline, extra], axis=1).loc[:, self.feature_names_]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @staticmethod
    def _extra_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXTRA_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing F1 feature columns: {sorted(missing)}")
        return frame.loc[:, list(EXTRA_FEATURES)].apply(pd.to_numeric, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
