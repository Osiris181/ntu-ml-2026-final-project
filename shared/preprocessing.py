"""E1 numeric median/StandardScaler and categorical Unknown/one-hot policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "intraday_return_1d", "daily_return_1d", "momentum_5d", "price_range_1d",
    "relative_volume_20d", "relative_amount_20d", "trade_size_ratio_20d",
    "foreign_ownership", "foreign_net_buy_ratio", "trust_net_buy_ratio",
    "rsi_normalized", "sma_distance", "macd_normalized", "atr_normalized", "bb_width",
    "vix_normalized", "nasdaq_return", "sp500_return", "taiex_return",
)
CATEGORICAL_FEATURES = ("stock_id", "day_of_week")
F0_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


class BaselinePreprocessor:
    """A fresh instance must be fitted on each training fold only.

    StandardScaler here uses the training mean and population standard deviation.
    Unknown test categories yield all-zero columns for that field.
    """

    def fit(self, train: pd.DataFrame) -> "BaselinePreprocessor":
        self._check_columns(train)
        numeric = self._numeric(train)
        self.medians_ = numeric.median()
        if self.medians_.isna().any():
            missing = self.medians_.index[self.medians_.isna()].tolist()
            raise ValueError(f"All-missing training numeric feature(s): {missing}")
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0, 1)
        self.categories_ = {
            field: tuple(sorted(self._category(train[field]).unique()))
            for field in CATEGORICAL_FEATURES
        }
        self.feature_names_ = (*NUMERIC_FEATURES, *(
            f"{field}={value}" for field in CATEGORICAL_FEATURES
            for value in self.categories_[field]
        ))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise RuntimeError("Fit on the training fold before transform")
        self._check_columns(frame)
        numeric = (self._numeric(frame).fillna(self.medians_) - self.means_) / self.scales_
        encoded = {}
        for field in CATEGORICAL_FEATURES:
            values = self._category(frame[field])
            for value in self.categories_[field]:
                encoded[f"{field}={value}"] = values.eq(value).to_numpy(dtype=float)
        categorical = pd.DataFrame(encoded, index=frame.index)
        return pd.concat([numeric, categorical], axis=1).loc[:, self.feature_names_]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @staticmethod
    def _check_columns(frame: pd.DataFrame) -> None:
        absent = set(F0_FEATURES) - set(frame.columns)
        if absent:
            raise ValueError(f"Missing F0 feature columns: {sorted(absent)}")
        if frame["stock_id"].isna().any() or frame["stock_id"].astype(str).eq("").any():
            raise ValueError("Missing stock ID is a row-key error")
        if "date" in frame and frame["date"].isna().any():
            raise ValueError("Missing date is a row-key error")

    @staticmethod
    def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
        return values.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _category(values: pd.Series) -> pd.Series:
        return values.fillna("Unknown").astype(str)
