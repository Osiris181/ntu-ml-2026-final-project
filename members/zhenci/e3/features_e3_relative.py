"""Member-owned market-relative features derived only from shared lagged F0.

For prediction date t, all three inputs below already represent completed
stock sessions under shared.features. The same-date peer aggregation therefore
uses historical values, never target-day prices or labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_loader import validate_keys
from shared.preprocessing import F0_FEATURES, BaselinePreprocessor

RELATIVE_FEATURES = (
    "intraday_vs_peer_median_1d",
    "momentum_vs_peer_median_5d",
    "relative_volume_peer_percentile_20d",
)
GLOBAL_FEATURES = ("vix_normalized", "nasdaq_return", "sp500_return", "taiex_return")


def build_relative_dataset(f0: pd.DataFrame) -> pd.DataFrame:
    """Add three cross-stock fields without changing any F0 rows or columns."""
    validate_keys(f0)
    if not pd.api.types.is_datetime64_any_dtype(f0["date"]):
        raise TypeError("F0 dates must be parsed as datetime")
    if set(F0_FEATURES) - set(f0.columns):
        raise ValueError("F0 is missing required baseline features")
    result = f0.loc[:, ["date", *F0_FEATURES]].copy()
    for name in ("intraday_return_1d", "momentum_5d", "relative_volume_20d"):
        result[name] = pd.to_numeric(result[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    by_date = result.groupby("date", sort=False)
    result[RELATIVE_FEATURES[0]] = result["intraday_return_1d"] - by_date["intraday_return_1d"].transform("median")
    result[RELATIVE_FEATURES[1]] = result["momentum_5d"] - by_date["momentum_5d"].transform("median")
    result[RELATIVE_FEATURES[2]] = by_date["relative_volume_20d"].rank(pct=True, method="average")
    validate_keys(result)
    return result.sort_values(["date", "stock_id"]).reset_index(drop=True)


class RelativePreprocessor:
    """Fit shared F0 preprocessing and optional extra numeric fields per fold."""

    def __init__(self, *, include_relative: bool, include_global: bool):
        self.include_relative = include_relative
        self.include_global = include_global

    def fit(self, train: pd.DataFrame) -> "RelativePreprocessor":
        self.base_ = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        self.base_names_ = tuple(name for name in self.base_.feature_names_
                                 if self.include_global or name not in GLOBAL_FEATURES)
        if self.include_relative:
            numeric = self._relative_numeric(train)
            self.medians_ = numeric.median()
            if self.medians_.isna().any():
                raise ValueError("A relative feature is entirely missing in training")
            filled = numeric.fillna(self.medians_)
            self.means_ = filled.mean()
            self.scales_ = filled.std(ddof=0).replace(0, 1)
        self.feature_names_ = (*self.base_names_, *(RELATIVE_FEATURES if self.include_relative else ()))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_"):
            raise RuntimeError("Fit on training rows before transform")
        base = self.base_.transform(frame.loc[:, list(F0_FEATURES)]).loc[:, list(self.base_names_)]
        if not self.include_relative:
            return base
        extra = (self._relative_numeric(frame).fillna(self.medians_) - self.means_) / self.scales_
        return pd.concat([base, extra], axis=1).loc[:, list(self.feature_names_)]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def to_state(self) -> dict:
        if not hasattr(self, "base_"):
            raise RuntimeError("Fit before saving preprocessing")
        state = {"format_version": 1, "include_relative": self.include_relative,
                 "include_global": self.include_global, "base": self.base_.to_state(),
                 "feature_names": list(self.feature_names_)}
        if self.include_relative:
            state["medians"] = [float(self.medians_[name]) for name in RELATIVE_FEATURES]
            state["means"] = [float(self.means_[name]) for name in RELATIVE_FEATURES]
            state["scales"] = [float(self.scales_[name]) for name in RELATIVE_FEATURES]
        return state

    @classmethod
    def from_state(cls, state: dict) -> "RelativePreprocessor":
        if state.get("format_version") != 1 or type(state.get("include_relative")) is not bool or type(state.get("include_global")) is not bool:
            raise ValueError("Invalid relative preprocessing state")
        processor = cls(include_relative=state["include_relative"], include_global=state["include_global"])
        processor.base_ = BaselinePreprocessor.from_state(state["base"])
        processor.base_names_ = tuple(name for name in processor.base_.feature_names_
                                      if processor.include_global or name not in GLOBAL_FEATURES)
        processor.feature_names_ = (*processor.base_names_, *(RELATIVE_FEATURES if processor.include_relative else ()))
        if state.get("feature_names") != list(processor.feature_names_):
            raise ValueError("Saved relative feature order changed")
        if processor.include_relative:
            for field, attr in (("medians", "medians_"), ("means", "means_"), ("scales", "scales_")):
                values = np.asarray(state.get(field), dtype=float)
                if values.shape != (len(RELATIVE_FEATURES),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid relative preprocessing {field}")
                if field == "scales" and (values <= 0).any():
                    raise ValueError("Relative scales must be positive")
                setattr(processor, attr, pd.Series(values, index=RELATIVE_FEATURES))
        return processor

    @staticmethod
    def _relative_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(RELATIVE_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing relative features: {sorted(missing)}")
        return frame.loc[:, list(RELATIVE_FEATURES)].apply(pd.to_numeric, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
