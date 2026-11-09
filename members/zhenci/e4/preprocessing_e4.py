"""Training-fold-only preprocessing for F2's five numeric additions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.preprocessing import F0_FEATURES, BaselinePreprocessor

from .features_e4 import F2_EXTRA_FEATURES


class F2Preprocessor:
    """Keep the shared F0 transform and scale F2 additions per training fold."""

    def fit(self, train: pd.DataFrame) -> "F2Preprocessor":
        self.base_ = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        numeric = self._extra_numeric(train)
        self.medians_ = numeric.median()
        if self.medians_.isna().any():
            missing = self.medians_.index[self.medians_.isna()].tolist()
            raise ValueError(f"All-missing F2 training feature(s): {missing}")
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0, 1)
        self.feature_names_ = (*self.base_.feature_names_, *F2_EXTRA_FEATURES)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_"):
            raise RuntimeError("Fit the F2 preprocessor on training rows first")
        baseline = self.base_.transform(frame.loc[:, list(F0_FEATURES)])
        extra = (self._extra_numeric(frame).fillna(self.medians_) - self.means_) / self.scales_
        return pd.concat([baseline, extra], axis=1).loc[:, self.feature_names_]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @staticmethod
    def _extra_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(F2_EXTRA_FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing F2 feature columns: {sorted(missing)}")
        return frame.loc[:, list(F2_EXTRA_FEATURES)].apply(pd.to_numeric, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
