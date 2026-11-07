"""Check causal F1 stock-history features and fold-fitted transformations."""

import unittest

import numpy as np
import pandas as pd

from members.zhenci.features_e2 import EXTRA_FEATURES, F1Preprocessor, build_f1_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES


def stock_history():
    days = pd.bdate_range("2025-01-01", periods=16)
    market = pd.DataFrame([
        {
            "stock_id": stock_id,
            "date": day,
            "etl:adj_close": 100 + 3 * index + index * index / 10 + offset,
            "etl:adj_open": 99 + 3 * index + index * index / 10 + offset,
        }
        for stock_id, offset in (("A", 0), ("B", 20))
        for index, day in enumerate(days)
    ])
    f0 = market.loc[:, ["stock_id", "date"]].copy()
    for feature in NUMERIC_FEATURES:
        f0[feature] = 0.0
    f0["day_of_week"] = f0["date"].dt.day_name()
    return market, f0, days


class ZhenciE2FeaturesTests(unittest.TestCase):
    def test_new_features_use_only_completed_stock_sessions(self):
        market, f0, days = stock_history()
        built = build_f1_dataset(market.sample(frac=1, random_state=42), f0)
        self.assertEqual(len(built), len(f0))
        self.assertTrue(set(EXTRA_FEATURES).issubset(built.columns))
        self.assertEqual(built.loc[built.stock_id.eq("A") & built.date.eq(days[0]), "momentum_10d"].isna().all(), True)
        stock_a = market.loc[market.stock_id.eq("A")].reset_index(drop=True)
        row = built.loc[built.stock_id.eq("A") & built.date.eq(days[12])].iloc[0]
        close = stock_a["etl:adj_close"]
        self.assertAlmostEqual(row.prior_overnight_gap, stock_a.loc[11, "etl:adj_open"] / close.loc[10] - 1)
        self.assertAlmostEqual(row.momentum_10d, close.loc[11] / close.loc[1] - 1)
        returns = np.array([close.loc[i] / close.loc[i - 1] - 1 for i in range(7, 12)])
        self.assertAlmostEqual(row.close_volatility_5d, returns.std(ddof=0))

        changed = market.copy()
        changed.loc[changed.date.ge(days[12]), ["etl:adj_open", "etl:adj_close"]] = 1_000_000
        revised = build_f1_dataset(changed, f0)
        revised_row = revised.loc[revised.stock_id.eq("A") & revised.date.eq(days[12])].iloc[0]
        for feature in EXTRA_FEATURES:
            self.assertAlmostEqual(row[feature], revised_row[feature])

    def test_f1_keeps_identical_f0_preprocessing_and_training_only_statistics(self):
        market, f0, days = stock_history()
        f1 = build_f1_dataset(market, f0)
        train = f1.loc[f1.date.lt(days[12])]
        valid = f1.loc[f1.date.ge(days[12])]
        baseline = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        extended = F1Preprocessor().fit(train)
        baseline_valid = baseline.transform(valid.loc[:, list(F0_FEATURES)])
        extended_valid = extended.transform(valid)
        pd.testing.assert_frame_equal(baseline_valid, extended_valid.loc[:, baseline_valid.columns])
        self.assertTrue(np.isfinite(extended_valid.to_numpy()).all())
        medians = extended.medians_.copy()
        altered = valid.copy()
        altered["momentum_10d"] = 1_000_000
        extended.transform(altered)
        pd.testing.assert_series_equal(medians, extended.medians_)


if __name__ == "__main__":
    unittest.main()
