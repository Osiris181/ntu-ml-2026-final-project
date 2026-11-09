"""Check F2's completed-session windows, row keys, and fold-only transforms."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from members.zhenci.e4.features_e4 import (
    F2_DATASET_COLUMNS,
    F2_EXTRA_FEATURES,
    build_f2_dataset,
    load_f2_dataset,
)
from members.zhenci.e4.preprocessing_e4 import F2Preprocessor
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES


def stock_history():
    days = pd.bdate_range("2025-01-01", periods=33)
    market = pd.DataFrame([
        {
            "stock_id": stock_id,
            "date": day,
            "etl:adj_open": 99 + 2 * index + index * index / 30 + offset,
            "etl:adj_close": 100 + 2 * index + index * index / 20 + offset,
            "y": index % 2,
        }
        for stock_id, offset in (("A", 0), ("B", 50))
        for index, day in enumerate(days)
    ])
    f0 = market.loc[:, ["stock_id", "date"]].copy()
    for feature in NUMERIC_FEATURES:
        f0[feature] = 0.0
    f0["foreign_ownership"] = f0["date"].map({day: index for index, day in enumerate(days)})
    f0["day_of_week"] = f0["date"].dt.day_name()
    return market, f0, days


class ZhenciE4FeaturesTests(unittest.TestCase):
    def test_full_windows_use_t_minus_one_and_overnight_denominator_from_t_minus_two(self):
        market, f0, days = stock_history()
        built = build_f2_dataset(market.sample(frac=1, random_state=42), f0)
        self.assertEqual(built.columns.tolist(), list(F2_DATASET_COLUMNS))
        self.assertEqual(len(built), len(f0))
        self.assertNotIn("y", built.columns)
        self.assertNotIn("etl:adj_open", built.columns)
        stock = market.loc[market.stock_id.eq("A")].reset_index(drop=True)
        intraday = stock["etl:adj_close"] / stock["etl:adj_open"] - 1
        overnight = stock["etl:adj_open"] / stock["etl:adj_close"].shift(1) - 1

        row5 = built.loc[built.stock_id.eq("A") & built.date.eq(days[5])].iloc[0]
        self.assertAlmostEqual(row5.e4_intraday_mean_5, intraday.iloc[0:5].mean())
        self.assertTrue(np.isnan(row5.e4_overnight_mean_5))
        row20 = built.loc[built.stock_id.eq("A") & built.date.eq(days[20])].iloc[0]
        self.assertAlmostEqual(row20.e4_intraday_mean_20, intraday.iloc[0:20].mean())
        self.assertTrue(np.isnan(row20.e4_overnight_mean_20))

        row21 = built.loc[built.stock_id.eq("A") & built.date.eq(days[21])].iloc[0]
        self.assertAlmostEqual(overnight.iloc[20], stock.loc[20, "etl:adj_open"] / stock.loc[19, "etl:adj_close"] - 1)
        self.assertAlmostEqual(row21.e4_intraday_mean_5, intraday.iloc[16:21].mean())
        self.assertAlmostEqual(row21.e4_overnight_mean_5, overnight.iloc[16:21].mean())
        self.assertAlmostEqual(row21.e4_intraday_mean_20, intraday.iloc[1:21].mean())
        self.assertAlmostEqual(row21.e4_overnight_mean_20, overnight.iloc[1:21].mean())
        self.assertAlmostEqual(
            row21.e4_overnight_minus_intraday_20,
            overnight.iloc[1:21].mean() - intraday.iloc[1:21].mean(),
        )
        self.assertEqual(row21.foreign_ownership, 21)

        changed = market.copy()
        changed.loc[changed.date.ge(days[21]), ["etl:adj_open", "etl:adj_close", "y"]] = 1_000_000
        changed_row = build_f2_dataset(changed, f0).loc[
            lambda frame: frame.stock_id.eq("A") & frame.date.eq(days[21])
        ].iloc[0]
        for feature in F2_EXTRA_FEATURES:
            self.assertAlmostEqual(row21[feature], changed_row[feature])

    def test_missing_or_undefined_prices_keep_rows_and_require_complete_windows(self):
        market, f0, days = stock_history()
        market.loc[market.stock_id.eq("A") & market.date.eq(days[10]), "etl:adj_open"] = np.nan
        market.loc[market.stock_id.eq("B") & market.date.eq(days[10]), "etl:adj_close"] = 0
        built = build_f2_dataset(market, f0)
        self.assertEqual(len(built), len(market))
        row_a11 = built.loc[built.stock_id.eq("A") & built.date.eq(days[11])].iloc[0]
        self.assertTrue(np.isnan(row_a11.e4_intraday_mean_5))
        self.assertTrue(np.isnan(row_a11.e4_overnight_mean_5))
        row_a16 = built.loc[built.stock_id.eq("A") & built.date.eq(days[16])].iloc[0]
        self.assertTrue(np.isfinite(row_a16.e4_intraday_mean_5))
        self.assertTrue(np.isfinite(row_a16.e4_overnight_mean_5))
        row_b12 = built.loc[built.stock_id.eq("B") & built.date.eq(days[12])].iloc[0]
        self.assertTrue(np.isnan(row_b12.e4_overnight_mean_5))
        self.assertTrue(np.isfinite(built.loc[:, F2_EXTRA_FEATURES].to_numpy()[~built.loc[:, F2_EXTRA_FEATURES].isna().to_numpy()]).all())

    def test_updated_history_agrees_with_historical_rows_and_future_key_needs_no_prices(self):
        market, f0, days = stock_history()
        historical_market = market.loc[market.date.le(days[25])].copy()
        historical_f0 = f0.loc[f0.date.le(days[25])].copy()
        historical = build_f2_dataset(historical_market, historical_f0)
        updated = build_f2_dataset(market, f0)
        common = updated.loc[updated.date.le(days[25])].reset_index(drop=True)
        pd.testing.assert_frame_equal(historical, common)

        future_date = days[-1] + pd.offsets.BDay(1)
        placeholders = pd.DataFrame({
            "stock_id": ["A", "B"], "date": [future_date, future_date],
            "etl:adj_open": [np.nan, np.nan], "etl:adj_close": [np.nan, np.nan],
            "y": [np.nan, np.nan],
        })
        future_market = pd.concat([market, placeholders], ignore_index=True)
        future_f0 = pd.concat([
            f0,
            pd.DataFrame({
                **{"stock_id": ["A", "B"], "date": [future_date, future_date]},
                **{field: [np.nan, np.nan] for field in NUMERIC_FEATURES},
                "day_of_week": [future_date.day_name()] * 2,
            }),
        ], ignore_index=True)
        future = build_f2_dataset(future_market, future_f0)
        self.assertEqual(len(future), len(f0) + 2)
        future_rows = future.loc[future.date.eq(future_date)]
        self.assertEqual(set(future_rows.stock_id), {"A", "B"})
        self.assertTrue(np.isfinite(future_rows.loc[:, F2_EXTRA_FEATURES].to_numpy()).all())

    def test_key_and_schema_errors_are_rejected(self):
        market, f0, days = stock_history()
        with self.assertRaisesRegex(ValueError, "same stock/date keys"):
            build_f2_dataset(market, f0.loc[~f0.date.eq(days[0])])
        with self.assertRaisesRegex(ValueError, "same stock/date keys"):
            build_f2_dataset(market.loc[~market.date.eq(days[0])], f0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_f2_dataset(pd.concat([market, market.iloc[[0]]]), f0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f2.csv"
            built = build_f2_dataset(market, f0)
            built.to_csv(path, index=False)
            loaded = load_f2_dataset(path)
            self.assertEqual(loaded.columns.tolist(), list(F2_DATASET_COLUMNS))
            built.assign(y=1).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected F2 dataset columns"):
                load_f2_dataset(path)

    def test_f2_preprocessing_preserves_f0_and_fits_extras_on_train_only(self):
        market, f0, days = stock_history()
        f2 = build_f2_dataset(market, f0)
        train = f2.loc[f2.date.between(days[21], days[27])].copy()
        valid = f2.loc[f2.date.ge(days[28])].copy()
        baseline = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        extended = F2Preprocessor().fit(train)
        baseline_valid = baseline.transform(valid.loc[:, list(F0_FEATURES)])
        extended_valid = extended.transform(valid)
        pd.testing.assert_frame_equal(baseline_valid, extended_valid.loc[:, baseline_valid.columns])
        self.assertTrue(np.isfinite(extended_valid.to_numpy()).all())
        medians = extended.medians_.copy()
        altered = valid.copy()
        altered["e4_intraday_mean_5"] = 1_000_000
        altered["e4_overnight_mean_20"] = np.nan
        self.assertTrue(np.isfinite(extended.transform(altered).to_numpy()).all())
        pd.testing.assert_series_equal(medians, extended.medians_)


if __name__ == "__main__":
    unittest.main()
