"""Sequence-order and boundary checks for the member-owned E12 LSTM."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from members.zhenci.e12.train_e12 import _sequence_data, sequence_end_positions
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES


class SequenceTests(unittest.TestCase):
    def test_complete_windows_follow_stock_sessions_not_calendar_days(self):
        dates = pd.bdate_range("2020-01-01", periods=22).delete(7)
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp("2020-02-03")]))
        dates = dates.sort_values()
        market = pd.DataFrame({"stock_id":"A", "date":dates, "y":np.arange(22)%2})
        f0 = market[["stock_id", "date"]].copy()
        for feature in NUMERIC_FEATURES:
            f0[feature] = np.arange(22, dtype=float)
        f0["day_of_week"] = dates.day_of_week.astype(str)
        train_rows = market.iloc[:20].copy()
        valid_rows = market.iloc[20:].copy()
        fold = SimpleNamespace(train_end=dates[19].date().isoformat(),
                               eval_start=dates[20].date().isoformat(),
                               eval_end=dates[21].date().isoformat())
        pre = BaselinePreprocessor().fit(f0.iloc[:20][list(F0_FEATURES)])
        train, valid, context = _sequence_data(market, f0, fold, ["A"], pre,
                                               train_rows, valid_rows)
        self.assertEqual(train["sequences"].shape, (1,20,19))
        self.assertEqual(valid["sequences"].shape, (2,20,19))
        self.assertEqual(valid["keys"]["date"].tolist(), dates[20:].tolist())
        expected = pre.transform(f0[list(F0_FEATURES)])
        self.assertAlmostEqual(float(valid["sequences"][0,0,0]),
                               float(expected["intraday_return_1d"].iloc[1]), places=6)
        self.assertAlmostEqual(float(valid["sequences"][0,-1,0]),
                               float(expected["intraday_return_1d"].iloc[20]), places=6)
        self.assertAlmostEqual(float(valid["sequences"][1,-1,0]),
                               float(expected["intraday_return_1d"].iloc[21]), places=6)
        self.assertTrue(any(name.startswith("stock_id=") for name in context))
        self.assertTrue(any(name.startswith("day_of_week=") for name in context))

    def test_duplicate_session_is_rejected(self):
        dates = np.array(["2020-01-01", "2020-01-02", "2020-01-02"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            sequence_end_positions(dates, window=2)


if __name__=="__main__":
    unittest.main()
