"""Checks that E12 final-refit and template windows match the CV sequence rule."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from members.zhenci.e12.sequences import complete_training_keys, sequence_batch
from members.zhenci.e12.train_e12 import _sequence_data
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES


class SubmissionSequenceTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-02", periods=26).delete(6)
        parts = []
        for stock_id, offset in (("A", 0), ("B", 100)):
            part = pd.DataFrame({"stock_id": stock_id, "date": dates})
            for feature in NUMERIC_FEATURES:
                part[feature] = np.arange(len(dates), dtype=float) + offset
            part["day_of_week"] = dates.day_name()
            parts.append(part)
        self.f0 = pd.concat(parts, ignore_index=True)
        self.pre = BaselinePreprocessor().fit(self.f0.loc[
            self.f0["date"].le(dates[19]), list(F0_FEATURES)])
        self.dates = dates

    def test_complete_keys_exclude_first_nineteen_sessions_per_stock(self):
        keys = complete_training_keys(self.f0[["stock_id", "date"]])
        self.assertEqual(len(keys), 12)
        self.assertEqual(keys.groupby("stock_id")["date"].min().to_dict(),
                         {"A": self.dates[19], "B": self.dates[19]})

    def test_requested_windows_preserve_order_stock_and_cutoff(self):
        keys = pd.DataFrame({"stock_id": ["B", "A", "B"],
                             "date": [self.dates[21], self.dates[20], self.dates[20]]})
        batch, context = sequence_batch(self.f0, keys, self.pre)
        self.assertTrue(batch["keys"].equals(keys))
        self.assertEqual(batch["sequences"].shape, (3, 20, 19))
        self.assertTrue(any(x.startswith("stock_id=") for x in context))
        self.assertTrue(any(x.startswith("day_of_week=") for x in context))
        transformed = self.pre.transform(self.f0[list(F0_FEATURES)])
        numeric = "intraday_return_1d"
        self.assertAlmostEqual(float(batch["sequences"][0, 0, 0]),
                               float(transformed.loc[25 + 2, numeric]), places=6)
        self.assertAlmostEqual(float(batch["sequences"][0, -1, 0]),
                               float(transformed.loc[25 + 21, numeric]), places=6)
        self.assertAlmostEqual(float(batch["sequences"][1, -1, 0]),
                               float(transformed.loc[20, numeric]), places=6)
        changed = self.f0.copy()
        changed.loc[changed["date"].gt(self.dates[21]), numeric] = 99999
        future_changed, _ = sequence_batch(changed, keys, self.pre)
        np.testing.assert_array_equal(batch["sequences"], future_changed["sequences"])

    def test_missing_required_history_is_rejected(self):
        keys = pd.DataFrame({"stock_id": ["A"], "date": [self.dates[18]]})
        with self.assertRaisesRegex(ValueError, "Fewer than 20"):
            sequence_batch(self.f0, keys, self.pre)

    def test_submission_builder_matches_existing_cv_builder(self):
        market = self.f0[["stock_id", "date"]].copy()
        market["y"] = np.arange(len(market)) % 2
        train_rows = market.loc[market["date"].le(self.dates[19])].copy()
        valid_rows = market.loc[market["date"].between(self.dates[20], self.dates[21])].copy()
        fold = SimpleNamespace(train_end=self.dates[19].date().isoformat(),
                               eval_start=self.dates[20].date().isoformat(),
                               eval_end=self.dates[21].date().isoformat())
        cv_train, cv_valid, cv_context = _sequence_data(
            market, self.f0, fold, ["A", "B"], self.pre, train_rows, valid_rows)
        f0_source = self.f0.loc[self.f0["date"].le(self.dates[21])]
        for reference in (cv_train, cv_valid):
            built, context = sequence_batch(f0_source, reference["keys"], self.pre)
            self.assertEqual(context, cv_context)
            self.assertTrue(built["keys"].equals(reference["keys"]))
            np.testing.assert_array_equal(built["sequences"], reference["sequences"])
            np.testing.assert_array_equal(built["context"], reference["context"])


if __name__ == "__main__":
    unittest.main()
