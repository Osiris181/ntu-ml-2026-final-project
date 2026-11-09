"""Check member-owned regression target and binary decision rule."""

import unittest

import numpy as np
import pandas as pd

from members.zhenci.e2.train_e2_return_regression import (
    binary_from_regression, comparison_params, training_intraday_returns,
)


class ZhenciE2ReturnRegressionTests(unittest.TestCase):
    def test_training_targets_use_adjusted_training_prices(self):
        rows = pd.DataFrame({
            "etl:adj_open": [100.0, 200.0, 50.0],
            "etl:adj_close": [102.0, 190.0, 50.0],
            "y": [1, 0, 0],
        })
        np.testing.assert_allclose(training_intraday_returns(rows), [0.02, -0.05, 0.0])
        changed = rows.copy()
        changed.loc[0, "y"] = 0
        with self.assertRaisesRegex(ValueError, "disagree"):
            training_intraday_returns(changed)

    def test_positive_expected_return_is_the_up_prediction(self):
        np.testing.assert_array_equal(binary_from_regression(np.array([-0.1, 0.0, 0.01])), [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            binary_from_regression(np.array([np.nan]))

    def test_sampling_changes_both_arms_without_mutating_e1_reference(self):
        original = {"subsample": 1.0, "colsample_bytree": 1.0, "max_depth": 6}
        self.assertEqual(comparison_params(original, False), original)
        sampled = comparison_params(original, True)
        self.assertEqual(sampled["subsample"], 0.8)
        self.assertEqual(sampled["colsample_bytree"], 0.8)
        self.assertEqual(sampled["max_depth"], 6)
        self.assertEqual(original["subsample"], 1.0)


if __name__ == "__main__":
    unittest.main()
