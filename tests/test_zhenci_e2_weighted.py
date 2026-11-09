"""Check score-aligned training weights without fitting XGBoost."""

import unittest

import numpy as np
import pandas as pd

from members.zhenci.e2.train_e2_weighted import training_return_weights


class ZhenciE2WeightedTests(unittest.TestCase):
    def test_training_weights_follow_absolute_realized_return(self):
        rows = pd.DataFrame({
            "etl:adj_open": [100.0, 100.0, 100.0],
            "etl:adj_close": [101.0, 95.0, 100.0],
            "y": [1, 0, 0],
        })
        weights = training_return_weights(rows)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[1] / weights[0], 5.0)
        self.assertEqual(weights[2], 0.0)
        changed = rows.copy()
        changed.loc[0, "y"] = 0
        with self.assertRaisesRegex(ValueError, "disagree"):
            training_return_weights(changed)
        self.assertTrue(np.isfinite(weights).all())


if __name__ == "__main__":
    unittest.main()
