"""Check E3 training row and weighting policies without training a model."""

import unittest

import numpy as np
import pandas as pd

from members.zhenci.e3.save_e3_regime import chosen_configuration
from members.zhenci.e3.train_e3_regime import CANDIDATES, NUM_BOOST_ROUND, PARAMS, recency_weights, select_training_rows
from shared.preprocessing import F0_FEATURES
from shared.splits import E1_SEEDS


class E3RegimeTests(unittest.TestCase):
    def test_training_universe_preserves_dates_and_target_stocks(self):
        rows = pd.DataFrame({"stock_id": ["A", "B", "A", "B"],
                             "date": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-02-01", "2022-02-01"]),
                             "y": [0, 1, 1, 0]})
        all_rows = select_training_rows(rows, ["A"], "Top100@Y")
        target_rows = select_training_rows(rows, ["A"], "Top50@Y")
        self.assertEqual(len(all_rows), 4)
        self.assertEqual(target_rows["stock_id"].tolist(), ["A", "A"])
        self.assertEqual(target_rows["date"].tolist(), rows.loc[rows.stock_id.eq("A"), "date"].tolist())

    def test_recency_weights_use_only_training_dates(self):
        dates = pd.Series(pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]))
        weights = recency_weights(dates, "2025-12-31", 365)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertIsNone(recency_weights(dates, "2025-12-31", None))
        with self.assertRaisesRegex(ValueError, "exceed"):
            recency_weights(pd.Series(pd.to_datetime(["2026-01-01"])), "2025-12-31", 365)

    def test_only_cv_winning_alternative_can_be_saved(self):
        configs = [{**candidate, "params": PARAMS, "num_boost_round": NUM_BOOST_ROUND,
                    "mean_cv_official_score": 0.1 if index == 1 else 0.0}
                   for index, candidate in enumerate(CANDIDATES)]
        summary = {"model": "XGBoost", "experiment": "E2", "feature_set": "F0",
                   "evaluation_universe": "Top50@Y", "split_version": "1.2",
                   "threshold_rule": "predict 1 when p_up > 0.5, else 0",
                   "feature_order": list(F0_FEATURES), "seeds": list(E1_SEEDS),
                   "configurations": configs, "selected_config_id": CANDIDATES[1]["config_id"],
                   "baseline_cv_score": 0.0, "improvement_over_e1": 0.1}
        self.assertEqual(chosen_configuration(summary), CANDIDATES[1])
        summary["selected_config_id"] = CANDIDATES[0]["config_id"]
        with self.assertRaisesRegex(ValueError, "validation selected the E1 control"):
            chosen_configuration(summary)


if __name__ == "__main__":
    unittest.main()
