"""Validate E3 regime threshold behavior and chronological rule selection."""

import unittest

import numpy as np

from members.zhenci.e3.evaluate_e3_market_regime import CANDIDATES, choose_candidate, regime_predictions
from members.zhenci.e3.predict_e3_market_regime import selected_rule
from shared.splits import E1_SEEDS


class MarketRegimeThresholdTests(unittest.TestCase):
    def test_regime_rule_uses_prior_value_and_missing_fallback(self):
        candidate = CANDIDATES[3]  # 0.525 after negative TAIEX, 0.5 otherwise.
        probabilities = np.array([0.51, 0.53, 0.51, 0.51, 0.5])
        prior_taiex = np.array([-0.01, -0.01, 0.01, np.nan, 0.0])
        np.testing.assert_array_equal(
            regime_predictions(probabilities, prior_taiex, candidate),
            np.array([0, 1, 1, 1, 0]),
        )
        # A fixed cutoff must stay fixed even on dates with no eligible TAIEX.
        self.assertEqual(
            regime_predictions(np.array([0.49]), np.array([np.nan]), CANDIDATES[1]).item(),
            1,
        )

    def test_walk_forward_selection_ignores_later_year_scores(self):
        scores = {candidate["config_id"]: {seed: {year: 0.0 for year in ("2023", "2024", "2025")}
                                             for seed in E1_SEEDS} for candidate in CANDIDATES}
        favored = CANDIDATES[3]["config_id"]
        later_only = CANDIDATES[4]["config_id"]
        for seed in E1_SEEDS:
            scores[favored][seed]["2023"] = 0.01
            scores[later_only][seed]["2025"] = 1.0
        self.assertEqual(choose_candidate(scores, ("2023",)), favored)
        self.assertEqual(choose_candidate(scores, ("2023", "2024")), favored)
        self.assertEqual(choose_candidate(scores, ("2023", "2024", "2025")), later_only)

    def test_exporter_rejects_wrong_model_or_changed_threshold_rule(self):
        comparisons = [{**candidate, "mean_cv_official_score": float(index)}
                       for index, candidate in enumerate(CANDIDATES)]
        summary = {"source_model_configuration": "e1_test",
                   "comparisons": comparisons,
                   "selected_same_period_config_id": CANDIDATES[-1]["config_id"]}
        self.assertEqual(selected_rule(summary, "e1_test"), CANDIDATES[-1])
        with self.assertRaisesRegex(ValueError, "different E1 configurations"):
            selected_rule(summary, "other_model")
        comparisons[-1]["missing"] = 0.475
        with self.assertRaisesRegex(ValueError, "definitions differ"):
            selected_rule(summary, "e1_test")


if __name__ == "__main__":
    unittest.main()
