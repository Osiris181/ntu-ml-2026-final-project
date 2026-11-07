"""Verify the member's E2 selection rule without training models."""

import unittest

from members.zhenci.train_e2 import CANDIDATES, select_best_config


class ZhenciE2Tests(unittest.TestCase):
    def test_selects_highest_five_seed_cv_mean_and_breaks_ties_by_grid_order(self):
        summaries = [
            {"config_id": candidate["config_id"], "mean_cv_official_score": score}
            for candidate, score in zip(CANDIDATES, [0.01, 0.03, 0.02, 0.03], strict=True)
        ]
        self.assertEqual(select_best_config(summaries), CANDIDATES[1]["config_id"])
        with self.assertRaisesRegex(ValueError, "prespecified grid"):
            select_best_config(summaries[::-1])


if __name__ == "__main__":
    unittest.main()
