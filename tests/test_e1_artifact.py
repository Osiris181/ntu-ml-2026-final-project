import unittest

import pandas as pd

from members.zhenci.predict_e1 import require_template_features
from members.zhenci.save_e1 import holdout_training_rows
from shared.splits import HOLDOUT


class E1ArtifactTests(unittest.TestCase):
    def test_template_features_require_every_stock_date_in_template_order(self):
        dates = ["2026-06-01", "2026-06-02"]
        ids = ["A", "B"]
        f0 = pd.DataFrame({
            "stock_id": ["B", "A", "B", "A"],
            "date": pd.to_datetime([dates[1], dates[1], dates[0], dates[0]]),
            "example_feature": [4, 2, 3, 1],
        })
        selected = require_template_features(f0, ids, dates)
        self.assertEqual(selected["example_feature"].tolist(), [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "F0 lacks 1 template stock/date rows"):
            require_template_features(f0.iloc[:-1], ids, dates)

    def test_provisional_refit_uses_shared_holdout_training_window_only(self):
        ids = [str(i) for i in range(100)]
        market = pd.DataFrame([
            {"stock_id": stock_id, "date": pd.Timestamp(day), "y": label}
            for day, label in (("2025-12-31", 0), ("2026-01-02", 1))
            for stock_id in ids
        ])
        train = holdout_training_rows(market, ids[:50], ids)
        self.assertEqual(len(train), 100)
        self.assertEqual(train["date"].max(), pd.Timestamp(HOLDOUT.train_end))
        self.assertTrue(train["y"].eq(0).all())

        changed = market.copy()
        changed.loc[changed["date"].gt(HOLDOUT.train_end), "y"] = 0
        pd.testing.assert_frame_equal(train, holdout_training_rows(changed, ids[:50], ids))

        missing_stock = market.loc[~(
            market["stock_id"].eq(ids[-1]) & market["date"].eq(HOLDOUT.train_end)
        )]
        with self.assertRaisesRegex(ValueError, "all 100 supplied stocks"):
            holdout_training_rows(missing_stock, ids[:50], ids)


if __name__ == "__main__":
    unittest.main()
