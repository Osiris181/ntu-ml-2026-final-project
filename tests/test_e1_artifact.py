import tempfile
import unittest
from pathlib import Path

import pandas as pd

from members.zhenci.predict_e1 import require_template_features
from members.zhenci.save_e1 import save_e1


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

    def test_provisional_refit_rejects_holdout_dates_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "artifact"
            with self.assertRaisesRegex(ValueError, "training cutoff"):
                save_e1(Path(temp) / "missing_summary.json", "2026-05-31", 42, output, Path(temp), Path(temp))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
