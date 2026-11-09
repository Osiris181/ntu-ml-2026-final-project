"""Check market-relative features, fold preprocessing and threshold choice."""

import unittest

import numpy as np
import pandas as pd

from members.zhenci.e3.evaluate_e3_threshold import THRESHOLDS, choose_threshold
from members.zhenci.e3.features_e3_relative import GLOBAL_FEATURES, RELATIVE_FEATURES, RelativePreprocessor, build_relative_dataset
from shared.preprocessing import BaselinePreprocessor, F0_FEATURES, NUMERIC_FEATURES
from shared.splits import E1_SEEDS


def sample_f0():
    rows = []
    for date, offset in (("2025-01-02", 0), ("2025-01-03", 1), ("2025-01-06", 2)):
        for stock_id, difference in (("A", 0), ("B", 1)):
            row = {name: 0.0 for name in NUMERIC_FEATURES}
            row.update(stock_id=stock_id, date=pd.Timestamp(date), day_of_week="Monday",
                       intraday_return_1d=0.1 * (offset + difference),
                       momentum_5d=0.2 * (offset + difference),
                       relative_volume_20d=1.0 + difference)
            rows.append(row)
    return pd.DataFrame(rows).loc[:, ["date", *F0_FEATURES]]


class E3VariantTests(unittest.TestCase):
    def test_relative_features_use_same_date_lagged_inputs_and_preserve_rows(self):
        f0 = sample_f0()
        built = build_relative_dataset(f0.sample(frac=1, random_state=42))
        self.assertEqual(len(built), len(f0))
        first_a = built.loc[built.stock_id.eq("A") & built.date.eq(pd.Timestamp("2025-01-02"))].iloc[0]
        self.assertAlmostEqual(first_a[RELATIVE_FEATURES[0]], -0.05)
        self.assertAlmostEqual(first_a[RELATIVE_FEATURES[1]], -0.1)
        self.assertAlmostEqual(first_a[RELATIVE_FEATURES[2]], 0.5)
        revised = f0.copy()
        revised.loc[revised.date.eq(pd.Timestamp("2025-01-06")), "momentum_5d"] = 1000
        earlier = build_relative_dataset(revised)
        pd.testing.assert_frame_equal(
            built.loc[built.date.lt(pd.Timestamp("2025-01-06"))].reset_index(drop=True),
            earlier.loc[earlier.date.lt(pd.Timestamp("2025-01-06"))].reset_index(drop=True),
        )

    def test_relative_preprocessor_fits_train_only_and_round_trips(self):
        built = build_relative_dataset(sample_f0())
        train = built.loc[built.date.lt(pd.Timestamp("2025-01-06"))]
        valid = built.loc[built.date.eq(pd.Timestamp("2025-01-06"))]
        processor = RelativePreprocessor(include_relative=True, include_global=False).fit(train)
        before = processor.transform(valid)
        self.assertFalse(set(GLOBAL_FEATURES) & set(before.columns))
        restored = RelativePreprocessor.from_state(processor.to_state())
        pd.testing.assert_frame_equal(before, restored.transform(valid))
        changed = valid.copy()
        changed[RELATIVE_FEATURES[0]] = 1000
        processor.transform(changed)
        pd.testing.assert_series_equal(processor.medians_, restored.medians_)

    def test_f0_control_matches_shared_baseline_preprocessor(self):
        built = build_relative_dataset(sample_f0())
        train = built.loc[built.date.lt(pd.Timestamp("2025-01-06"))]
        valid = built.loc[built.date.eq(pd.Timestamp("2025-01-06"))]
        shared = BaselinePreprocessor().fit(train.loc[:, list(F0_FEATURES)])
        control = RelativePreprocessor(include_relative=False, include_global=True).fit(train)
        pd.testing.assert_frame_equal(
            shared.transform(valid.loc[:, list(F0_FEATURES)]), control.transform(valid)
        )

    def test_threshold_uses_requested_years_and_prefers_half_on_tie(self):
        scores = {threshold: {seed: {"2023": 0.0, "2024": 0.0, "2025": 0.0}
                              for seed in E1_SEEDS} for threshold in THRESHOLDS}
        self.assertEqual(choose_threshold(scores, ("2023",)), 0.5)
        for seed in E1_SEEDS:
            scores[0.45][seed]["2023"] = 0.2
            scores[0.55][seed]["2024"] = 1.0
        self.assertEqual(choose_threshold(scores, ("2023",)), 0.45)
        self.assertEqual(choose_threshold(scores, ("2024",)), 0.55)


if __name__ == "__main__":
    unittest.main()
