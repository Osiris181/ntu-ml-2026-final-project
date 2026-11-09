"""Guard the paired E4 comparison against changed controls and missing keys."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from members.zhenci.e3.evaluate_e3_market_regime import CANDIDATES
from members.zhenci.e4.train_e4 import (
    E1_CONFIG_ID,
    E3_RULE_ID,
    EXPECTED_E1_PARAMS,
    align_saved_probabilities,
    check_f2_matches_f0,
    checked_source_score,
    load_control_summary,
    load_fixed_rule,
    load_saved_e1_predictions,
)
from shared.preprocessing import F0_FEATURES, NUMERIC_FEATURES
from shared.splits import E1_SEEDS


def _e1_summary() -> dict:
    return {
        "member": "zhenci", "model": "XGBoost", "experiment": "E1",
        "config_id": E1_CONFIG_ID, "feature_set": "F0", "feature_order": list(F0_FEATURES),
        "training_universe": "Top100@Y", "evaluation_universe": "Top50@Y",
        "split_version": "1.2", "seeds": list(E1_SEEDS),
        "threshold_rule": "predict 1 when p_up > 0.5, else 0",
        "params": EXPECTED_E1_PARAMS.copy(), "num_boost_round": 100,
        "fold_official_scores_by_seed": {
            str(seed): {year: 0.0 for year in ("2023", "2024", "2025")}
            for seed in E1_SEEDS
        },
    }


class E4RunnerInputTests(unittest.TestCase):
    def test_original_unsampled_e1_configuration_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            original = _e1_summary()
            path.write_text(json.dumps(original), encoding="utf-8")
            self.assertEqual(load_control_summary(path)["params"], EXPECTED_E1_PARAMS)

            sampled = _e1_summary()
            sampled["params"]["subsample"] = 0.8
            path.write_text(json.dumps(sampled), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-sample"):
                load_control_summary(path)

            wrong_rounds = _e1_summary()
            wrong_rounds["num_boost_round"] = 200
            path.write_text(json.dumps(wrong_rounds), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-sample"):
                load_control_summary(path)

    def test_e3_rule_is_pinned_to_predeclared_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            comparisons = [{**candidate, "mean_cv_official_score": float(index)}
                           for index, candidate in enumerate(CANDIDATES)]
            summary = {
                "source_model_configuration": E1_CONFIG_ID,
                "seeds": list(E1_SEEDS),
                "comparisons": comparisons,
                "selected_same_period_config_id": E3_RULE_ID,
            }
            path.write_text(json.dumps(summary), encoding="utf-8")
            rule, _ = load_fixed_rule(path, E1_CONFIG_ID)
            self.assertEqual((rule["negative"], rule["nonnegative"], rule["missing"]),
                             (0.525, 0.475, 0.5))

            summary["selected_same_period_config_id"] = CANDIDATES[-2]["config_id"]
            # Make this alternative the recorded best to isolate E4's pinning.
            comparisons[-2]["mean_cv_official_score"] = 100.0
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed E4 comparison"):
                load_fixed_rule(path, E1_CONFIG_ID)

    def test_saved_e1_probabilities_and_exact_keys_are_checked(self):
        summary = _e1_summary()
        rows = []
        for year in ("2023", "2024", "2025"):
            for seed in E1_SEEDS:
                rows.append({
                    "member": "zhenci", "model": "XGBoost", "experiment": "E1",
                    "config_id": E1_CONFIG_ID, "fold": year, "seed": seed,
                    "stock_id": "2330", "date": f"{year}-01-03",
                    "p_up": 0.6, "predicted_y": 1,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            source = load_saved_e1_predictions(path, summary)
            valid = pd.DataFrame({"stock_id": ["2330"],
                                  "date": [pd.Timestamp("2023-01-03")]})
            np.testing.assert_array_equal(
                align_saved_probabilities(valid, source, "2023", 42), np.array([0.6])
            )
            extra_valid = pd.concat([
                valid,
                pd.DataFrame({"stock_id": ["2317"], "date": [pd.Timestamp("2023-01-03")]})
            ], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                align_saved_probabilities(extra_valid, source, "2023", 42)

            corrupted = pd.DataFrame(rows)
            corrupted.loc[0, "predicted_y"] = 0
            corrupted.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_saved_e1_predictions(path, summary)

            corrupted = pd.DataFrame(rows)
            corrupted.loc[0, "p_up"] = 1.1
            corrupted.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "probabilities"):
                load_saved_e1_predictions(path, summary)

    def test_reused_control_scores_must_reproduce_archived_scores(self):
        recorded = {"42": {"2023": 0.00058}}
        checked_source_score(recorded, 42, "2023", 0.00058, "E1")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            checked_source_score(recorded, 42, "2023", 0.00059, "E1")

    def test_f2_cannot_change_shared_f0_inputs(self):
        f0 = pd.DataFrame({
            "stock_id": ["2330", "2317"],
            "date": [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-03")],
            **{field: [0.1, np.nan] for field in NUMERIC_FEATURES},
            "day_of_week": ["Tuesday", "Tuesday"],
        })
        f2 = f0.iloc[::-1].reset_index(drop=True)
        check_f2_matches_f0(f2, f0)

        altered = f2.copy()
        altered.loc[altered["stock_id"].eq("2330"), "taiex_return"] = 0.2
        with self.assertRaisesRegex(ValueError, "taiex_return"):
            check_f2_matches_f0(altered, f0)

        wrong_key = f2.copy()
        wrong_key.loc[0, "stock_id"] = "9999"
        with self.assertRaisesRegex(ValueError, "stock/date keys"):
            check_f2_matches_f0(wrong_key, f0)


if __name__ == "__main__":
    unittest.main()
